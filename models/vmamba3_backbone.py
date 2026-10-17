import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.constants import *
from models.vmamba_backbone import VisualMamba


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    # f(x) = 1 + x for x >= 0, 1 / (1 - x) for x < 0
    return x.clamp_min(0) + torch.reciprocal(1 - x.clamp_max(0))


def _rotate_pairs(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate the first 2*S dims of x (interleaved pairs) by `angle` (..., S); remaining dims untouched."""
    S = angle.shape[-1]
    rot, rest = x[..., : 2 * S], x[..., 2 * S:]
    x1, x2 = rot[..., 0::2], rot[..., 1::2]
    cos, sin = angle.cos(), angle.sin()
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return torch.cat([out, rest], dim=-1)


class Mamba3SSM(nn.Module):
    """
    One scan direction of the Mamba-3 mixer (exponential-trapezoidal SSM, complex state via
    data-dependent RoPE, optional MIMO). Pure PyTorch; operates on already-projected x and z.

        h_t = a_t h_{t-1} + b_t v_{t-1} + g_t v_t,   v_t = sum_r B_t^(r) x_t^(r)^T
        y_t^(r) = C_t^(r)^T h_t
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        d_state: int,
        headdim: int,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
    ):
        super().__init__()
        assert d_inner % headdim == 0, "d_inner must be divisible by headdim"
        assert rope_fraction in (0.5, 1.0)

        self.d_state = d_state
        self.headdim = headdim
        self.nheads = d_inner // headdim
        self.is_mimo = is_mimo
        self.rank = mimo_rank if is_mimo else 1
        self.A_floor = A_floor

        split = int(d_state * rope_fraction)
        split -= split % 2
        self.num_rope_angles = split // 2
        assert self.num_rope_angles > 0, "d_state too small for RoPE"

        R, N, H, P = self.rank, d_state, self.nheads, headdim

        # [B, C, dd_dt, dd_A, trap, angles]
        self.proj = nn.Linear(d_model, 2 * R * N + 3 * H + self.num_rope_angles, bias=False)

        dt = torch.exp(torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.dt_bias._no_weight_decay = True

        self.B_norm = RMSNorm(N)
        self.C_norm = RMSNorm(N)
        self.B_bias = nn.Parameter(torch.ones(H, R, N))
        self.C_bias = nn.Parameter(torch.ones(H, R, N))

        self.D = nn.Parameter(torch.ones(H))
        self.D._no_weight_decay = True

        if is_mimo:
            self.mimo_x = nn.Parameter(torch.ones(H, R, P) / R)
            self.mimo_z = nn.Parameter(torch.ones(H, R, P))
            self.mimo_o = nn.Parameter(torch.ones(H, R, P) / R)

    # ---- scans: all inputs fp32 ----
    # ADT/beta/gamma: (b,l,H); B,C: (b,l,H,R,N); x: (b,l,H,R,P) -> y: (b,l,H,R,P)
    @staticmethod
    def scan_recurrent(ADT, beta, gamma, B, C, x):
        b, l, H, R, N = B.shape
        P = x.shape[-1]
        h = x.new_zeros(b, H, N, P)
        v_prev = x.new_zeros(b, H, N, P)
        ys = []
        for t in range(l):
            v = torch.einsum("bhrn,bhrp->bhnp", B[:, t], x[:, t])
            alpha = ADT[:, t].exp()[..., None, None]
            h = alpha * h + beta[:, t][..., None, None] * v_prev + gamma[:, t][..., None, None] * v
            ys.append(torch.einsum("bhrn,bhnp->bhrp", C[:, t], h))
            v_prev = v
        return torch.stack(ys, dim=1)

    @staticmethod
    def scan_parallel(ADT, beta, gamma, B, C, x):
        # y = (L o C B^T) x with L = semiseparable(alpha) @ 2-band(gamma, beta)  (paper eq. 7)
        l = ADT.shape[1]
        c = ADT.permute(0, 2, 1).cumsum(-1)                     # (b,H,l)
        g = gamma.permute(0, 2, 1)
        bt = beta.permute(0, 2, 1)

        c_next = torch.cat([c[..., 1:], c[..., -1:]], dim=-1)   # c_{s+1}
        b_next = torch.cat([bt[..., 1:], torch.zeros_like(bt[..., :1])], dim=-1)  # beta_{s+1}

        tril = torch.tril(torch.ones(l, l, dtype=torch.bool, device=ADT.device))
        strict = torch.tril(torch.ones(l, l, dtype=torch.bool, device=ADT.device), diagonal=-1)

        d1 = (c[..., :, None] - c[..., None, :]).masked_fill(~tril, -float("inf"))
        d2 = (c[..., :, None] - c_next[..., None, :]).masked_fill(~strict, -float("inf"))
        L = d1.exp() * g[..., None, :] + d2.exp() * b_next[..., None, :]      # (b,H,t,s)

        S = torch.einsum("bthrn,bshqn->bhrqts", C, B) * L[:, :, None, None]
        return torch.einsum("bhrqts,bshqp->bthrp", S, x)

    def forward(self, u, x, z, parallel: bool = True):
        """u: (b,l,d_model) normalized input; x,z: (b,l,H*P). Returns gated output (b,l,H*P)."""
        b, l, _ = u.shape
        H, P, R, N, S = self.nheads, self.headdim, self.rank, self.d_state, self.num_rope_angles

        p = self.proj(u)
        with torch.autocast(device_type=u.device.type, enabled=False):
            p = p.float()
            Bp, Cp, dd_dt, dd_A, trap, ang = torch.split(p, [R * N, R * N, H, H, H, S], dim=-1)

            A = (-heavy_tail_activation(dd_A)).clamp(max=-self.A_floor)
            DT = F.softplus(dd_dt + self.dt_bias)
            ADT = A * DT
            lam = torch.sigmoid(trap)
            gamma = lam * DT
            beta = (1 - lam) * DT * ADT.exp()

            Bm = self.B_norm(Bp.reshape(b, l, 1, R, N)) + self.B_bias   # (b,l,H,R,N)
            Cm = self.C_norm(Cp.reshape(b, l, 1, R, N)) + self.C_bias

            # data-dependent RoPE: cumulative rotation (prod of R_i^T) on both B and C
            cum = torch.cumsum(DT[..., None] * ang[:, :, None, :], dim=1)    # (b,l,H,S)
            cum = -cum[:, :, :, None, :]
            Bm = _rotate_pairs(Bm, cum)
            Cm = _rotate_pairs(Cm, cum)

            xh = x.float().reshape(b, l, H, 1, P)
            zh = z.float().reshape(b, l, H, 1, P)
            if self.is_mimo:
                xh = xh * self.mimo_x.float()
                zh = zh * self.mimo_z.float()

            scan = self.scan_parallel if parallel else self.scan_recurrent
            y = scan(ADT, beta, gamma, Bm, Cm, xh)
            y = y + self.D.view(1, 1, H, 1, 1) * xh
            y = y * F.silu(zh)

            if self.is_mimo:
                y = (y * self.mimo_o.float()).sum(dim=3)
            else:
                y = y.squeeze(3)
        return y.reshape(b, l, H * P)


class Vim3Block(nn.Module):
    """Bidirectional Mamba-3 block in the layout of VimBlock (shared x/z projection, per-direction SSM)."""

    def __init__(
        self,
        d_model: int,
        expand: int,
        ssm_dim: int,
        headdim: int = MAMBA3_HEADDIM,
        is_mimo: bool = MAMBA3_MIMO,
        mimo_rank: int = MAMBA3_MIMO_RANK,
        rope_fraction: float = MAMBA3_ROPE_FRACTION,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.to_xz = nn.Linear(d_model, 2 * expand, bias=False)
        kw = dict(d_model=d_model, d_inner=expand, d_state=ssm_dim, headdim=headdim,
                  is_mimo=is_mimo, mimo_rank=mimo_rank, rope_fraction=rope_fraction)
        self.ssm_fwd = Mamba3SSM(**kw)
        self.ssm_bwd = Mamba3SSM(**kw)
        self.to_out = nn.Linear(expand, d_model, bias=False)
        nn.init.zeros_(self.to_out.weight)

    def forward(self, x: torch.Tensor):
        u = self.norm(x)
        xp, z = self.to_xz(u).chunk(2, dim=-1)

        y_f = self.ssm_fwd(u, xp, z)
        y_b = self.ssm_bwd(u.flip(1), xp.flip(1), z.flip(1)).flip(1)
        return x + self.to_out((y_f + y_b).to(x.dtype))


class VisualMamba3(VisualMamba):
    """VisualMamba with Mamba-3 blocks; patch embed, masking and feature API are inherited."""

    def __init__(
        self,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        num_classes=NUM_CLASSES,
        embed_dim=VMAMBA_EMBED_DIM,
        depth=VMAMBA_DEPTH,
        ssm_dim=SSM_DIM,
        expand_dim=EXPAND_DIM,
        learning_rate=LR,
        mask_ratio=MASK_RATIO,
        use_cls_token: bool = False,
        headdim=MAMBA3_HEADDIM,
        is_mimo=MAMBA3_MIMO,
        mimo_rank=MAMBA3_MIMO_RANK,
        rope_fraction=MAMBA3_ROPE_FRACTION,
        **kwargs,
    ):
        super().__init__(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
            embed_dim=embed_dim, depth=0, ssm_dim=ssm_dim, expand_dim=expand_dim,
            learning_rate=learning_rate, mask_ratio=mask_ratio, use_cls_token=use_cls_token, **kwargs,
        )
        self.backbone = nn.ModuleList([
            Vim3Block(embed_dim, expand_dim, ssm_dim, headdim, is_mimo, mimo_rank, rope_fraction)
            for _ in range(depth)
        ])
