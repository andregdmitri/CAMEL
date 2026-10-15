import torch
import torch.nn as nn

from config.constants import IMG_SIZE


class MaskedCNNStudent(nn.Module):
    """
    Encoder-only CNN student for RETFound feature distillation with MAE-style masking.

    Random patches (on the teacher's token grid) are zeroed in pixel space; the CNN feature map is
    resized to that grid so tokens/ids_keep align with the teacher. API matches `VisualMamba`:

        forward_features(x, return_pooled=True, apply_mask=False)
            -> pooled (B, C)  or  (tokens, mask, ids_keep, ids_restore)
    """

    def __init__(self, backbone, img_size=IMG_SIZE, grid_patch_size=16, mask_ratio=0.0):
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = mask_ratio
        self.grid_patch_size = grid_patch_size
        self.grid = img_size // grid_patch_size

        # Probe the real channel count of the feature map (num_features is not always reliable).
        was_training = self.training
        self.eval()
        with torch.no_grad():
            self.embed_dim = self._feature_map(torch.zeros(1, 3, img_size, img_size)).shape[1]
        self.train(was_training)

    def _feature_map(self, x):
        return self.backbone.forward_features(x)  # (B, C, h, w)

    def _random_mask(self, B, device):
        N = self.grid * self.grid
        len_keep = max(1, int(N * (1 - self.mask_ratio)))
        ids_shuffle = torch.rand(B, N, device=device).argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.ones(B, N, device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return mask, ids_keep, ids_restore

    def forward_features(self, x, return_pooled=True, apply_mask=False):
        mask = ids_keep = ids_restore = None
        if apply_mask and self.mask_ratio > 0.0:
            B = x.size(0)
            mask, ids_keep, ids_restore = self._random_mask(B, x.device)
            pix = mask.view(B, 1, self.grid, self.grid).to(x.dtype)
            pix = nn.functional.interpolate(pix, size=x.shape[-2:], mode="nearest")
            x = x * (1.0 - pix)

        feats = self._feature_map(x)
        if feats.shape[-1] != self.grid or feats.shape[-2] != self.grid:
            feats = nn.functional.interpolate(feats, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        tokens = feats.flatten(2).transpose(1, 2)  # (B, grid*grid, C)

        if return_pooled:
            return tokens.mean(dim=1)
        return tokens, mask, ids_keep, ids_restore

    def forward(self, x):
        return self.forward_features(x, return_pooled=True)
