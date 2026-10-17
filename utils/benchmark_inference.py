"""Benchmark wall-clock inference time for every model in the LEMUR results table.

Covers all 11 modes of `results/results_best_ckpt.csv`, i.e. both the plain and
the distilled variant of each family:

    mobilenet / dmobilenet, efficientnet / defficientnet, tinyvit / dtinyvit,
    unet / dunet, vmamba / dvmamba, retfound_finetune (teacher, no distilled form)

Each model is warmed up `--warmup` times and then timed `--runs` times (default
5). Latency is measured at `--batch-size` (default 1) and throughput at
`--throughput-batch` (default 32). Timing uses `time.perf_counter` around the
forward call, with `torch.cuda.synchronize()` on both sides when on GPU, so the
measurement includes the full device execution rather than just kernel launch.

Loading recipes (there are four, see `docs`/history for why):

  * plain light models  -> `train/train.py::MODEL_REGISTRY` classifier modules,
    weights from `ckpt["state_dict"]` (`backbone.*` / `classifier.*`).
  * distilled `d*` modes -> student backbone built per mode (same kwargs as
    `train/head.py`) then wrapped in `VMambaHeadTask`; the checkpoint holds
    `backbone.*` (+ `head.*`, `loss_fn.weight`), so `strict=False`.
  * `retfound_finetune` -> `RETFoundTask.load_from_checkpoint`. NOTE the
    constructor always reads the 3.95 GB RETFound_cfp_weights.pth, so the model
    is built ONCE and later checkpoints are hot-swapped into it.
  * everything is `map_location="cpu"` first, then moved to `--device`.

Usage:
    # all 11 models, one representative checkpoint each, 5 timed runs on CPU
    python utils/benchmark_inference.py

    # same but on GPU, with FLOPs/params cross-check
    python utils/benchmark_inference.py --device cuda --flops

    # one row per (mode, dataset)
    python utils/benchmark_inference.py --granularity dataset

    # every checkpoint in the table (220 rows, retfound ones auto-skipped if missing)
    python utils/benchmark_inference.py --granularity checkpoint
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from collections import OrderedDict

import torch

# Make the repo root importable when this file is run as `python utils/benchmark_inference.py`.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.constants import (  # noqa: E402
    IMG_SIZE,
    IN_CHANS,
    LR,
    NUM_CLASSES,
    PATCH_SIZE,
    TINYVIT_MODEL,
    VMAMBA_DEPTH,
    VMAMBA_EMBED_DIM,
)
from models.efficientnet import EfficientNetClassifier  # noqa: E402
from models.mobilenet import MobileNetClassifier  # noqa: E402
from models.tinyvit import TinyViTClassifier, TinyViTStudent  # noqa: E402
from models.unet import UNetClassifier  # noqa: E402
from models.vmamba import VMambaClassifier  # noqa: E402
from models.vmamba_backbone import VisualMamba  # noqa: E402

DEFAULT_CSV = os.path.join(REPO_ROOT, "results", "results_best_ckpt.csv")

# Display names, kept in sync with utils/plot_friedman_neminy.py.
MODE_TO_MODEL = {
    "mobilenet": "MobileNet",
    "dmobilenet": "MobileNet-D",
    "efficientnet": "EfficientNet",
    "defficientnet": "EfficientNet-D",
    "tinyvit": "TinyViT",
    "dtinyvit": "TinyViT-D",
    "unet": "UNet",
    "dunet": "UNet-D",
    "vmamba": "VMamba",
    "dvmamba": "VMamba-D",
    "retfound_finetune": "RETFound",
}
MODE_ORDER = [
    "mobilenet",
    "dmobilenet",
    "efficientnet",
    "defficientnet",
    "tinyvit",
    "dtinyvit",
    "unet",
    "dunet",
    "vmamba",
    "dvmamba",
    "retfound_finetune",
]

LIGHT_CLASSIFIERS = {
    "mobilenet": MobileNetClassifier,
    "efficientnet": EfficientNetClassifier,
    "unet": UNetClassifier,
    "vmamba": VMambaClassifier,
    "tinyvit": TinyViTClassifier,
}

# CSV mode -> `--student` value used by train/head.py.
DISTILLED_STUDENT = {
    "dvmamba": "vmamba",
    "dtinyvit": "tinyvit",
    "dmobilenet": "mobilenet",
    "defficientnet": "efficientnet",
    "dunet": "unet",
}

# Column order of the output CSV.
OUT_FIELDS = [
    "mode",
    "model",
    "distilled",
    "family",
    "dataset",
    "run_id",
    "checkpoint",
    "device",
    "params_M",
    "backbone_params_M",
    "nonzero_params_M",
    "compressed",
    "batch_size",
    "latency_ms_mean",
    "latency_ms_median",
    "latency_ms_std",
    "latency_ms_min",
    "latency_ms_max",
    "per_image_ms",
    "throughput_batch",
    "throughput_img_s",
    "flops_G",
    "energy_kwh",
    "co2_kg",
]


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def _load_state_dict(path: str) -> OrderedDict:
    """Load a checkpoint and unwrap the Lightning `state_dict`."""
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("state_dict", ckpt)


def build_student_backbone(student: str):
    """Build the bare student backbone exactly as `train/head.py` does."""
    if student == "tinyvit":
        return TinyViTStudent(model_name=TINYVIT_MODEL, pretrained=False)
    if student in ("mobilenet", "efficientnet", "unet"):
        from train.head import CNN_STUDENTS

        return CNN_STUDENTS[student](img_size=IMG_SIZE, mask_ratio=0.0)
    if student == "vmamba":
        return VisualMamba(
            img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            in_chans=IN_CHANS,
            embed_dim=VMAMBA_EMBED_DIM,
            depth=VMAMBA_DEPTH,
            learning_rate=0.0,
            mask_ratio=0.0,
            use_cls_token=False,
        )
    raise ValueError(f"Unknown student backbone: {student!r}")


# RETFoundTask.__init__ always loads the 3.95 GB CFP weights, so build it once.
_RETFOUND_CACHE: dict = {}


def build_retfound(ckpt_path: str):
    from train.train_retfound import RETFoundTask

    if "model" not in _RETFOUND_CACHE:
        model = RETFoundTask.load_from_checkpoint(
            ckpt_path,
            strict=False,
            mode="finetune",
            class_weights=None,
            map_location="cpu",
        )
        _RETFOUND_CACHE["model"] = model
    else:
        model = _RETFOUND_CACHE["model"]
        model.load_state_dict(_load_state_dict(ckpt_path), strict=False)
    return model


def build_model(mode: str, ckpt_path: str, compress: bool = False):
    """Instantiate + load weights for one (mode, checkpoint). Returns an nn.Module.

    `compress` replays the prune + dynamic-quantize step of `train/head.py` (distilled modes only).
    """
    if mode in LIGHT_CLASSIFIERS:
        cls = LIGHT_CLASSIFIERS[mode]
        kwargs = {"num_classes": NUM_CLASSES, "pretrained": False, "lr": LR, "class_weights": None}
        if mode == "unet":
            kwargs["in_channels"] = IN_CHANS
        model = cls(**kwargs)
        model.load_state_dict(_load_state_dict(ckpt_path), strict=False)

    elif mode in DISTILLED_STUDENT:
        from train.head import VMambaHeadTask

        backbone = build_student_backbone(DISTILLED_STUDENT[mode])
        model = VMambaHeadTask(backbone, lr=LR, class_weights=None)
        model.load_state_dict(_load_state_dict(ckpt_path), strict=False)
        if compress:
            from train.prune import apply_dynamic_quantization, apply_structured_pruning

            model.eval()
            model.backbone = apply_structured_pruning(model.backbone, amount=0.2, n=2, dim=0)
            model.head = apply_structured_pruning(model.head, amount=0.2, n=2, dim=0)
            model.backbone = apply_dynamic_quantization(model.backbone)
            model.head = apply_dynamic_quantization(model.head)

    elif mode == "retfound_finetune":
        model = build_retfound(ckpt_path)

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_forward(model, x, device, warmup: int, runs: int) -> list[float]:
    """Return `runs` wall-clock latencies (ms) for one forward pass of `x`."""
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        _synchronize(device)

        times = []
        for _ in range(runs):
            _synchronize(device)
            t0 = time.perf_counter()
            model(x)
            _synchronize(device)
            times.append((time.perf_counter() - t0) * 1000.0)
    return times


def count_params_M(model, nonzero: bool = False) -> float:
    """Parameter count in millions; unpacks dynamically-quantized Linear weights."""
    from torch.ao.nn.quantized.dynamic import Linear as QLinear

    total = 0
    for m in model.modules():
        if isinstance(m, QLinear):
            tensors = [m.weight().dequantize()]
            if m.bias() is not None:
                tensors.append(m.bias())
        else:
            tensors = list(m.parameters(recurse=False))
        for t in tensors:
            total += int(torch.count_nonzero(t)) if nonzero else t.numel()
    return total / 1e6


# --------------------------------------------------------------------------- #
# Checkpoint selection
# --------------------------------------------------------------------------- #
def select_rows(df, granularity: str, dataset_pref: list[str], seed_pref: str) -> list[dict]:
    """Pick the (mode, checkpoint) pairs to benchmark, honouring `granularity`."""
    picks: list[dict] = []

    for mode in MODE_ORDER:
        group = df[df["mode"] == mode]
        if group.empty:
            continue
        group = group[group["model_path"].map(os.path.isfile)]
        if group.empty:
            print(f"[!] {mode}: no checkpoint found on disk, skipping")
            continue

        if granularity == "checkpoint":
            chosen = [r for _, r in group.iterrows()]
        elif granularity == "dataset":
            chosen = []
            for ds in dataset_pref:
                sub = group[group["dataset"] == ds]
                if not sub.empty:
                    chosen.append(sub.iloc[0])
        else:  # one representative per mode
            row = None
            for ds in dataset_pref:
                sub = group[(group["dataset"] == ds) & (group["seed"] == seed_pref)]
                if not sub.empty:
                    row = sub.iloc[0]
                    break
            if row is None:
                for ds in dataset_pref:
                    sub = group[group["dataset"] == ds]
                    if not sub.empty:
                        row = sub.iloc[0]
                        break
            if row is None:
                row = group.iloc[0]
            chosen = [row]

        for r in chosen:
            picks.append(r.to_dict())

    return picks


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Results table to read checkpoints from.")
    parser.add_argument("--output", default=None, help="Output CSV path (default: <csv dir>/inference_time.csv).")
    parser.add_argument("--granularity", choices=["mode", "dataset", "checkpoint"], default="mode",
                        help="'mode' = one representative checkpoint per model (11 rows); "
                             "'dataset' = one per (mode, dataset); 'checkpoint' = every row.")
    parser.add_argument("--dataset", default="mbrset",
                        help="Preferred dataset when picking a representative checkpoint (comma-separated for a priority list).")
    parser.add_argument("--seed", default="42", help="Preferred seed when picking a representative checkpoint.")
    parser.add_argument("--mode", action="append", default=None,
                        help="Only benchmark these modes (repeatable). Default: all 11.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. 'cpu' or 'cuda'.")
    parser.add_argument("--threads", type=int, default=0,
                        help="Torch CPU threads (0 = leave the default). Pinning to a small number "
                             "usually reduces variance a lot on a shared machine.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the latency measurement.")
    parser.add_argument("--throughput-batch", type=int, default=32,
                        help="Batch size for the throughput measurement (0 disables it).")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs per model.")
    parser.add_argument("--warmup", type=int, default=10, help="Number of untimed warmup iterations.")
    parser.add_argument("--flops", action="store_true", help="Also compute GFLOPs + params with ptflops.")
    parser.add_argument("--no-compress", action="store_true",
                        help="Benchmark distilled models as dense fp32 instead of pruned + int8-quantized (CPU only).")
    parser.add_argument("--co2", action="store_true",
                        help="Track energy (kWh) and CO2 (kg) per model with CodeCarbon over the timed passes.")
    parser.add_argument("--list", action="store_true", help="Only list the selected checkpoints, do not time.")
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)

    if args.mode:
        wanted = set(args.mode)
        unknown = wanted - set(MODE_TO_MODEL)
        if unknown:
            print(f"[!] Unknown mode(s): {sorted(unknown)}", file=sys.stderr)
            return 2

    dataset_pref = [d.strip() for d in args.dataset.split(",") if d.strip()]
    picks = select_rows(df, args.granularity, dataset_pref, str(args.seed))
    if args.mode:
        picks = [p for p in picks if p["mode"] in args.mode]

    if not picks:
        print("[!] Nothing to benchmark.", file=sys.stderr)
        return 1

    if args.list:
        for p in picks:
            print(f"  {p['mode']:18s} {p['dataset']:9s} seed={p['seed']:>3s} {p['model_path']}")
        print(f"\n[i] {len(picks)} checkpoint(s) selected.")
        return 0

    device = torch.device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    out_fields = list(OUT_FIELDS)
    if not args.flops:
        out_fields.remove("flops_G")
    if not args.co2:
        out_fields.remove("energy_kwh")
        out_fields.remove("co2_kg")
    if not args.throughput_batch:
        out_fields.remove("throughput_batch")
        out_fields.remove("throughput_img_s")

    print(f"[i] Device: {device}  |  models: {len(picks)}  |  runs: {args.runs}  |  warmup: {args.warmup}")
    if args.threads:
        print(f"[i] Torch CPU threads pinned to {args.threads}")
    print(f"[i] Latency batch size: {args.batch_size}"
          + (f"  |  throughput batch size: {args.throughput_batch}" if args.throughput_batch else ""))
    if device.type == "cuda":
        print(f"[i] GPU: {torch.cuda.get_device_name(0)}")
    print()

    header = (f"{'model':16s} {'mode':18s} {'params(M)':>9s} {'mean(ms)':>9s} "
              f"{'median':>9s} {'std':>8s} {'min':>8s} {'max':>8s} {'img/s':>9s}")
    print(header)
    print("-" * len(header))

    results: list[dict] = []

    for p in picks:
        mode = p["mode"]
        compressed = mode in DISTILLED_STUDENT and not args.no_compress
        # Dynamic int8 quantization only runs on CPU.
        dev = torch.device("cpu") if compressed else device
        try:
            model = build_model(mode, p["model_path"], compress=compressed)
            model = model.to(dev)
        except Exception as e:  # noqa: BLE001
            print(f"[!] {mode} ({p['dataset']}, {p.get('id')}): failed to build -> {type(e).__name__}: {e}")
            continue

        x = torch.randn(args.batch_size, IN_CHANS, IMG_SIZE, IMG_SIZE, device=dev)
        tracker = None
        if args.co2:
            from codecarbon import EmissionsTracker

            tracker = EmissionsTracker(save_to_file=False, log_level="error", measure_power_secs=1)
            tracker.start()
        try:
            times = time_forward(model, x, dev, args.warmup, args.runs)
        except Exception as e:  # noqa: BLE001
            if tracker is not None:
                tracker.stop()
            print(f"[!] {mode} ({p['dataset']}): forward failed -> {type(e).__name__}: {e}")
            continue

        lat_mean = statistics.fmean(times)
        lat_median = statistics.median(times)
        lat_std = statistics.stdev(times) if len(times) > 1 else 0.0
        lat_min, lat_max = min(times), max(times)
        per_image = lat_mean / args.batch_size

        thr_img_s = ""
        if args.throughput_batch:
            xb = torch.randn(args.throughput_batch, IN_CHANS, IMG_SIZE, IMG_SIZE, device=dev)
            t_thr = statistics.fmean(time_forward(model, xb, dev, max(2, args.warmup // 2), args.runs))
            thr_img_s = args.throughput_batch / (t_thr / 1000.0)

        co2_kg = energy_kwh = ""
        if tracker is not None:
            co2_kg = tracker.stop() or 0.0
            energy_kwh = tracker.final_emissions_data.energy_consumed

        n_params = count_params_M(model)
        n_nonzero = count_params_M(model, nonzero=True)
        backbone = getattr(model, "backbone", None)
        n_backbone = count_params_M(backbone) if backbone is not None else n_params

        flops_g = ""
        if args.flops:
            from utils.flops import compute_flops

            flops, _ = compute_flops(model, IMG_SIZE)
            flops_g = flops / 1e9

        row = {
            "mode": mode,
            "model": MODE_TO_MODEL[mode],
            "distilled": mode in DISTILLED_STUDENT,
            "family": MODE_TO_MODEL[mode].replace("-D", ""),
            "dataset": p["dataset"],
            "run_id": p.get("id") or "",
            "checkpoint": p["model_path"],
            "device": f"{dev.type}:{args.threads}" if dev.type == "cpu" else str(dev),
            "params_M": round(n_params, 4),
            "backbone_params_M": round(n_backbone, 4),
            "nonzero_params_M": round(n_nonzero, 4),
            "compressed": compressed,
            "batch_size": args.batch_size,
            "latency_ms_mean": round(lat_mean, 4),
            "latency_ms_median": round(lat_median, 4),
            "latency_ms_std": round(lat_std, 4),
            "latency_ms_min": round(lat_min, 4),
            "latency_ms_max": round(lat_max, 4),
            "per_image_ms": round(per_image, 4),
        }
        if args.throughput_batch:
            row["throughput_batch"] = args.throughput_batch
            row["throughput_img_s"] = round(thr_img_s, 4)
        if args.flops:
            row["flops_G"] = round(flops_g, 4)
        if args.co2:
            row["energy_kwh"] = energy_kwh
            row["co2_kg"] = co2_kg
        results.append(row)

        thr_s = f"{thr_img_s:9.2f}" if isinstance(thr_img_s, float) else f"{'-':>9s}"
        print(f"{MODE_TO_MODEL[mode]:16s} {mode:18s} {n_params:9.4f} {lat_mean:9.3f} "
              f"{lat_median:9.3f} {lat_std:8.3f} {lat_min:8.3f} {lat_max:8.3f} {thr_s}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not results:
        print("[!] No results produced.", file=sys.stderr)
        return 1

    out_path = args.output or os.path.join(os.path.dirname(os.path.abspath(args.csv)), "inference_time.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in out_fields})

    print()
    print(f"[i] {len(results)} model(s) benchmarked.")
    print(f"[i] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
