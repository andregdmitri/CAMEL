<p align="center">
  <img src="images/camel.png" alt="CAMEL" width="320">
</p>

# CAMEL: Cross-Architecture Masked Embedding Learning
## Previously called LEMUR: A Ultra-Lightweight Distilled Retinal Foundation Model Using Mamba
CAMEL distills the RETFound retinal foundation model (teacher) into ultra-lightweight students of different architectures (VMamba/Vim, TinyViT, MobileNet, EfficientNet, UNet), then trains a classification head on the distilled backbone for retinal fundus grading. The repository contains the full pipeline: distillation, head training (with optional pruning and quantization), baselines, RETFound fine-tuning, evaluation and efficiency benchmarking.

Supported datasets: IDRiD, APTOS, mBRSET, Messidor, PAPILA (set the data paths in `config/constants.py` or via environment variables / `.env`).

---

## Repository layout

- `main.py` — unified CLI entrypoint
- `config/constants.py` — defaults, dataset/checkpoint paths (overridable through environment variables)
- `dataloader/` — Lightning DataModules (`idrid.py`, `aptos.py`, `mbrset.py`, `messidor.py`, `mured.py`, `papila.py`)
- `models/` — teacher (`retfound.py`, `models_vit.py`), students (`vmamba*.py`, `tinyvit.py`, `mobilenet.py`, `efficientnet.py`, `unet.py`, `cnn_student.py`) and distillation module (`dist.py`)
- `train/` — `distill.py` (Phase I), `head.py` (Phase II), `train.py` (baselines), `train_retfound.py`, `prune.py`
- `eval/` — `eval_vmamba.py`, `eval_retfound.py`, `shared_eval.py`
- `optimizers/` — warmup + cosine schedule
- `utils/` — `benchmark_inference.py`, `flops.py`, `verify_flops_params.py`, `transforms.py`, `results.py`, ...
- `sh/` — the exact commands used for all experiments
- `results/` — result tables (`results_best_ckpt.csv`, `inference_time.csv`)
- `images/` — figures (`images/plot/` holds the plotting scripts)

---

## CLI (`main.py`)

```bash
python main.py --run <mode> [options]
```

### Modes (`--run`)

| Mode | Description |
|---|---|
| `distill` | Phase I: distill the RETFound teacher into the student selected with `--student` |
| `head` | Phase II: train the classification head on a distilled backbone (requires `--load_backbone`) |
| `eval` | Evaluate a student checkpoint saved by head training (requires `--load_model`) |
| `retfound_linear` | RETFound linear probe |
| `retfound_finetune` | RETFound fine-tuning |
| `retfound_eval` | Evaluate a RETFound Lightning checkpoint (requires `--load_model`) |
| `mobilenet`, `efficientnet`, `unet`, `vmamba`, `vmamba3`, `tinyvit` | Train the corresponding lightweight model directly (no distillation) |

### Options

| Option | Default | Description |
|---|---|---|
| `--lr` | `5e-4` (`LR`) | Learning rate for distillation, head training and RETFound |
| `--mask_ratio` | `0.0` (`MASK_RATIO`) | Mask ratio used for the student masking |
| `--dist_epochs` | `1000` | Distillation epochs |
| `--head_epochs` | `1000` | Head-training epochs |
| `--teacher_ckpt` | `None` | Override the teacher checkpoint path |
| `--student` | `vmamba` | Student architecture: `vmamba`, `vmamba3`, `tinyvit`, `mobilenet`, `efficientnet`, `unet` |
| `--load_backbone` | `None` | Distilled backbone checkpoint (required for `--run head`) |
| `--load_model` | `None` | Full model checkpoint (required for `--run eval` and `--run retfound_eval`) |
| `--dataset` | `idrid` | `idrid`, `aptos`, `mbrset`, `messidor`, `papila` |
| `--seed` | `42` | Random seed |
| `--augmentation` | `default` | `default`, `none`, `retina_all`, `retina_strong` |
| `--num_workers` | `32` (`NUM_WORKERS`) | DataLoader workers |
| `--prune` | off | Apply structured pruning to the backbone and head after head training (`--run head`) |
| `--prune_amount` | `0.2` | Fraction of channels pruned |
| `--prune_n` | `2` | Norm order (`n`) of the structured pruning |
| `--prune_dim` | `0` | Dimension pruned |
| `--quantize` | off | Apply int8 dynamic quantization to the backbone and head after head training (`--run head`, CPU inference only) |
| `--checkpoint` | `None` | Pretrained RETFound checkpoint |
| `--epochs` | `1000` | Epochs for RETFound linear/fine-tune modes |
| `--model` | `None` | Lightweight model name (set automatically when `--run` is a lightweight model) |
| `--pretrained` | off | Use ImageNet-pretrained backbone weights when available |

Other hyperparameters (batch size, patch size, VMamba depth/embedding dimension, patience, weight decay, ...) are read from `config/constants.py` and can be overridden with environment variables.

### Examples

```bash
# Phase I: distillation (RETFound -> student)
python main.py --run distill --student tinyvit --dataset idrid --seed 42 --mask_ratio 0.75 --augmentation retina_all

# Phase II: head training on the distilled backbone, with pruning + quantization
python main.py --run head --student tinyvit --dataset idrid --seed 42 --mask_ratio 0.0 \
    --augmentation retina_all --load_backbone distillation/<run_id>/checkpoints/best_distillation_42_idrid.ckpt \
    --prune --quantize

# Evaluate a student checkpoint
python main.py --run eval --dataset aptos --load_model CHECKPOINT_PATH

# Lightweight baseline trained without distillation
python main.py --run mobilenet --dataset mbrset --seed 42 --augmentation retina_all

# RETFound teacher
python main.py --run retfound_linear   --dataset idrid --lr 3e-4
python main.py --run retfound_finetune --dataset aptos --lr 1e-5
python main.py --run retfound_eval     --dataset idrid --load_model checkpoints/retfound_finetuned.ckpt
```

The full set of commands used for the experiments is in `sh/`.

### Checkpoints

- Distillation and head training are saved as Lightning checkpoints (`best_distillation_*.ckpt`, `best_head.ckpt`).
- Head training additionally writes `vmamba_final_head[_pruned][_quantized].pth` with two keys, `backbone` and `head` (each a `state_dict`), which `--run eval` expects.
- RETFound evaluation expects a Lightning checkpoint compatible with `train/train_retfound.RETFoundTask`.

---

## Results

Validation metrics on the best checkpoint (selected by `val/f1`), reported as mean ± std over 5 seeds (42-46). Models with the `-D` suffix are distilled from RETFound with CAMEL; the others are trained directly. `Vim` denotes the VMamba student. Source: `results/results_best_ckpt.csv`.

#### APTOS

| Model | Runs | Accuracy | F1 (macro) | AUROC | AUPR |
|---|---|---|---|---|---|
| MobileNet | 5 | 0.803 ± 0.004 | 0.647 ± 0.017 | 0.930 ± 0.006 | 0.673 ± 0.027 |
| MobileNet-D | 5 | 0.817 ± 0.006 | 0.681 ± 0.018 | 0.937 ± 0.004 | 0.701 ± 0.006 |
| EfficientNet | 5 | 0.786 ± 0.019 | 0.619 ± 0.024 | 0.928 ± 0.006 | 0.659 ± 0.020 |
| EfficientNet-D | 5 | 0.832 ± 0.009 | 0.698 ± 0.016 | 0.936 ± 0.006 | 0.697 ± 0.014 |
| UNet | 5 | 0.805 ± 0.019 | 0.644 ± 0.037 | 0.937 ± 0.006 | 0.708 ± 0.022 |
| UNet-D | 5 | 0.852 ± 0.019 | 0.735 ± 0.027 | 0.943 ± 0.007 | 0.744 ± 0.023 |
| Vim | 5 | 0.588 ± 0.151 | 0.296 ± 0.102 | 0.787 ± 0.080 | 0.430 ± 0.060 |
| Vim-D | 5 | 0.744 ± 0.004 | 0.445 ± 0.007 | 0.867 ± 0.009 | 0.527 ± 0.014 |
| TinyViT | 5 | 0.789 ± 0.023 | 0.619 ± 0.048 | 0.928 ± 0.010 | 0.663 ± 0.041 |
| TinyViT-D | 5 | 0.849 ± 0.008 | 0.736 ± 0.013 | 0.939 ± 0.013 | 0.738 ± 0.030 |
| RETFound | 5 | 0.834 ± 0.005 | 0.703 ± 0.010 | 0.932 ± 0.003 | 0.703 ± 0.010 |

#### IDRiD

| Model | Runs | Accuracy | F1 (macro) | AUROC | AUPR |
|---|---|---|---|---|---|
| MobileNet | 5 | 0.208 ± 0.059 | 0.168 ± 0.055 | 0.580 ± 0.015 | 0.317 ± 0.031 |
| MobileNet-D | 5 | 0.369 ± 0.025 | 0.295 ± 0.022 | 0.570 ± 0.033 | 0.323 ± 0.019 |
| EfficientNet | 5 | 0.235 ± 0.033 | 0.189 ± 0.040 | 0.587 ± 0.025 | 0.304 ± 0.009 |
| EfficientNet-D | 5 | 0.454 ± 0.036 | 0.384 ± 0.080 | 0.725 ± 0.080 | 0.406 ± 0.062 |
| UNet | 5 | 0.361 ± 0.123 | 0.334 ± 0.115 | 0.745 ± 0.069 | 0.406 ± 0.081 |
| UNet-D | 5 | 0.452 ± 0.062 | 0.425 ± 0.079 | 0.755 ± 0.094 | 0.477 ± 0.086 |
| Vim | 5 | 0.254 ± 0.056 | 0.203 ± 0.027 | 0.569 ± 0.026 | 0.296 ± 0.021 |
| Vim-D | 5 | 0.301 ± 0.028 | 0.255 ± 0.010 | 0.588 ± 0.012 | 0.314 ± 0.006 |
| TinyViT | 5 | 0.373 ± 0.064 | 0.348 ± 0.058 | 0.775 ± 0.018 | 0.435 ± 0.026 |
| TinyViT-D | 5 | 0.538 ± 0.049 | 0.471 ± 0.019 | 0.779 ± 0.020 | 0.477 ± 0.037 |
| RETFound | 5 | 0.557 ± 0.052 | 0.440 ± 0.034 | 0.740 ± 0.022 | 0.448 ± 0.028 |

#### mBRSET

| Model | Runs | Accuracy | F1 (macro) | AUROC | AUPR |
|---|---|---|---|---|---|
| MobileNet | 5 | 0.799 ± 0.024 | 0.361 ± 0.168 | 0.641 ± 0.135 | 0.369 ± 0.149 |
| MobileNet-D | 5 | 0.812 ± 0.007 | 0.483 ± 0.013 | 0.790 ± 0.032 | 0.499 ± 0.039 |
| EfficientNet | 5 | 0.810 ± 0.015 | 0.471 ± 0.166 | 0.754 ± 0.116 | 0.483 ± 0.153 |
| EfficientNet-D | 5 | 0.836 ± 0.009 | 0.558 ± 0.023 | 0.857 ± 0.008 | 0.600 ± 0.023 |
| UNet | 5 | 0.895 ± 0.006 | 0.693 ± 0.027 | 0.939 ± 0.014 | 0.758 ± 0.014 |
| UNet-D | 5 | 0.885 ± 0.010 | 0.663 ± 0.011 | 0.932 ± 0.010 | 0.728 ± 0.025 |
| Vim | 5 | 0.759 ± 0.052 | 0.177 ± 0.003 | 0.549 ± 0.017 | 0.226 ± 0.010 |
| Vim-D | 5 | 0.741 ± 0.058 | 0.180 ± 0.007 | 0.528 ± 0.033 | 0.215 ± 0.014 |
| TinyViT | 5 | 0.809 ± 0.016 | 0.476 ± 0.034 | 0.805 ± 0.020 | 0.536 ± 0.023 |
| TinyViT-D | 5 | 0.821 ± 0.015 | 0.570 ± 0.022 | 0.824 ± 0.013 | 0.569 ± 0.015 |
| RETFound | 5 | 0.682 ± 0.096 | 0.272 ± 0.008 | 0.628 ± 0.063 | 0.275 ± 0.031 |

#### Messidor

| Model | Runs | Accuracy | F1 (macro) | AUROC | AUPR |
|---|---|---|---|---|---|
| MobileNet | 5 | 0.490 ± 0.036 | 0.331 ± 0.031 | 0.728 ± 0.015 | 0.347 ± 0.026 |
| MobileNet-D | 5 | 0.545 ± 0.025 | 0.419 ± 0.047 | 0.759 ± 0.033 | 0.408 ± 0.045 |
| EfficientNet | 5 | 0.536 ± 0.023 | 0.405 ± 0.113 | 0.738 ± 0.054 | 0.412 ± 0.099 |
| EfficientNet-D | 5 | 0.559 ± 0.035 | 0.461 ± 0.111 | 0.785 ± 0.065 | 0.485 ± 0.120 |
| UNet | 5 | 0.466 ± 0.042 | 0.315 ± 0.020 | 0.726 ± 0.023 | 0.331 ± 0.021 |
| UNet-D | 5 | 0.486 ± 0.027 | 0.321 ± 0.047 | 0.732 ± 0.017 | 0.353 ± 0.030 |
| Vim | 5 | 0.441 ± 0.047 | 0.322 ± 0.030 | 0.719 ± 0.010 | 0.321 ± 0.020 |
| Vim-D | 5 | 0.849 ± 0.003 | 0.184 ± 0.000 | 0.647 ± 0.025 | 0.272 ± 0.048 |
| TinyViT | 5 | 0.526 ± 0.022 | 0.409 ± 0.034 | 0.770 ± 0.027 | 0.433 ± 0.063 |
| TinyViT-D | 5 | 0.571 ± 0.020 | 0.552 ± 0.028 | 0.805 ± 0.018 | 0.532 ± 0.017 |
| RETFound | 5 | 0.523 ± 0.042 | 0.299 ± 0.016 | 0.646 ± 0.080 | 0.358 ± 0.076 |

### Computational complexity and efficiency

Computational complexity, inference efficiency and environmental footprint of the base architectures on an NVIDIA RTX A5000 GPU. Latency is measured per image at batch size 1; throughput is images per second at batch size 32. Energy and CO<sub>2</sub> are scaled by 10<sup>-6</sup> for readability. Results are aggregated across all runs (mean ± std).

| Model | Params (M) | GFLOPs | Latency (ms) | Throughput (img/s) | Energy (10<sup>-6</sup> kWh) | CO<sub>2</sub> (10<sup>-6</sup> kg) |
|---|---|---|---|---|---|---|
| MobileNet | 1.52 | 0.111 | 1.18 ± 0.02 | 11611.9 ± 82.5 | 0.95 ± 1.95 | 0.09 ± 0.19 |
| EfficientNet | 4.01 | 0.769 | 1.83 ± 0.04 | 2649.6 ± 3.2 | 8.50 ± 2.10 | 0.84 ± 0.21 |
| UNet | 4.85 | 6.561 | 0.71 ± 0.00 | 2348.2 ± 30.9 | 8.60 ± 1.85 | 0.84 ± 0.18 |
| VMamba | 5.02 | 0.476 | 17.46 ± 0.75 | 1691.2 ± 77.8 | 18.15 ± 1.70 | 1.78 ± 0.17 |
| TinyViT | 5.07 | 2.510 | 1.92 ± 0.03 | 1695.6 ± 7.4 | 14.40 ± 2.20 | 1.41 ± 0.21 |
| **Foundation Teacher** | | | | | | |
| RETFound | 303.31 | 119.292 | 14.79 ± 0.12 | 93.2 ± 0.9 | 278.50 ± 2.90 | 27.38 ± 0.28 |

To reproduce the timing, energy and CO<sub>2</sub> measurements:

```bash
python utils/benchmark_inference.py --granularity checkpoint --device cuda --co2 [--flops] [--no-compress]
```

Useful flags: `--granularity {mode,dataset,checkpoint}`, `--mode <name>` (repeatable), `--batch-size`, `--throughput-batch`, `--runs`, `--warmup`, `--threads`, `--output`. Distilled (`-D`) models are benchmarked pruned + int8-quantized on CPU unless `--no-compress` is given; results are written to `results/inference_time.csv`.


<p align="center">
  <img src="images/camel_big.png" alt="camel_big" width="320">
</p>
