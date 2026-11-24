# Brain tumor segmentation on BraTS 2020: CNN vs transformer vs pretraining

Five 2D segmentation methods trained and evaluated under **identical conditions** (same patient split,
same loss, same schedule, same metrics), to answer:

1. How much do **transformers** add over a strong CNN (U-Net)?
2. Does **self-supervised** pretraining (DINOv2) beat **supervised** ImageNet pretraining under the same decoder?
3. Does **in-domain** self-supervised adaptation (I-JEPA on BraTS MRI) beat out-of-domain pretraining alone?

Status: all methods implemented and smoke-tested on real data; full training runs not done yet.

## Methods

| # | `--model` | Architecture | Backbone pretraining | Params (trainable / total) |
|---|---|---|---|---|
| 1 | `unet` | U-Net, 32→512 channels, InstanceNorm, skip connections | none | 7.8M / 7.8M |
| 2 | `jepa` | ViT-S/14 + DPT decoder | DINOv2 → **I-JEPA continued pretraining on BraTS** (`pretrain_jepa.py`) | 4.3M / 26.0M (frozen) |
| 3 | `swinunetr` | SwinUNETR 2D (Swin encoder + CNN decoder, MONAI), feature size 48 | none | 25.1M / 25.1M |
| 4 | `dinov2` | ViT-S/14 + DPT decoder | DINOv2 (self-supervised, natural images) | 4.3M / 26.0M (frozen) |
| 5 | `vit` | ViT-S/16 + DPT decoder | ImageNet-21k → 1k (supervised) | 4.3M / 26.1M (frozen) |

Methods 2, 4 and 5 share the **same DPT decoder** and the same 4-channel input adaptation, so the only
difference between them is how the backbone was pretrained. Each runs **frozen** (decoder only, default) or
**fine-tuned** (`--finetune`, with layer-wise learning-rate decay).

Note: method 3 uses MONAI's SwinUNETR (Swin Transformer encoder with a CNN decoder), not the
pure-transformer Swin-Unet.

## Setup

```bash
# from the repo root, Python 3.11
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### Data
1. Request **BraTS 2020** training data from the
   [challenge site](https://www.med.upenn.edu/cbica/brats2020/data.html) (registration and manual approval).
2. Point the code at the folder holding the `BraTS20_Training_XXX` directories:
   ```bash
   export BRATS_ROOT=/path/to/MICCAI_BraTS2020_TrainingData        # PowerShell: $env:BRATS_ROOT="..."
   ```
3. Build the index (once, ~10 min; reads every volume to compute normalisation statistics):
   ```bash
   python data.py
   ```
   This writes `cache/index.json`. **It is already committed**, so everyone uses the same split; only
   rebuild it if you change the split logic. All 369 patients are used (patient 355's misnamed seg file is
   handled).

Pretrained backbones download from Hugging Face on first use into `$HF_HOME`
(default in the scripts: `D:\huggingface_cache`; set `HF_HOME` to change it).

## Training

All commands run from this folder.

```bash
# Method 1: U-Net
python train.py --model unet --epochs 20

# Method 3: SwinUNETR
python train.py --model swinunetr --epochs 20

# Methods 4 and 5: frozen backbone, DPT decoder only
python train.py --model dinov2 --epochs 20
python train.py --model vit --epochs 20

# ...and fine-tuned (layer-wise LR decay on the backbone)
python train.py --model dinov2 --finetune --epochs 20
python train.py --model vit --finetune --epochs 20

# Method 2: stage 1, I-JEPA continued pretraining (training patients only, no labels)
python pretrain_jepa.py --epochs 100
#           stage 2, segmentation with the adapted backbone
python train.py --model jepa --backbone-ckpt runs/jepa_pretrain/backbone.pt --epochs 20
python train.py --model jepa --backbone-ckpt runs/jepa_pretrain/backbone.pt --finetune --epochs 20
```

Add `--wandb` to log to Weights & Biases (project `brats-segmentation-comparison`).
Resume an interrupted run with `--resume runs/<run_name>/last.pt` (same other arguments).

### Evaluation on the test split

```bash
python train.py --model unet --eval-only --split test --resume runs/unet_s42/best.pt
```

Writes `runs/<run_name>/metrics_test.json`: per-region Dice and HD95 (mean and per patient), parameter
counts and inference time. Metrics are computed on **full 3D volumes** at the original 240×240
resolution, with predictions restricted to the brain mask.

### Outputs

Each run writes to `runs/<run_name>/` (default name `<model>[_ft]_s<seed>`):

| File | Content |
|---|---|
| `config.json` | all arguments |
| `log.jsonl` | one line per epoch: train loss, LR, epoch time, peak VRAM, val Dice per region |
| `best.pt` | weights with the best validation mean Dice |
| `last.pt` | full state for resuming (model, optimizer, scheduler, scaler, epoch) |
| `metrics_<split>.json` | test metrics from `--eval-only` |

`pretrain_jepa.py` writes `runs/jepa_pretrain/backbone.pt` (the EMA target encoder, used downstream) and
`last.pt`.

### `train.py` options

| Option | Default | Meaning |
|---|---|---|
| `--model` | required | `unet`, `swinunetr`, `dinov2`, `vit`, `jepa` |
| `--finetune` | off | ViT methods: train the backbone too |
| `--backbone-ckpt` | — | `jepa`: backbone from `pretrain_jepa.py` |
| `--epochs` | 50 | training epochs |
| `--batch-size` / `--accum` | 8 / 4 | effective batch = 32 |
| `--lr` | 3e-4 U-Net, 1e-4 SwinUNETR, 1e-3 DPT decoder | peak learning rate |
| `--backbone-lr` / `--layer-decay` | 5e-5 / 0.75 | fine-tuning: top-layer LR and per-layer decay |
| `--weight-decay` | 0.01 | AdamW, not applied to biases, norms, embeddings |
| `--neg-fraction` | 0.3 | share of tumor-free training slices kept |
| `--val-every` | 1 | validate every N epochs |
| `--amp` | off | fp16 autocast (see hardware notes) |
| `--seed` | 42 | seeds Python, NumPy, PyTorch |
| `--max-steps` / `--max-val-patients` | — | caps for quick smoke tests |

### `pretrain_jepa.py` options

| Option | Default | Meaning |
|---|---|---|
| `--epochs` | 100 | pretraining epochs |
| `--batch-size` / `--accum` | 32 / 2 | effective batch = 64 |
| `--lr` / `--predictor-lr` | 5e-5 / 5e-4 | encoder (low, to keep DINOv2 features) / new predictor |
| `--ema` | 0.996 1.0 | EMA momentum, linear schedule |
| `--weight-decay` | 0.04 | |

Watch `jepa/std_pred` in the log: if it falls towards 0 while `jepa/std_target` stays high, the
representation is collapsing.

## Design choices

**Data.**
- Patient-level split 70/15/15 (258/55/56), seed 42, so no patient's slices leak across splits.
- 2D axial slices with all 4 sequences stacked as channels (FLAIR, T1, T1ce, T2), each z-scored over brain
  voxels per patient, center-cropped 240 → 224.
- Training keeps every tumor slice and 30% of tumor-free slices; validation/test use whole volumes.
- Augmentation: random horizontal/vertical flips.
- Slices are read straight from the memory-mapped `.nii` files, so there is no slice cache on disk.

**Targets and loss.**
- Region-based outputs: 3 sigmoid channels for the nested evaluation regions WT (labels 1+2+4),
  TC (1+4) and ET (4), instead of a 4-class softmax. Training targets then match exactly what is evaluated.
- Loss = BCE + soft Dice. Dice is computed over the whole batch, and only for regions present in the batch
  (absent regions are left to BCE), so tumor-free slices don't cause unstable Dice gradients.

**Optimisation.**
- AdamW, linear warmup (5%) then cosine decay, gradient clipping at 1.0, gradient accumulation.
- U-Net uses InstanceNorm because the per-step batch is small.
- Fine-tuned ViTs use gradient checkpointing and layer-wise LR decay.

**Method 2 (I-JEPA).**
- Starts from the same DINOv2 ViT-S/14 as method 4, so method 2 vs 4 isolates the effect of in-domain adaptation.
- Multi-block masking: 4 target blocks of 15–20% of the slice, sampled inside the brain; one context
  block of 85–100% with the targets removed.
- Smooth-L1 loss in embedding space against an EMA target encoder. Only training patients are used, so
  val/test images are never seen.

**Evaluation.** Volume-level Dice and HD95 per region, predictions pasted back to 240×240 and masked to the
brain. Best checkpoint selected on validation mean Dice.

## Hardware notes
Developed on a **GTX 1660 Ti (6 GB)**:
- **fp32 by default.** On this card cuDNN's fp16 convolutions return NaN, and fp32 is just as fast (no
  tensor cores). Use `--amp` on GPUs with tensor cores.
- Peak memory in smoke tests: U-Net 1.5 GB, SwinUNETR 2.6 GB, DPT frozen 1.0 GB, DPT fine-tuned 1.3 GB,
  I-JEPA pretraining 2.3 GB.
- Speed: U-Net ≈ 0.9 s per step of 8 slices, i.e. ≈ 40 min per epoch (~21.5k training slices).
  Frozen-backbone DPT runs are about twice as fast.
- Hugging Face downloads failed on this machine with certifi's certificate bundle; the scripts use the
  OS certificate store through `truststore` when it is installed.

## Files

| File | Purpose |
|---|---|
| `data.py` | split, index, normalisation, slice dataset, volume iterator |
| `metrics.py` | label remapping, WT/TC/ET regions, Dice, HD95 (with self-check) |
| `utils.py` | region BCE+Dice loss, warmup-cosine schedule, 2D/3D visualisation |
| `models/unet_models.py` | method 1 |
| `models/vit_dpt.py` | backbones for methods 2, 4, 5 and the shared DPT decoder |
| `train.py` | shared training / evaluation harness for all methods |
| `pretrain_jepa.py` | method 2, stage 1 |
| `PLAN.md` | original project plan (written before implementation; README reflects current decisions) |
| `3d-nii-visualizer/` | submodule: third-party 3D NIfTI viewer |

Quick self-checks: `python utils.py`, `python metrics.py`, `python -m models.unet_models`,
`python -m models.vit_dpt`.

## References
- U-Net: Ronneberger et al., MICCAI 2015, [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- ViT: Dosovitskiy et al., ICLR 2021, [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- DPT: Ranftl et al., ICCV 2021, [arXiv:2103.13413](https://arxiv.org/abs/2103.13413)
- DINOv2: Oquab et al., [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
- I-JEPA: Assran et al., CVPR 2023, [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
- Swin UNETR: Hatamizadeh et al., [arXiv:2201.01266](https://arxiv.org/abs/2201.01266)
- nnU-Net: Isensee et al., Nature Methods 18:203–211 (2021), [doi:10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z)
- BraTS 2020: [challenge data page](https://www.med.upenn.edu/cbica/brats2020/data.html)
