# IA4Med: medical imaging with computer vision and vision-language models

Research code that compares **trained computer-vision segmentation models** with **vision-language
models (VLMs)** on medical images, and explores VLM-based medical report generation.

The main benchmark is brain tumor segmentation on **BraTS 2020** (multi-modal MRI). The CV models and
the VLM segmenters share **one patient-level split and one set of metrics**, so all results go in the
same comparison table.

> Research use only. Nothing here is a medical device or validated for clinical use.

## Repository layout

```
computer_vision/
  brain_segmentation/             BraTS 2020 tumor segmentation, 5 methods, shared training harness
  ChestXRayAnatomySegmentation/   [submodule] CXAS, chest X-ray anatomy segmentation (third-party)
vlm/
  segmentation/                   text-prompted segmentation (BiomedParse v2) on the same BraTS split,
                                  plus a survey of other VLM / foundation segmenters
  medical_understanding/          Qwen2.5-VL-3B fine-tuning for chest X-ray report generation (on hold)
requirements.txt                  environment for the CV side and the VLM-segmentation bridge
```

| Part | Status | README |
|---|---|---|
| Brain tumor segmentation (CV) | Implemented, smoke-tested; full training runs not done yet | [computer_vision/brain_segmentation](computer_vision/brain_segmentation/README.md) |
| VLM segmentation | Export/score bridge implemented; BiomedParse itself not yet installed | [vlm/segmentation](vlm/segmentation/README.md) |
| Chest X-ray report generation | Training pipeline written; no completed run | [vlm/medical_understanding](vlm/medical_understanding/README.md) |

## Methods compared on BraTS 2020

| # | Method | Side | Pretraining |
|---|---|---|---|
| 1 | U-Net | CV | none |
| 2 | ViT-S + DPT, I-JEPA continued pretraining on BraTS | CV | DINOv2 → in-domain self-supervised |
| 3 | SwinUNETR (2D) | CV | none |
| 4 | ViT-S + DPT, DINOv2 backbone | CV | self-supervised, natural images |
| 5 | ViT-S + DPT, ImageNet ViT backbone | CV | supervised, ImageNet |
| ref | BiomedParse v2, text prompts | VLM | biomedical image-text; zero-shot here |

Results will be added once full training runs are done.

## Quick start

```bash
# 1. clone with the third-party submodules
git clone --recursive https://github.com/guermal/my_IA4Med.git
cd my_IA4Med

# 2. environment (Python 3.11)
conda create -n ia4med python=3.11 -y
conda activate ia4med
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 3. point to BraTS 2020 and build the split/index once
export BRATS_ROOT=/path/to/MICCAI_BraTS2020_TrainingData      # PowerShell: $env:BRATS_ROOT="..."
cd computer_vision/brain_segmentation
python data.py

# 4. train and evaluate a model
python train.py --model unet --epochs 20
python train.py --model unet --eval-only --split test --resume runs/unet_s42/best.pt
```

See each folder's README for every method, option and design choice.

## Data
BraTS 2020 is not redistributed here. Request it from the
[BraTS 2020 challenge](https://www.med.upenn.edu/cbica/brats2020/data.html) (account registration and manual
approval) and cite the challenge papers listed on its registration page.
The repo only contains the derived patient split (`computer_vision/brain_segmentation/cache/index.json`:
patient IDs, slice lists and normalisation statistics), so every method uses the exact same patients.

## License
The code in this repository is released under the [Apache License 2.0](LICENSE).
Submodules, pretrained weights and datasets keep their own licenses and terms (see below and each
folder's README; e.g. BiomedParse weights and BraTS data have their own usage terms).

## Third-party code
- [ChestXRayAnatomySegmentation](https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation) (CXAS), as a submodule, under its own license.
- [3d-nii-visualizer](https://github.com/adamkwolf/3d-nii-visualizer), a NIfTI 3D viewer, as a submodule, under its own license.
- Pretrained backbones come from [timm](https://github.com/huggingface/pytorch-image-models) (DINOv2, ViT); SwinUNETR from [MONAI](https://monai.io).
