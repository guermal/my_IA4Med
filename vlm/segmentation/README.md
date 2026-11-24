# VLM segmentation: text-prompted BiomedParse v2 on BraTS

A zero-shot reference to compare against the computer-vision models in
`computer_vision/brain_segmentation`: same patient split, same WT/TC/ET metrics.
`biomedparse_brats.py` does the two ends (export to BiomedParse's input, score its output);
BiomedParse itself runs in its own environment.

## Status
- Done: exporter, scorer, and a round-trip self-check (`python biomedparse_brats.py check`).
- Not done: BiomedParse is not installed and its weights are not downloaded (steps 1–3 below need you).
- Chest X-ray: v2 lists no chest X-ray targets. Only v1 has them ("left lung", "lung opacity", …),
  and v1 needs detectron2 + mpi4py, which are awkward on Windows. Left for later.

## Other methods (not implemented yet)
Facts below were checked on 2026-11-24 against the papers' abstracts, repos and model cards, not the
full papers. Re-check benchmark numbers before quoting them.

### A. Text-prompted segmentation (the VLM route; BiomedParse belongs here)
| Method | What it is | Notes for us |
|---|---|---|
| **SAT** (Zhao et al., arXiv 2312.17183) | 3D volume + medical term as text prompt → mask. Text encoder gets medical knowledge (>6,000 concepts) through contrastive learning. Trained on SAT-DS: 22,186 3D scans, 302,033 masks, 72 datasets. SAT-Nano 110M / SAT-Pro 447M params | Natively 3D, which suits BraTS volumes. The authors report SAT-Pro beats BiomedParse on 29 of 30 categories. The strongest second VLM-seg candidate. |

### B. Multimodal LLMs with pixel grounding (segmentation + text)
These bridge the segmentation side and the report-generation project (`../medical_understanding`).
| Method | What it is | Notes for us |
|---|---|---|
| **MedPLIB** (arXiv 2412.09278) | End-to-end biomedical MLLM with pixel-level understanding; cites LISA (reasoning segmentation) as a precursor | Answers questions *and* outputs masks |
| **UniBiomed** (arXiv 2504.21336) | MLLM + SAM: segments targets and writes grounded descriptions; trained on >27M image-annotation-text triplets, 10 modalities | Authors report +10.25% average Dice over BiomedParse. Closest to "segmentation + medical summary" in one model |

### C. Promptable segmenters (visual prompts, not language)
Not VLMs: they need a box, point or scribble. Best used as a reference upper bound or for annotation.
| Method | Prompt | Notes for us |
|---|---|---|
| **MedSAM** (Ma et al., *Nature Comms* 15:654, 2024) | box | SAM fine-tuned on >1M medical image-mask pairs; 2D. A box taken from the ground truth is an oracle, so report it as an upper bound only. |
| **MedSAM2** (Ma et al., arXiv 2504.03600) | box/point, propagated through 3D | Built on SAM 2; >455K 3D pairs + 76K video frames. Not to be confused with "Medical SAM 2" (arXiv 2408.00874), a different paper. |
| **SAM-Med3D** (arXiv 2310.15161) | 3D points | Fully 3D; one point per volume instead of one prompt per slice |
| **nnInteractive** (DKFZ, arXiv 2503.08373) | points, scribbles, boxes, lasso | Won the CVPR 2025 interactive 3D segmentation challenge. napari plugin (`napari-nninteractive`), which our `utils.py` already uses. Checkpoint is CC BY-NC-SA 4.0. **Best tool if new data needs annotating.** |
| **VISTA3D** (NVIDIA / MONAI, arXiv 2406.05285) | class or 3D clicks | **CT only** (127 structures), so not for BraTS MRI; relevant if a CT project starts |

### D. Ready-trained models (belong on the computer-vision side)
| Method | Notes for us |
|---|---|
| **MONAI `brats_mri_segmentation`** | SegResNet trained on BraTS 2018, all 4 sequences, outputs WT/TC/ET directly. Reported val Dice WT 0.90 / TC 0.86 / ET 0.79. Apache-2.0, not gated. The easiest "no training" reference row; BraTS 2018 overlaps BraTS 2020 patients, so it is contaminated too. |
| **BrainSegFounder** (*Medical Image Analysis*, 2024) | ViT foundation model pretrained self-supervised on 41,400 UK Biobank brain MRIs, fine-tuned on BraTS and ATLAS. Code + weights at github.com/lab-smile/BrainSegFounder. Directly relevant to our method 2 (in-domain pretraining). |
| **FleXray** (arXiv 2609.26756) | Chest X-ray anatomy (60 structures), CNN U-Net. X-ray only, no tumors; relevant to the chest X-ray work, like CXAS in `computer_vision/`. |

### Suggested order
1. BiomedParse (done here, needs setup) → 2. SAT (second text-prompted model, natively 3D) →
3. MONAI BraTS bundle as a no-training reference → 4. UniBiomed / MedPLIB when we revisit the report project.

## Steps
1. **Accept the weights' terms.** The weights are gated: open https://huggingface.co/microsoft/BiomedParse
   while logged in and agree to share your information. Then run `huggingface-cli login`.
2. **Separate environment** (pins torch 2.6 / timm 0.9.16 / transformers 4.40, which clash with the main environment from the top-level `requirements.txt`):
   ```
   conda create -n biomedparse python=3.10.14 -y
   conda activate biomedparse
   pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
   git clone -b v2 https://github.com/microsoft/BiomedParse.git
   pip install -r BiomedParse/assets/requirements/inference.txt
   ```
   Windows support isn't stated anywhere in their docs; if something Linux-only shows up, use WSL.
3. **Export, infer, score:**
   ```
   # in the main env, from this folder
   python biomedparse_brats.py export --split test --sequence t1ce
   # in biomedparse env, from the BiomedParse clone (downloads biomedparse_v2.ckpt, ~4.5 GB)
   python inference.py --input_dir <this folder>/inputs/t1ce --output_dir <this folder>/outputs/t1ce --slice_batch_size 1
   # back in the main env
   python biomedparse_brats.py score --split test --pred-dir outputs/t1ce
   ```
   Lower `--slice_batch_size` if the 6 GB GPU runs out of memory (no memory figures are published).

## Design choices and caveats
- **Prompts** use v2's own BraTS vocabulary (BraTS-2024 names): "Non-enhancing tumor core" → label 1,
  "Surrounding non-enhancing FLAIR hyperintensity" → label 2 (edema), "Enhancing tissue" → label 4.
- **One MRI sequence per run.** BiomedParse is single-channel; our CV models see all four. Default is
  T1ce (best for enhancing tissue and core); try `--sequence flair` too (best for edema / whole tumor).
- **Intensity scaling** to [0, 255] uses 0.5–99.5 brain percentiles. The challenge data only documents the
  [0, 255] range, not its exact MRI recipe, so this is a guess that may cost some accuracy.
- **Contamination.** BiomedParse's training data lists BraTS 2023, which contains most BraTS 2020 patients.
  Results are a reference row, not a fair zero-shot comparison.
- Research use only, not for clinical use (BiomedParse terms).
