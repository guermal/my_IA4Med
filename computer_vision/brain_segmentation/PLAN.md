# Brain Tumor MRI Segmentation — Project Plan

## Context

The README states a one-line research goal: compare five segmentation approaches for brain tumor
detection — direct UNet, an "encoder-decoder method", SwinUNet, DINO features + a learnable DPT-style
decoder, and a plain pretrained ViT + the same decoder (to isolate the effect of DINO's self-supervised
pretraining vs. standard supervised ViT pretraining).

That's a reasonable model-comparison arc, but on its own it skips everything that makes such a
comparison meaningful or even runnable: what data to train on, how to split it without leaking patient
information between train/val/test, how to handle class imbalance, what loss/metrics to standardize on,
how to keep all five methods on a level playing field, and how to fit any of it inside a 6GB GPU. This
plan fills in those gaps and turns "try 5 methods" into an executable, leakage-free, hardware-feasible
pipeline with a defined output: a real comparison table plus qualitative figures.

**Verified current state (checked directly against the filesystem, not assumed):**
- `D:\brain_tumor\brats\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\` already contains all
  **369 `BraTS20_Training_XXX` patient folders**, each with `flair`, `t1`, `t1ce`, `t2`, `seg` NIfTI
  volumes. This is **BraTS2020** (240×240×155 volumes, labels `{0,1,2,4}`). The download is effectively
  complete.
- `D:\brain_tumor\archive.zip` is a **separate, unrelated** Roboflow 2D-JPEG dataset with box-shaped
  pseudo-masks (not BraTS). It should **not** be mixed into this pipeline.
- BraTS ships no publicly-labeled held-out validation/test set — train/val/test must be carved out of
  the 369 labeled patients ourselves.
- **The local Python environment's `torch` install is currently broken**: `import torch` fails with
  `ImportError: DLL load failed while importing _C`. The GPU itself is visible to the driver
  (`nvidia-smi` reports a GTX 1660 Ti, 6144 MiB, driver 610.88 — new enough for CUDA 12.x wheels). This
  must be fixed before anything else — it's a blocking prerequisite.
- No `requirements.txt`/`environment.yml` exists anywhere in the wider `VLM_projects` workspace. This
  project is the first to need heavy medical-imaging/ViT dependencies (`monai`, `nibabel`, `timm`,
  `einops`), none of which are installed anywhere yet.
- Sibling project `medical_understanding/` has reusable conventions worth mirroring: setting `HF_HOME`
  to a `D:\huggingface_cache` path before importing `torch`/`transformers`, `wandb` for experiment
  tracking, absolute `D:\...` path constants, and a `torch.utils.data.Dataset` subclass pattern.

Decisions already confirmed: use BraTS (not the box-mask dataset) as the real segmentation target; plan
strictly for the local 6GB GPU (no cloud assumed); cover the full pipeline (environment, data, harness,
all 5 models, evaluation) — not just architecture-level notes.

---

## What the README's plan was missing

1. **No real ground-truth strategy.** BraTS fixes this, but brings its own requirements: multi-modal
   4-channel input, multi-class labels that need remapping to the standard WT/TC/ET evaluation regions,
   and volumes (not natural 2D images) that must be sliced.
2. **No leakage-safe data split.** Any split must happen at the **patient level**, not slice level —
   slices from the same patient are highly correlated; a slice-level split would silently inflate every
   reported Dice score.
3. **No class-imbalance / negative-slice strategy.** Tumors occupy a small fraction of most slices, and
   many axial slices contain no tumor at all.
4. **No defined loss/metric convention.** BraTS has a standard: Dice+CE loss, per-region (WT/TC/ET)
   Dice, Hausdorff95 computed on reconstructed volumes.
5. **No shared training harness.** Without one, each of the 5 methods risks being trained/evaluated
   slightly differently, invalidating the comparison that's the whole point of the project.
6. **No hardware feasibility check.** Forces concrete choices: 2D slice-based segmentation (not 3D
   volumetric), frozen backbones by default for the transformer methods, fp16 mixed precision, narrow
   channel widths.
7. **Ambiguous method 2.** "Encoder-decoder method" as distinct from UNet was never defined.
8. **No fairness guarantee between methods 4 and 5.** The DINO-vs-plain-ViT comparison only means
   something if everything except the backbone's pretraining is held identical.
9. **No environment/dependency plan**, and the environment that exists is currently non-functional.
10. **No defined "done" state.** No comparison table, no qualitative visualization plan, no ablations,
    no multi-seed variance check.

---

## Roadmap

### Phase 0 — Unblock the environment
- Fix the broken `torch` install: uninstall the current mismatched/CPU install, reinstall a CUDA-wheel
  build appropriate for the GTX 1660 Ti / driver 610.88.
- Create `requirements.txt` covering: `torch`/`torchvision` (via the CUDA wheel index, called out
  separately), `monai`, `nibabel`, `timm`, `einops`, `transformers` (fallback for DINOv2 weights),
  `albumentations`, `wandb`, `scikit-image`, `matplotlib`, `pandas`, `tqdm`, `pyyaml`.
- Add a small environment-verification step confirming CUDA is visible to torch, the GPU name matches,
  and one BraTS NIfTI file loads correctly.
- `git init` the project and add a `.gitignore` (data, checkpoints, wandb logs, caches).

### Phase 1 — Data pipeline
- Build a manifest over the 369 BraTS patient folders, skipping/logging any incomplete folder.
- Split **at the patient level** into train/val/test (~70/15/15) with a fixed seed, persisted as ID
  lists so every later step reads the exact same split. Once created, this file should not change.
- Preprocessing per patient: z-score normalize each modality independently over nonzero (brain)
  voxels, remap raw BraTS labels `{0,1,2,4}` to a contiguous 4-class scheme for training, and define
  WT/TC/ET as standard *evaluation-time* unions of those classes.
- Slice each volume along the axial axis into 2D image/label pairs (4-channel stacked modality image),
  resized to a size compatible across architectures (224×224 default; DINOv2's patch-14 backbone may
  need a documented per-method override).
- Explicit negative-slice policy: keep every tumor-containing slice; keep only a subsampled fraction of
  tumor-free slices for training; validation/test always use every slice, unsampled.
- Cache processed 2D slices to disk once, rather than re-decoding NIfTI volumes on every epoch across
  5 separate training runs.
- Sanity-check the cache before trusting it: label histograms, tumor-fraction distribution, rendered
  slice+mask overlays.

### Phase 2 — Shared training/eval harness + UNet baseline (method 1)
- Build the shared, config-driven harness first, validated end-to-end using UNet before investing in
  more exotic architectures.
- Pieces: a config system (shared base + per-method override), a model registry (name → build
  function), a shared train/validate loop, shared Dice+CE loss (background excluded from the Dice
  term), and shared metrics (per-batch slice-level Dice during training; volume-level Dice + Hausdorff95
  — reconstructed from per-slice predictions — at validation/test time).
- fp16 mixed precision (the 1660 Ti has no native bf16 path) and gradient accumulation on by default.
- Centralized seeding; every run logged to one shared Weights & Biases project, mirroring
  `medical_understanding`'s conventions (including setting `HF_HOME`/`TORCH_HOME` before any
  model/backbone import).
- Baseline model: UNet with a narrower channel width than the typical nnU-Net default, sized to fit 6GB.

### Phase 3 — Method 2: resolve the "encoder-decoder method" ambiguity
- Concrete definition: the same UNet architecture/channel widths as method 1, **with skip connections
  removed** (pure encoder → bottleneck → decoder). Turns a vague step into a clean, single-variable
  ablation against method 1, demonstrating the effect of skip connections on boundary-precise
  segmentation.
- Also validates that the registry/config mechanism generalizes to a second architecture.

### Phase 4 — Method 3: SwinUNet
- A Swin-Tiny-scale transformer encoder with a UNet-style decoder (patch-expanding blocks + skip
  connections), sized down to fit 6GB.
- Natural complexity step between the CNN methods (1–2) and the fully transformer-backbone methods
  (4–5); surfaces transformer attention memory costs before the harder methods.

### Phase 5 — Methods 4 & 5: DINOv2 + DPT decoder, and plain ViT + the same DPT decoder
- Build the DPT-style decoder (multi-layer ViT feature reassembly → multi-scale spatial feature maps →
  progressive fusion/upsampling) **once**, shared by both methods.
- Backbones: DINOv2 ViT-S/14 for method 4, ImageNet-supervised ViT-S/16 (matched parameter scale) for
  method 5. Both **frozen by default**; fine-tuning either is a separate, explicitly labeled ablation.
- Explicit, documented decision needed for handling the 4-channel MRI input against RGB-pretrained
  backbones (either a 3-channel modality composite, or an adapted patch-embedding layer) — whichever is
  chosen must be applied identically to both methods.
- Build and debug methods 4 and 5 in the same phase to minimize the risk of one quietly getting a
  fix/tweak the other doesn't.

### Phase 6 — Ablations and multi-seed runs
- Frozen-vs-fine-tuned backbone ablation for methods 4 and 5.
- The skip-connection ablation (methods 1 vs. 2) is already produced for free — call it out explicitly.
- Multi-seed reruns for the top 1–2 models by mean Dice, reporting mean ± std per region.

### Phase 7 — Final comparison and reporting
- Unified results table: per-region Dice (WT/TC/ET), per-region Hausdorff95, trainable vs. total
  parameter count, peak VRAM usage, per-slice inference time, total training time.
- Qualitative comparison grid: the same small set of representative held-out slices, ground truth vs.
  each method's prediction, side by side.
- Closing summary answering the project's actual research question: does DINOv2's self-supervised
  pretraining outperform supervised ImageNet ViT pretraining under an identical decoder, and how does
  that gap compare to the UNet baseline and to the skip-connection ablation.

### Optional stretch (explicitly out of scope for the primary comparison)
- Full 3D patch-based training as a reference point for what's lost/gained vs. the 2D-slice approach —
  only after all 5 2D methods are complete and compared, and likely only feasible for the UNet baseline
  given the 6GB ceiling.

---

## Verification

Since this phase of work is infrastructure/process, "verification" means confirming each phase produces
trustworthy artifacts before moving to the next:
- Phase 0: environment check passes (CUDA visible, correct GPU detected, one real NIfTI file loads).
- Phase 1: split files are patient-disjoint across train/val/test; exploration notebook confirms
  slice/mask alignment and a sane tumor-fraction distribution.
- Phase 2: UNet baseline's validation Dice visibly improves over epochs (not flat/NaN), checkpoint
  save/resume works, wandb run shows sane curves.
- Phases 3–5: each new method trains without VRAM OOM at its planned batch size (profiled empirically),
  and produces a test-metrics artifact in the same format as the UNet baseline.
- Phase 6–7: the comparison table and qualitative grid are generated purely from each method's saved
  test-metrics artifacts, so the final report is reproducible from the run outputs.
