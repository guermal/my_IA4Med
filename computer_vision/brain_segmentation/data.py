# BraTS2020 data pipeline: patient-level split + 2D axial slice datasets.
#
# BraTS .nii files are uncompressed, so nibabel memory-maps them and reading one axial slice is cheap.
# We therefore don't cache slices to disk; we only cache a small index (split, per-patient z-score stats,
# which slices contain brain / tumor) in cache/index.json, built once by `python data.py`.

import json
import os
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from metrics import derive_brats_regions

DATA_ROOT = Path(os.environ.get(
    "BRATS_ROOT", r"D:\brain_tumor\brats\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"))
INDEX_PATH = Path(__file__).parent / "cache" / "index.json"
MODALITIES = ("flair", "t1", "t1ce", "t2")
CROP = slice(8, 232)  # 240 -> 224 center crop; BraTS brains fit inside it (ponytail: fixed crop, no per-volume bbox)
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SEED = 42


def patient_files(pdir: Path) -> dict:
    """Map modality -> path, plus 'seg'. The seg is whatever .nii isn't a modality (patient 355's seg is
    misnamed 'W39_1998.09.19_Segm.nii' in the official release). Ignores the stray .nii.gz duplicates."""
    files = {}
    for f in pdir.glob("*.nii"):
        mod = next((m for m in MODALITIES if f.name.lower().endswith(f"_{m}.nii")), "seg")
        files[mod] = f
    return files


def build_index():
    """Scan all patients once: patient-level split, per-modality brain mean/std, slice lists."""
    patients = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
    rng = random.Random(SEED)
    shuffled = patients[:]
    rng.shuffle(shuffled)
    n_train = round(len(shuffled) * SPLIT_RATIOS[0])
    n_val = round(len(shuffled) * SPLIT_RATIOS[1])
    split = {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train:n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val:]),
    }

    stats = {}
    for i, pid in enumerate(patients):
        files = patient_files(DATA_ROOT / pid)
        if len(files) != 5:
            print(f"skipping {pid}: incomplete ({sorted(files)})")
            continue
        vols = [np.asarray(nib.load(files[m]).dataobj, dtype=np.float32) for m in MODALITIES]
        brain = np.any(np.stack(vols) > 0, axis=0)
        seg = np.asarray(nib.load(files["seg"]).dataobj)
        brain_frac = brain[CROP, CROP].mean(axis=(0, 1))
        stats[pid] = {
            "mean": [float(v[brain].mean()) for v in vols],
            "std": [float(v[brain].std()) for v in vols],
            "brain_slices": [int(z) for z in np.nonzero(brain_frac > 0.05)[0]],
            "tumor_slices": [int(z) for z in np.nonzero((seg > 0).any(axis=(0, 1)))[0]],
        }
        print(f"[{i + 1}/{len(patients)}] {pid}")

    for k in split:
        split[k] = [p for p in split[k] if p in stats]
    assert not set(split["train"]) & set(split["val"]) and not set(split["train"]) & set(split["test"]) \
        and not set(split["val"]) & set(split["test"]), "split is not patient-disjoint"
    INDEX_PATH.parent.mkdir(exist_ok=True)
    INDEX_PATH.write_text(json.dumps({"seed": SEED, "split": split, "patients": stats}, indent=1))
    print({k: len(v) for k, v in split.items()})


def load_index() -> dict:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"{INDEX_PATH} missing; run `python data.py` once to build it")
    return json.loads(INDEX_PATH.read_text())


def seg_to_regions(seg: np.ndarray) -> torch.Tensor:
    """Raw BraTS labels (..., H, W) -> float (3, ..., H, W) nested regions WT, TC, ET."""
    r = derive_brats_regions(torch.from_numpy(seg.astype(np.int64)))
    return torch.stack([r["WT"], r["TC"], r["ET"]]).float()


class PatientVolume:
    """Memory-mapped access to one patient's 4 modalities + seg, z-score normalised over brain voxels."""

    def __init__(self, pid: str, info: dict):
        files = patient_files(DATA_ROOT / pid)
        self.imgs = [nib.load(files[m]).dataobj for m in MODALITIES]
        self.seg = nib.load(files["seg"]).dataobj
        self.mean, self.std = info["mean"], info["std"]

    def image(self, z) -> np.ndarray:
        """(4, H, W) for an int z, or (4, H, W, Z) for a slice of z; background stays 0."""
        out = []
        for img, m, s in zip(self.imgs, self.mean, self.std):
            a = np.asarray(img[:, :, z], dtype=np.float32)
            out.append(np.where(a > 0, (a - m) / (s + 1e-8), 0.0))
        return np.stack(out).astype(np.float32)

    def label(self, z) -> np.ndarray:
        return np.asarray(self.seg[:, :, z])


class BratsSlices(Dataset):
    """2D axial slices for training. Returns x (4,224,224) and y (3,224,224) = WT/TC/ET.

    Keeps every tumour slice and `neg_fraction` of the tumour-free brain slices (fixed with the seed, so all
    methods see the same set). Pass neg_fraction=1.0 for unlabeled pretraining (I-JEPA) to use every slice."""

    def __init__(self, split: str = "train", neg_fraction: float = 0.3, augment: bool = True):
        index = load_index()
        rng = random.Random(SEED)
        self.info = index["patients"]
        self.augment = augment
        self.items = []
        for pid in index["split"][split]:
            tumor = set(self.info[pid]["tumor_slices"])
            for z in self.info[pid]["brain_slices"]:
                if z in tumor or rng.random() < neg_fraction:
                    self.items.append((pid, z))
        self._vols = {}

    def __len__(self):
        return len(self.items)

    def _vol(self, pid):
        # ponytail: unbounded per-worker cache of nibabel proxies (headers only, a few KB each), fine for 369 patients
        if pid not in self._vols:
            self._vols[pid] = PatientVolume(pid, self.info[pid])
        return self._vols[pid]

    def __getitem__(self, i):
        pid, z = self.items[i]
        vol = self._vol(pid)
        x = torch.from_numpy(vol.image(z)[:, CROP, CROP])
        y = seg_to_regions(vol.label(z)[CROP, CROP])
        if self.augment:
            for dim in (1, 2):
                if random.random() < 0.5:
                    x, y = x.flip(dim), y.flip(dim)
        return x, y


def iter_volumes(split: str):
    """Yield (pid, x (Z,4,224,224) tensor, gt regions (3,240,240,Z) bool, brain mask (240,240,Z) bool) per
    patient, for volume-level evaluation at full 240x240 resolution."""
    index = load_index()
    for pid in index["split"][split]:
        vol = PatientVolume(pid, index["patients"][pid])
        img = vol.image(slice(None))                       # (4, 240, 240, Z)
        x = torch.from_numpy(img[:, CROP, CROP]).permute(3, 0, 1, 2).contiguous()
        gt = seg_to_regions(vol.label(slice(None))).bool()
        brain = torch.from_numpy((img != 0).any(axis=0))
        yield pid, x, gt, brain


if __name__ == "__main__":
    build_index()
    ds = BratsSlices("train")
    x, y = ds[0]
    assert x.shape == (4, 224, 224) and y.shape == (3, 224, 224)
    assert (y[2] <= y[1]).all() and (y[1] <= y[0]).all(), "regions must be nested ET <= TC <= WT"
    print(f"train slices: {len(ds)}  sample ok")
