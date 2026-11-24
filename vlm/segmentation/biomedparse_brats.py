# VLM-side segmentation: zero-shot, text-prompted BiomedParse v2 (3D) on our BraTS2020 split, scored with
# the SAME split and metrics as the computer-vision models, so it can sit in the same results table.
#
# BiomedParse runs in its own env (pinned torch 2.6 / timm 0.9 clash with ours), so this file only does the
# two ends:
#   1. export : BraTS patients -> BiomedParse v2 input NPZs  (imgs (Z,H,W) in [0,255] + text_prompts)
#   2. score  : BiomedParse output NPZs (key 'segs', label map) -> WT/TC/ET Dice + HD95 json
# In between you run their `inference.py` (see README.md).
#
#   python biomedparse_brats.py export --split test --sequence t1ce
#   python biomedparse_brats.py score  --split test --pred-dir outputs/t1ce
#
# CAVEAT: BiomedParse training data lists BraTS 2023, which includes most BraTS 2020 patients, so these
# numbers are possibly contaminated (not true zero-shot) and are reported as a reference row only.

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

CV_DIR = Path(__file__).resolve().parents[2] / "computer_vision" / "brain_segmentation"
sys.path.insert(0, str(CV_DIR))
from data import PatientVolume, load_index, seg_to_regions  # noqa: E402
from metrics import dice_score, hausdorff95                   # noqa: E402

HERE = Path(__file__).parent
REGIONS = ("WT", "TC", "ET")
# Prompt wording from BiomedParse v2's training vocabulary for BraTS (CVPR25_TextSegFMData_with_class.json),
# which uses BraTS-2024 label names. They map onto BraTS2020 labels: 1=NCR/NET, 2=ED, 4=ET.
PROMPTS = {"1": "Non-enhancing tumor core",
           "2": "Surrounding non-enhancing FLAIR hyperintensity",
           "3": "Enhancing tissue",
           "instance_label": 0}
# predicted prompt id -> BraTS2020 label
PRED_TO_BRATS = {1: 1, 2: 2, 3: 4}
SEQUENCES = ("flair", "t1", "t1ce", "t2")


def to_uint8(vol: np.ndarray) -> np.ndarray:
    """Raw MRI -> [0, 255]: clip brain voxels to their 0.5-99.5 percentiles, rescale, background stays 0.
    ponytail: the challenge only documents 'intensity range [0, 255]', not its exact MRI recipe; this
    percentile scheme is the usual MedSAM-style choice. Swap it if their preprocessing turns out different."""
    brain = vol > 0
    lo, hi = np.percentile(vol[brain], (0.5, 99.5))
    out = np.clip((vol - lo) / max(hi - lo, 1e-8), 0, 1) * 255
    out[~brain] = 0
    return out.astype(np.uint8)


def export(split: str, sequence: str, out_dir: Path, limit=None):
    """One NPZ per patient, one MRI sequence per file (BiomedParse is single-channel; our CV models see all 4)."""
    index = load_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_i = SEQUENCES.index(sequence)
    for n, pid in enumerate(index["split"][split][:limit]):
        vol = PatientVolume(pid, index["patients"][pid])
        raw = np.asarray(vol.imgs[seq_i], dtype=np.float32)              # (240, 240, Z)
        imgs = np.transpose(to_uint8(raw), (2, 0, 1))                     # (Z, H, W) as BiomedParse expects
        np.savez_compressed(out_dir / f"{pid}.npz", imgs=imgs, text_prompts=PROMPTS)
        print(f"[{n + 1}] {pid} {imgs.shape}")


def segs_to_regions(segs: np.ndarray) -> torch.Tensor:
    """BiomedParse label map (Z, H, W) with prompt ids -> bool (3, H, W, Z) WT/TC/ET, our layout."""
    brats = np.zeros_like(segs, dtype=np.int64)
    for pred_id, label in PRED_TO_BRATS.items():
        brats[segs == pred_id] = label
    return seg_to_regions(np.transpose(brats, (1, 2, 0))).bool()


def score(split: str, pred_dir: Path, out_json: Path):
    index = load_index()
    per_patient, missing = {}, []
    for pid in index["split"][split]:
        f = pred_dir / f"{pid}.npz"
        if not f.exists():
            missing.append(pid)
            continue
        pred = segs_to_regions(np.load(f)["segs"])
        gt = seg_to_regions(np.asarray(PatientVolume(pid, index["patients"][pid]).seg)).bool()
        assert pred.shape == gt.shape, f"{pid}: pred {tuple(pred.shape)} vs gt {tuple(gt.shape)}"
        per_patient[pid] = {r: {"dice": dice_score(pred[k], gt[k]), "hd95": hausdorff95(pred[k], gt[k])}
                            for k, r in enumerate(REGIONS)}
    if not per_patient:
        raise SystemExit(f"no predictions found in {pred_dir}")
    summary = {r: {m: float(np.nanmean([p[r][m] for p in per_patient.values()])) for m in ("dice", "hd95")}
               for r in REGIONS}
    summary["mean_dice"] = float(np.mean([summary[r]["dice"] for r in REGIONS]))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"summary": summary, "per_patient": per_patient, "missing": missing,
                                    "note": "BiomedParse v2 zero-shot; possibly contaminated (BraTS 2023 in "
                                            "its training data)"}, indent=1))
    print(json.dumps(summary, indent=1), f"\n{len(per_patient)} scored, {len(missing)} missing -> {out_json}")


def self_check():
    """Round trip: GT -> fake BiomedParse output -> score must give Dice 1 on every region."""
    import tempfile
    index = load_index()
    pid = index["split"]["test"][0]
    seg = np.asarray(PatientVolume(pid, index["patients"][pid]).seg)
    fake = np.zeros(seg.shape, dtype=np.uint8)
    for pred_id, label in PRED_TO_BRATS.items():
        fake[seg == label] = pred_id
    with tempfile.TemporaryDirectory() as d:
        np.savez_compressed(Path(d) / f"{pid}.npz", segs=np.transpose(fake, (2, 0, 1)))
        score("test", Path(d), Path(d) / "m.json")
        s = json.loads((Path(d) / "m.json").read_text())["summary"]
    assert all(abs(s[r]["dice"] - 1) < 1e-6 for r in REGIONS), s
    print("biomedparse_brats self-check passed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--split", default="test")
    e.add_argument("--sequence", default="t1ce", choices=SEQUENCES)
    e.add_argument("--limit", type=int, default=None)
    s = sub.add_parser("score")
    s.add_argument("--split", default="test")
    s.add_argument("--pred-dir", type=Path, required=True)
    sub.add_parser("check")
    a = p.parse_args()
    if a.cmd == "export":
        export(a.split, a.sequence, HERE / "inputs" / a.sequence, a.limit)
    elif a.cmd == "score":
        score(a.split, a.pred_dir, HERE / "results" / f"metrics_{a.split}_{a.pred_dir.name}.json")
    else:
        self_check()
