# Evaluation metrics for BraTS-style brain tumor segmentation (Dice + Hausdorff95).

import torch


def brats_to_contiguous(label_map: torch.Tensor) -> torch.Tensor:
    """Remap raw BraTS labels {0,1,2,4} -> contiguous {0,1,2,3}, for training (CrossEntropyLoss/softmax
    need contiguous class indices matching the model's output channel count)."""
    remapped = label_map.clone()
    remapped[label_map == 4] = 3
    return remapped


def contiguous_to_brats(label_map: torch.Tensor) -> torch.Tensor:
    """Remap a model's contiguous {0,1,2,3} prediction (e.g. argmax output) back to raw BraTS
    labels {0,1,2,4}, so it's comparable to the ground-truth seg.nii, which is always raw-label space."""
    remapped = label_map.clone()
    remapped[label_map == 3] = 4
    return remapped


def derive_brats_regions(label_map: torch.Tensor) -> dict:
    """
    Derive the three standard BraTS evaluation regions from a raw-label-space label map.

    Args:
        label_map (torch.Tensor): integer labels {0,1,2,4} (background, NCR/NET, ED, ET), any shape.

    Returns:
        dict[str, torch.Tensor]: boolean masks — WT (whole tumor, labels 1+2+4),
            TC (tumor core, labels 1+4), ET (enhancing tumor, label 4).
    """
    return {
        "WT": label_map > 0,
        "TC": (label_map == 1) | (label_map == 4),
        "ET": label_map == 4,
    }


def dice_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, smooth: float = 1e-6) -> float:
    """Binary Dice score between two boolean masks of the same shape."""
    pred_mask, true_mask = pred_mask.float(), true_mask.float()
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum()
    return ((2. * intersection + smooth) / (union + smooth)).item()


def hausdorff95(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> float:
    """
    95th-percentile Hausdorff distance (voxel units) between two boolean masks.

    Undefined (returns NaN) if either mask is empty. Requires monai."""
    if pred_mask.sum() == 0 or true_mask.sum() == 0:
        return float('nan')
    from monai.metrics import compute_hausdorff_distance
    pred_b = pred_mask[None, None].float()
    true_b = true_mask[None, None].float()
    return compute_hausdorff_distance(pred_b, true_b, percentile=95).item()


def evaluate_segmentation(pred_label_map: torch.Tensor, true_label_map: torch.Tensor,
                           pred_is_contiguous: bool = False) -> dict:
    """
    Compute per-region Dice and HD95 between a predicted and ground-truth BraTS label map.

    Args:
        pred_label_map (torch.Tensor): model prediction (e.g. argmax output).
        true_label_map (torch.Tensor): ground truth from seg.nii — always raw BraTS labels {0,1,2,4}.
        pred_is_contiguous (bool): set True if pred_label_map came straight from a model trained on
            the remapped {0,1,2,3} scheme (see brats_to_contiguous) — it's converted back to raw
            BraTS labels {0,1,2,4} before comparison, so mixing the two label spaces can't silently
            zero out the ET score.

    Returns:
        dict: {"WT": {"dice": ..., "hd95": ...}, "TC": {...}, "ET": {...}}"""
    if pred_is_contiguous:
        pred_label_map = contiguous_to_brats(pred_label_map)

    pred_regions = derive_brats_regions(pred_label_map)
    true_regions = derive_brats_regions(true_label_map)
    return {
        region: {
            "dice": dice_score(pred_regions[region], true_regions[region]),
            "hd95": hausdorff95(pred_regions[region], true_regions[region]),
        }
        for region in pred_regions
    }


if __name__ == "__main__":
    raw = torch.tensor([0, 1, 2, 4, 4, 0])
    regions = derive_brats_regions(raw)
    assert regions["WT"].tolist() == [False, True, True, True, True, False]
    assert regions["TC"].tolist() == [False, True, False, True, True, False]
    assert regions["ET"].tolist() == [False, False, False, True, True, False]
    assert dice_score(regions["WT"], regions["WT"]) == 1.0

    disjoint = torch.tensor([True, False])
    other = torch.tensor([False, True])
    assert dice_score(disjoint, other) < 1e-3

    contiguous = brats_to_contiguous(raw)
    assert contiguous.tolist() == [0, 1, 2, 3, 3, 0]
    assert contiguous_to_brats(contiguous).tolist() == raw.tolist()

    result = evaluate_segmentation(contiguous, raw, pred_is_contiguous=True)
    assert result["ET"]["dice"] == 1.0, "contiguous->raw remap must round-trip before region comparison"

    try:
        print("HD95 (identical masks, expect 0):", hausdorff95(regions["WT"], regions["WT"]))
    except ImportError:
        print("skipped hd95 self-check: monai not installed")

    print("metrics.py self-check passed")
