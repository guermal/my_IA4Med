# Shared utilities for brain segmentation: loss, LR schedule, visualization.

import math

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

BRATS_LABEL_NAMES = {
    1: "NCR/NET — necrotic & non-enhancing tumor core",
    2: "ED — peritumoral edema",
    4: "ET — GD-enhancing tumor",
}


# ----------------------------------------------------------------------------------------------- loss

def region_dice_bce_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Region-based loss for nested WT/TC/ET: BCE + batch soft Dice, one sigmoid channel per region.

    Args:
        logits: (B, 3, H, W) raw network outputs.
        target: (B, 3, H, W) float 0/1 region masks.
    Batch Dice sums over the whole batch before the ratio, so tumour-free slices don't produce
    the exploding per-sample Dice gradients; regions absent from the whole batch are left to BCE
    (as in FleXray). Computed in fp32 for AMP safety."""
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    intersection = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    dice = (2 * intersection + eps) / (denom + eps)
    present = target.sum(dims) > 0          # Dice only for regions present in the batch; BCE handles the rest
    dice_loss = (1 - dice[present]).mean() if present.any() else logits.new_zeros(())
    return bce + dice_loss


# ----------------------------------------------------------------------------------------------- schedule

def warmup_cosine(total_steps: int, warmup_frac: float = 0.05, min_ratio: float = 1e-3):
    """LambdaLR factor: linear warmup, then cosine decay to min_ratio * base LR."""
    warmup = max(1, int(total_steps * warmup_frac))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return factor


# ----------------------------------------------------------------------------------------------- visualization

def read_nifti_image(file_path):
    """Read a NIfTI file and return its data as a NumPy array."""
    img = nib.load(file_path).get_fdata()
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    return img


def vizualize_brain_segmentation(image_path: str, mask_path: str, slice_indices: list = None, n_slices: int = 4):
    """Show several MRI slices with the segmentation mask overlaid on each."""
    import SimpleITK as sitk
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    image_array = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    if slice_indices is None:
        slice_indices = np.linspace(0, image_array.shape[0] - 1, n_slices + 2, dtype=int)[1:-1]

    _, axes = plt.subplots(1, len(slice_indices), figsize=(5 * len(slice_indices), 5))
    axes = np.atleast_1d(axes)
    cmap, norm = plt.get_cmap('jet'), plt.Normalize(vmin=0, vmax=4)
    for ax, z in zip(axes, slice_indices):
        ax.imshow(image_array[z], cmap='gray')
        ax.imshow(np.ma.masked_where(mask_array[z] == 0, mask_array[z]), cmap=cmap, norm=norm, alpha=0.5)
        ax.set_title(f'Slice {z}')
        ax.axis('off')
    handles = [mpatches.Patch(color=cmap(norm(k)), label=v) for k, v in BRATS_LABEL_NAMES.items()]
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.figlegend(handles=handles, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 0.0))
    plt.show()


def vizualize_brain_segmentation_3d(image_path: str, mask_path: str):
    """Show an MRI volume and its segmentation in 3D with napari."""
    import SimpleITK as sitk
    import napari

    image_array = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(image_array, name="MRI", colormap="gray", rendering="mip")
    viewer.add_labels(mask_array.astype(np.int32), name="Segmentation")
    napari.run()


if __name__ == "__main__":
    t = torch.zeros(2, 3, 32, 32)
    t[:, :, 10:20, 10:20] = 1
    perfect = (t * 2 - 1) * 20                       # confident, correct logits
    assert region_dice_bce_loss(perfect, t) < 0.01
    assert region_dice_bce_loss(-perfect, t) > 1.5
    empty = torch.zeros_like(t)                      # tumour-free batch, model predicts nothing: no blow-up
    assert region_dice_bce_loss(torch.full_like(t, -20), empty) < 0.01
    f = warmup_cosine(100)
    assert f(0) < f(4) <= 1.0 and f(99) < 0.01
    print("utils self-check passed")
