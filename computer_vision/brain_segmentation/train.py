# Shared train / eval harness for all methods. Same data, split, loss, schedule and metrics for every model.
#
#   python train.py --model unet
#   python train.py --model dinov2                    (frozen backbone, DPT decoder only)
#   python train.py --model dinov2 --finetune         (backbone fine-tuned with layer-wise LR decay)
#   python train.py --model jepa --backbone-ckpt runs/jepa_pretrain/backbone.pt
#   python train.py --model unet --eval-only --split test --resume runs/unet/best.pt

import os
os.environ.setdefault("HF_HOME", r"D:\huggingface_cache")
os.environ.setdefault("TORCH_HOME", r"D:\huggingface_cache\torch")
try:  # use the Windows cert store (certifi fails on this machine's HTTPS chain)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import BratsSlices, iter_volumes, CROP
from metrics import dice_score, hausdorff95
from models.unet_models import UNet
from models.vit_dpt import ViTDPT
from utils import region_dice_bce_loss, warmup_cosine

REGIONS = ("WT", "TC", "ET")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["unet", "swinunetr", "dinov2", "vit", "jepa"])
    p.add_argument("--finetune", action="store_true", help="ViT methods: train the backbone too (default frozen)")
    p.add_argument("--backbone-ckpt", help="jepa: backbone weights from pretrain_jepa.py")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum", type=int, default=4, help="gradient accumulation steps (effective batch = bs*accum)")
    p.add_argument("--lr", type=float, default=None, help="default: 3e-4 unet, 1e-4 swinunetr, 1e-3 decoder")
    p.add_argument("--backbone-lr", type=float, default=5e-5)
    p.add_argument("--layer-decay", type=float, default=0.75)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--neg-fraction", type=float, default=0.3, help="share of tumour-free train slices kept")
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None, help="cap steps per epoch (smoke tests)")
    p.add_argument("--max-val-patients", type=int, default=None, help="cap val volumes (smoke tests)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", help="checkpoint to resume from (or to evaluate with --eval-only)")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--split", default="test", help="split for --eval-only")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--amp", action="store_true",
                   help="fp16 autocast. Off by default: on the GTX 1660 Ti cuDNN's fp16 convs return NaN and fp32 "
                        "is just as fast (no tensor cores). Turn on for GPUs with tensor cores.")
    return p.parse_args()


def build_model(a):
    if a.model == "unet":
        return UNet()
    if a.model == "swinunetr":
        from monai.networks.nets import SwinUNETR
        return SwinUNETR(in_channels=4, out_channels=3, spatial_dims=2, feature_size=48, use_checkpoint=True)
    return ViTDPT(a.model, frozen=not a.finetune, backbone_ckpt=a.backbone_ckpt)


def param_groups(model, a):
    """AdamW groups: no weight decay on biases/norms/embeddings; layer-wise LR decay on a fine-tuned ViT."""
    groups = []
    backbone = getattr(model, "backbone", None)
    depth = len(backbone.blocks) if backbone is not None else 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        wd = 0.0 if p.ndim <= 1 or "pos_embed" in name or "token" in name else a.weight_decay
        lr = a.lr
        if name.startswith("backbone."):
            if ".blocks." in name:
                layer = int(name.split(".blocks.")[1].split(".")[0]) + 1
            elif ".norm." in name or name.startswith("backbone.norm"):
                layer = depth + 1
            else:                                   # patch_embed, cls_token, pos_embed
                layer = 0
            lr = a.backbone_lr * a.layer_decay ** (depth + 1 - layer)
        groups.append({"params": [p], "lr": lr, "weight_decay": wd})
    return groups


@torch.no_grad()
def evaluate(model, split, device, max_patients=None, with_hd95=False, amp=False):
    """Volume-level metrics: predict every axial slice, paste back into 240x240, mask to brain, then Dice
    (and optionally HD95) per region on the full 3D volume."""
    model.eval()
    per_patient = {}
    for n, (pid, x, gt, brain) in enumerate(iter_volumes(split)):
        if max_patients and n >= max_patients:
            break
        probs = []
        for i in range(0, len(x), 16):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                probs.append(torch.sigmoid(model(x[i:i + 16].to(device)).float()).cpu())
        pred_crop = torch.cat(probs).permute(1, 2, 3, 0) > 0.5           # (3, 224, 224, Z)
        pred = torch.zeros_like(gt)
        pred[:, CROP, CROP] = pred_crop
        pred &= brain
        per_patient[pid] = {}
        for k, r in enumerate(REGIONS):
            m = {"dice": dice_score(pred[k], gt[k])}
            if with_hd95:
                m["hd95"] = hausdorff95(pred[k], gt[k])
            per_patient[pid][r] = m
    summary = {r: {key: float(np.nanmean([p[r][key] for p in per_patient.values()]))
                   for key in per_patient[next(iter(per_patient))][r]} for r in REGIONS}
    summary["mean_dice"] = float(np.mean([summary[r]["dice"] for r in REGIONS]))
    model.train()
    return summary, per_patient


def main():
    a = parse_args()
    if a.lr is None:
        a.lr = {"unet": 3e-4, "swinunetr": 1e-4}.get(a.model, 1e-3)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda"
    run_name = a.run_name or f"{a.model}{'_ft' if a.finetune else ''}_s{a.seed}"
    run_dir = Path(__file__).parent / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(a).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{run_name}: {n_train / 1e6:.2f}M trainable / {n_total / 1e6:.2f}M total params")

    if a.eval_only:
        model.load_state_dict(torch.load(a.resume, map_location=device)["model"])
        t0 = time.time()
        summary, per_patient = evaluate(model, a.split, device, a.max_val_patients, with_hd95=True, amp=a.amp)
        summary["seconds"] = time.time() - t0
        out = run_dir / f"metrics_{a.split}.json"
        out.write_text(json.dumps({"summary": summary, "per_patient": per_patient,
                                   "params_total": n_total, "params_trainable": n_train}, indent=1))
        print(json.dumps(summary, indent=1), f"\nsaved {out}")
        return

    train_ds = BratsSlices("train", neg_fraction=a.neg_fraction)
    loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                        pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    steps_per_epoch = min(len(loader), a.max_steps or len(loader)) // a.accum
    optimizer = torch.optim.AdamW(param_groups(model, a))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(steps_per_epoch * a.epochs))
    scaler = torch.amp.GradScaler(enabled=a.amp)
    start_epoch, best = 0, -1.0

    if a.resume:
        ck = torch.load(a.resume, map_location=device)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]

    if a.wandb:
        import wandb
        wandb.init(project="brats-segmentation-comparison", name=run_name, config=vars(a))
    (run_dir / "config.json").write_text(json.dumps(vars(a), indent=1))
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, a.epochs):
        model.train()
        t0, running = time.time(), 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(loader):
            if a.max_steps and step >= a.max_steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
                logits = model(x)
            loss = region_dice_bce_loss(logits, y)
            scaler.scale(loss / a.accum).backward()
            running += loss.item()
            if (step + 1) % a.accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        n_steps = step + 1
        log = {"epoch": epoch, "train/loss": running / n_steps, "lr": optimizer.param_groups[-1]["lr"],
               "epoch_seconds": time.time() - t0, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}

        if (epoch + 1) % a.val_every == 0 or epoch == a.epochs - 1:
            summary, _ = evaluate(model, "val", device, a.max_val_patients, amp=a.amp)
            log.update({f"val/{r}_dice": summary[r]["dice"] for r in REGIONS})
            log["val/mean_dice"] = summary["mean_dice"]
            if summary["mean_dice"] > best:
                best = summary["mean_dice"]
                torch.save({"model": model.state_dict(), "epoch": epoch}, run_dir / "best.pt")
        print(json.dumps(log))
        if a.wandb:
            wandb.log(log)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best}, run_dir / "last.pt")
        with open(run_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")


if __name__ == "__main__":
    main()
