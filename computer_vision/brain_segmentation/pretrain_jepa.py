# Method 2, stage 1: I-JEPA continued pretraining of DINOv2 ViT-S/14 on BraTS training slices (no labels).
#
# Context encoder (trained) sees a large block minus the targets; target encoder (EMA copy, no grad) sees
# the full slice; a narrow predictor predicts the target encoder's embeddings of 4 target blocks from the
# context. Loss is in embedding space (smooth L1, as in the I-JEPA reference code), never in pixels.
#
#   python pretrain_jepa.py                         -> runs/jepa_pretrain/backbone.pt
#   python train.py --model jepa --backbone-ckpt runs/jepa_pretrain/backbone.pt

import os
os.environ.setdefault("HF_HOME", r"D:\huggingface_cache")
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from timm.models.vision_transformer import Block

from data import BratsSlices
from models.vit_dpt import make_backbone
from utils import warmup_cosine


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5, help="low: continued pretraining, don't wipe DINOv2")
    p.add_argument("--predictor-lr", type=float, default=5e-4, help="predictor is new, learns faster")
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--ema", type=float, nargs=2, default=(0.996, 1.0))
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default="jepa_pretrain")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--amp", action="store_true", help="fp16 autocast; off by default (NaN convs on GTX 1660 Ti)")
    return p.parse_args()


# ---------------------------------------------------------------------------------------------- masking

def block_size(scale, aspect, grid):
    n = grid * grid * random.uniform(*scale)
    ar = math.exp(random.uniform(math.log(aspect[0]), math.log(aspect[1])))
    return max(1, min(grid, round(math.sqrt(n * ar)))), max(1, min(grid, round(math.sqrt(n / ar))))


def sample_block(brain_tokens, h, w, grid):
    """Per sample, an (h, w) block whose top-left corner is drawn among positions where the block is
    mostly brain. Returns token indices (B, h*w)."""
    cover = F.avg_pool2d(brain_tokens[:, None], (h, w), stride=1)[:, 0]   # (B, grid-h+1, grid-w+1)
    rows, cols = torch.arange(h), torch.arange(w)
    out = []
    for c in cover:
        cand = (c > 0.5).nonzero()
        if len(cand):
            r, q = cand[random.randrange(len(cand))].tolist()
        else:
            r, q = random.randrange(c.shape[0]), random.randrange(c.shape[1])
        out.append(((r + rows)[:, None] * grid + (q + cols)[None, :]).flatten())
    return torch.stack(out)


def make_masks(x, grid, patch, n_targets=4):
    """I-JEPA multi-block masking. Targets: 4 blocks, 15-20% each, aspect 0.75-1.5, inside brain.
    Context: one 85-100% block minus all targets, truncated to the batch minimum so it stacks."""
    brain_tokens = (F.avg_pool2d((x != 0).any(1, keepdim=True).float(), patch)[:, 0] > 0.5).float().cpu()
    h, w = block_size((0.15, 0.2), (0.75, 1.5), grid)          # one target size per batch, as in I-JEPA
    targets = [sample_block(brain_tokens, h, w, grid) for _ in range(n_targets)]
    context = sample_block(torch.ones_like(brain_tokens), *block_size((0.85, 1.0), (1.0, 1.0), grid), grid)
    keep = []
    for b in range(x.shape[0]):
        blocked = torch.cat([t[b] for t in targets])
        ctx = context[b][~torch.isin(context[b], blocked)]
        keep.append(ctx[torch.randperm(len(ctx))])
    n_min = min(len(k) for k in keep)
    context = torch.stack([k[:n_min].sort().values for k in keep])
    return context, targets


# ---------------------------------------------------------------------------------------------- model

def gather_tokens(x, idx):
    return torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1]))


def encode(vit, img, keep=None):
    """timm ViT forward on a subset of patch tokens (cls token kept, as in fine-tuning). Returns patch tokens."""
    x = vit._pos_embed(vit.patch_embed(img))
    prefix, patches = x[:, :vit.num_prefix_tokens], x[:, vit.num_prefix_tokens:]
    if keep is not None:
        patches = gather_tokens(patches, keep)
    x = vit.norm_pre(torch.cat([prefix, patches], 1))
    for blk in vit.blocks:
        x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False) if x.requires_grad else blk(x)
    return vit.norm(x)[:, vit.num_prefix_tokens:]


class Predictor(nn.Module):
    """Narrow ViT: context embeddings + positional mask tokens -> predicted target embeddings."""

    def __init__(self, n_tokens, embed_dim=384, dim=192, depth=6, heads=6):
        super().__init__()
        self.inp = nn.Linear(embed_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, embed_dim)

    def forward(self, ctx, ctx_idx, tgt_idx):
        B = ctx.shape[0]
        pos = self.pos.expand(B, -1, -1)
        x = self.inp(ctx) + gather_tokens(pos, ctx_idx)
        m = self.mask_token.expand(B, tgt_idx.shape[1], -1) + gather_tokens(pos, tgt_idx)
        x = torch.cat([x, m], 1)
        for blk in self.blocks:
            x = blk(x)
        return self.out(self.norm(x[:, -tgt_idx.shape[1]:]))


# ---------------------------------------------------------------------------------------------- train

def main():
    a = parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda"
    run_dir = Path(__file__).parent / "runs" / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    encoder = make_backbone("jepa").to(device)
    target = copy.deepcopy(encoder).requires_grad_(False)
    patch = encoder.patch_embed.patch_size[0]
    grid = 224 // patch
    predictor = Predictor(grid * grid, encoder.embed_dim).to(device)

    def groups(module, lr):
        return [{"params": [p for p in module.parameters() if p.ndim > 1], "lr": lr, "weight_decay": a.weight_decay},
                {"params": [p for p in module.parameters() if p.ndim <= 1], "lr": lr, "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups(encoder, a.lr) + groups(predictor, a.predictor_lr))

    ds = BratsSlices("train", neg_fraction=1.0, augment=True)   # train patients only: no val/test leakage
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True,
                        drop_last=True, persistent_workers=a.workers > 0)
    steps_per_epoch = min(len(loader), a.max_steps or len(loader)) // a.accum
    total = steps_per_epoch * a.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(total))
    scaler = torch.amp.GradScaler(enabled=a.amp)
    if a.wandb:
        import wandb
        wandb.init(project="brats-segmentation-comparison", name=a.run_name, config=vars(a))
    (run_dir / "config.json").write_text(json.dumps(vars(a), indent=1))

    opt_step = 0
    for epoch in range(a.epochs):
        t0, running, std_pred, std_tgt = time.time(), 0.0, 0.0, 0.0
        for step, (x, _) in enumerate(loader):
            if a.max_steps and step >= a.max_steps:
                break
            x = x.to(device, non_blocking=True)
            ctx_idx, tgt_idx = make_masks(x, grid, patch)
            ctx_idx, tgt_idx = ctx_idx.to(device), [t.to(device) for t in tgt_idx]
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
                with torch.no_grad():
                    h = F.layer_norm(encode(target, x).float(), (encoder.embed_dim,))
                ctx = encode(encoder, x, ctx_idx)
                # one predictor pass per target block, batched by repeating the context
                M = len(tgt_idx)
                pred = predictor(ctx.repeat(M, 1, 1), ctx_idx.repeat(M, 1), torch.cat(tgt_idx))
            tgt = torch.cat([gather_tokens(h, t) for t in tgt_idx])
            loss = F.smooth_l1_loss(pred.float(), tgt)
            scaler.scale(loss / a.accum).backward()
            running += loss.item()
            std_pred += pred.float().std(0).mean().item()   # collapse monitor: should not go to ~0
            std_tgt += tgt.std(0).mean().item()
            if (step + 1) % a.accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                opt_step += 1
                m = a.ema[0] + (a.ema[1] - a.ema[0]) * opt_step / max(1, total)
                with torch.no_grad():
                    for pt, pe in zip(target.parameters(), encoder.parameters()):
                        pt.mul_(m).add_(pe.detach(), alpha=1 - m)
        n = step + 1
        log = {"epoch": epoch, "jepa/loss": running / n, "jepa/std_pred": std_pred / n,
               "jepa/std_target": std_tgt / n, "epoch_seconds": time.time() - t0,
               "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
        print(json.dumps(log))
        if a.wandb:
            wandb.log(log)
        # the target (EMA) encoder is what I-JEPA evaluates downstream
        torch.save(target.state_dict(), run_dir / "backbone.pt")
        torch.save({"encoder": encoder.state_dict(), "target": target.state_dict(),
                    "predictor": predictor.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch}, run_dir / "last.pt")


if __name__ == "__main__":
    main()
