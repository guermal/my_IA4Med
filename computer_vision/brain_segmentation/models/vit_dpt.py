# Methods 2, 4, 5: ViT-S backbone + the same DPT decoder. Only the backbone's pretraining differs:
#   dinov2 -> DINOv2 ViT-S/14 (method 4)
#   vit    -> ImageNet-supervised ViT-S/16 (method 5)
#   jepa   -> DINOv2 ViT-S/14 after I-JEPA continued pretraining on BraTS (method 2, see pretrain_jepa.py)
# 4-channel MRI input: timm adapts the RGB patch embedding identically for every backbone (repeat + rescale).

import os
os.environ.setdefault("HF_HOME", r"D:\huggingface_cache")   # must be set before timm imports huggingface_hub
try:  # use the OS cert store (certifi's bundle fails on some machines' HTTPS chains)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

BACKBONES = {
    "dinov2": "vit_small_patch14_dinov2.lvd142m",
    "jepa": "vit_small_patch14_dinov2.lvd142m",
    "vit": "vit_small_patch16_224.augreg_in21k_ft_in1k",
}
TAP_LAYERS = [2, 5, 8, 11]  # blocks 3, 6, 9, 12


def make_backbone(name: str, pretrained: bool = True):
    return timm.create_model(BACKBONES[name], pretrained=pretrained, in_chans=4, num_classes=0, img_size=224)


class ResidualConvUnit(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        return x + self.block(x)


class FusionBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.skip_rcu, self.rcu, self.out = ResidualConvUnit(c), ResidualConvUnit(c), nn.Conv2d(c, c, 1)

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False) + self.skip_rcu(skip)
        x = self.rcu(x)
        return self.out(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))


class DPTDecoder(nn.Module):
    """Reassemble 4 ViT layers into a 4x/2x/1x/0.5x pyramid, fuse coarse -> fine, predict per pixel."""

    def __init__(self, embed_dim=384, out_channels=3, dims=(64, 128, 256, 256), features=128):
        super().__init__()
        self.reassemble = nn.ModuleList([
            nn.Sequential(nn.Conv2d(embed_dim, dims[0], 1), nn.ConvTranspose2d(dims[0], dims[0], 4, stride=4)),
            nn.Sequential(nn.Conv2d(embed_dim, dims[1], 1), nn.ConvTranspose2d(dims[1], dims[1], 2, stride=2)),
            nn.Conv2d(embed_dim, dims[2], 1),
            nn.Sequential(nn.Conv2d(embed_dim, dims[3], 1), nn.Conv2d(dims[3], dims[3], 3, stride=2, padding=1)),
        ])
        self.project = nn.ModuleList([nn.Conv2d(d, features, 3, padding=1, bias=False) for d in dims])
        self.fusion = nn.ModuleList([FusionBlock(features) for _ in dims])
        self.head = nn.Sequential(nn.Conv2d(features, features // 2, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(features // 2, out_channels, 1))

    def forward(self, feats, out_size):
        f = [proj(re(x)) for x, re, proj in zip(feats, self.reassemble, self.project)]
        x = self.fusion[3](f[3])
        for i in (2, 1, 0):
            x = self.fusion[i](x, f[i])
        return F.interpolate(self.head(x), size=out_size, mode="bilinear", align_corners=False)


class ViTDPT(nn.Module):
    def __init__(self, backbone: str, frozen: bool = True, out_channels: int = 3, backbone_ckpt: str = None):
        super().__init__()
        self.backbone = make_backbone(backbone, pretrained=backbone_ckpt is None)
        if backbone_ckpt:
            self.backbone.load_state_dict(torch.load(backbone_ckpt, map_location="cpu"))
        self.frozen = frozen
        self.backbone.requires_grad_(not frozen)
        if not frozen:
            self.backbone.set_grad_checkpointing(True)
        self.decoder = DPTDecoder(self.backbone.embed_dim, out_channels)

    def forward(self, x):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            feats = self.backbone.forward_intermediates(x, indices=TAP_LAYERS, output_fmt="NCHW",
                                                        intermediates_only=True)
        return self.decoder(feats, x.shape[-2:])


if __name__ == "__main__":
    for name in ("dinov2", "vit"):
        m = ViTDPT(name, frozen=True)
        assert m(torch.randn(2, 4, 224, 224)).shape == (2, 3, 224, 224)
        n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{name}+DPT ok, trainable {n_train / 1e6:.1f}M / total {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M")
