# Method 1: 2D U-Net baseline. InstanceNorm instead of BatchNorm (small batches + grad accumulation).

import torch
import torch.nn as nn


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.InstanceNorm2d(c_out, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.InstanceNorm2d(c_out, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


class UNet(nn.Module):
    """Encoder 32-64-128-256, bottleneck 512, decoder with its own blocks (not the encoder's) and skips."""

    def __init__(self, in_channels=4, out_channels=3, base=32):
        super().__init__()
        widths = [base * 2 ** i for i in range(5)]                 # 32, 64, 128, 256, 512
        self.encoders = nn.ModuleList(
            [conv_block(in_channels, widths[0])] + [conv_block(widths[i], widths[i + 1]) for i in range(3)])
        self.bottleneck = conv_block(widths[3], widths[4])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in reversed(range(4))])
        self.decoders = nn.ModuleList([conv_block(widths[i] * 2, widths[i]) for i in reversed(range(4))])
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)


if __name__ == "__main__":
    m = UNet()
    assert m(torch.randn(2, 4, 224, 224)).shape == (2, 3, 224, 224)
    print(f"UNet ok, {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M params")
