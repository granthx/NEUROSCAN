"""
Model architectures for brain tumor segmentation.
Includes:
- Standard 2D U-Net (Baseline)
- Attention U-Net (with Additive Attention Gates)
- 2.5D U-Net (multi-slice context: 12 channels for 3 slices x 4 modalities)
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """
    Standard double convolution block: [Conv -> BatchNorm -> ReLU] x 2
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """
    Standard 2D U-Net with configurable base channels and depth.
    """

    def __init__(self, in_ch: int = 4, n_cls: int = 4, base: int = 16) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.n_cls = n_cls
        self.base = base

        f: List[int] = [base, base * 2, base * 4, base * 8, base * 16]

        self.enc0 = conv_block(in_ch, f[0])
        self.enc1 = conv_block(f[0], f[1])
        self.enc2 = conv_block(f[1], f[2])
        self.enc3 = conv_block(f[2], f[3])
        self.enc4 = conv_block(f[3], f[4])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up3 = nn.ConvTranspose2d(f[4], f[3], kernel_size=2, stride=2)
        self.dec3 = conv_block(f[4], f[3])

        self.up2 = nn.ConvTranspose2d(f[3], f[2], kernel_size=2, stride=2)
        self.dec2 = conv_block(f[3], f[2])

        self.up1 = nn.ConvTranspose2d(f[2], f[1], kernel_size=2, stride=2)
        self.dec1 = conv_block(f[2], f[1])

        self.up0 = nn.ConvTranspose2d(f[1], f[0], kernel_size=2, stride=2)
        self.dec0 = conv_block(f[1], f[0])

        self.head = nn.Conv2d(f[0], n_cls, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.enc0(x)
        p0 = self.pool(x0)

        x1 = self.enc1(p0)
        p1 = self.pool(x1)

        x2 = self.enc2(p1)
        p2 = self.pool(x2)

        x3 = self.enc3(p2)
        p3 = self.pool(x3)

        x4 = self.enc4(p3)

        # Decoder with skip connections
        d3 = self.up3(x4)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        d0 = self.up0(d1)
        d0 = self.dec0(torch.cat([d0, x0], dim=1))

        return self.head(d0)


class AttentionGate(nn.Module):
    """
    Additive Attention Gate (Oktay et al., 2018).
    Filters features propagated through the skip connection using gating signal
    from coarser resolution.
    """

    def __init__(self, f_g: int, f_x: int, f_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_x, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Align spatial dimensions if needed
        g_proj = self.W_g(g)
        x_proj = self.W_x(x)
        if g_proj.shape[2:] != x_proj.shape[2:]:
            g_proj = F.interpolate(g_proj, size=x_proj.shape[2:], mode="bilinear", align_corners=False)

        f = self.relu(g_proj + x_proj)
        alpha = self.psi(f)
        return x * alpha


class AttentionUNet(nn.Module):
    """
    2D Attention U-Net incorporating additive attention gates on skip connections.
    """

    def __init__(self, in_ch: int = 4, n_cls: int = 4, base: int = 16) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.n_cls = n_cls
        self.base = base

        f: List[int] = [base, base * 2, base * 4, base * 8, base * 16]

        self.enc0 = conv_block(in_ch, f[0])
        self.enc1 = conv_block(f[0], f[1])
        self.enc2 = conv_block(f[1], f[2])
        self.enc3 = conv_block(f[2], f[3])
        self.enc4 = conv_block(f[3], f[4])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Attention gates
        self.ag3 = AttentionGate(f_g=f[4], f_x=f[3], f_int=f[3] // 2)
        self.up3 = nn.ConvTranspose2d(f[4], f[3], kernel_size=2, stride=2)
        self.dec3 = conv_block(f[4], f[3])

        self.ag2 = AttentionGate(f_g=f[3], f_x=f[2], f_int=f[2] // 2)
        self.up2 = nn.ConvTranspose2d(f[3], f[2], kernel_size=2, stride=2)
        self.dec2 = conv_block(f[3], f[2])

        self.ag1 = AttentionGate(f_g=f[2], f_x=f[1], f_int=f[1] // 2)
        self.up1 = nn.ConvTranspose2d(f[2], f[1], kernel_size=2, stride=2)
        self.dec1 = conv_block(f[2], f[1])

        self.ag0 = AttentionGate(f_g=f[1], f_x=f[0], f_int=f[0] // 2)
        self.up0 = nn.ConvTranspose2d(f[1], f[0], kernel_size=2, stride=2)
        self.dec0 = conv_block(f[1], f[0])

        self.head = nn.Conv2d(f[0], n_cls, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.enc0(x)
        p0 = self.pool(x0)

        x1 = self.enc1(p0)
        p1 = self.pool(x1)

        x2 = self.enc2(p1)
        p2 = self.pool(x2)

        x3 = self.enc3(p2)
        p3 = self.pool(x3)

        x4 = self.enc4(p3)

        # Decoder with Attention Skip Connections
        a3 = self.ag3(g=x4, x=x3)
        d3 = self.up3(x4)
        d3 = self.dec3(torch.cat([d3, a3], dim=1))

        a2 = self.ag2(g=d3, x=x2)
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, a2], dim=1))

        a1 = self.ag1(g=d2, x=x1)
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, a1], dim=1))

        a0 = self.ag0(g=d1, x=x0)
        d0 = self.up0(d1)
        d0 = self.dec0(torch.cat([d0, a0], dim=1))

        return self.head(d0)


def build_model(
    name: str = "unet",
    in_channels: int = 4,
    num_classes: int = 4,
    base_filters: int = 16,
) -> nn.Module:
    """
    Factory function to instantiate models based on configuration.
    """
    model_key = name.lower()
    if model_key == "unet":
        return UNet(in_ch=in_channels, n_cls=num_classes, base=base_filters)
    elif model_key in ["attention_unet", "attention-unet", "att_unet"]:
        return AttentionUNet(in_ch=in_channels, n_cls=num_classes, base=base_filters)
    elif model_key in ["unet_25d", "unet25d", "2.5d_unet"]:
        # 2.5D uses UNet with in_channels (e.g., 12 channels for 3 adjacent slices)
        return UNet(in_ch=in_channels, n_cls=num_classes, base=base_filters)
    else:
        raise ValueError(f"Unknown model architecture name: {name}")
