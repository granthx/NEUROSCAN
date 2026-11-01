"""
Unit tests for model architecture shapes and forward passes.
"""
import pytest
import torch

from src.model import AttentionGate, AttentionUNet, UNet, build_model


def test_unet_output_shape():
    batch_size = 2
    in_channels = 4
    num_classes = 4
    height, width = 128, 128

    model = UNet(in_ch=in_channels, n_cls=num_classes, base=8)
    x = torch.randn(batch_size, in_channels, height, width)
    out = model(x)

    assert out.shape == (batch_size, num_classes, height, width), f"Expected {(batch_size, num_classes, height, width)}, got {out.shape}"


def test_attention_gate():
    batch_size = 2
    f_g = 64
    f_x = 32
    f_int = 16
    h, w = 32, 32

    ag = AttentionGate(f_g=f_g, f_x=f_x, f_int=f_int)
    g = torch.randn(batch_size, f_g, h // 2, w // 2)
    x = torch.randn(batch_size, f_x, h, w)

    out = ag(g, x)
    assert out.shape == x.shape, f"Attention Gate output shape {out.shape} does not match skip shape {x.shape}"


def test_attention_unet_forward():
    batch_size = 1
    in_channels = 4
    num_classes = 4
    h, w = 64, 64

    model = AttentionUNet(in_ch=in_channels, n_cls=num_classes, base=8)
    x = torch.randn(batch_size, in_channels, h, w)
    out = model(x)

    assert out.shape == (batch_size, num_classes, h, w)


def test_unet_25d_forward():
    # 2.5D uses 12 channels (3 slices * 4 modalities)
    batch_size = 2
    in_channels = 12
    num_classes = 4
    h, w = 64, 64

    model = build_model(name="unet_25d", in_channels=in_channels, num_classes=num_classes, base_filters=8)
    x = torch.randn(batch_size, in_channels, h, w)
    out = model(x)

    assert out.shape == (batch_size, num_classes, h, w)
