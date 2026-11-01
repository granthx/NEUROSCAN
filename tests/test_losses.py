"""
Sanity tests for segmentation loss functions.
"""
import pytest
import torch
import torch.nn.functional as F

from src.losses import ComboLoss, DiceFocalLoss, FocalLoss, soft_dice_loss


def test_soft_dice_loss_bounds():
    b, c, h, w = 2, 4, 32, 32
    target = torch.randint(0, c, (b, h, w))
    # Perfect match: logits large at true target
    target_one_hot = F.one_hot(target, c).permute(0, 3, 1, 2).float()
    perfect_logits = target_one_hot * 50.0

    perfect_loss = soft_dice_loss(perfect_logits, target, num_classes=c)
    assert 0.0 <= perfect_loss.item() < 0.05, f"Perfect prediction loss should be near 0, got {perfect_loss.item()}"

    # Disjoint prediction
    inverted_target = (target + 1) % c
    inverted_logits = F.one_hot(inverted_target, c).permute(0, 3, 1, 2).float() * 50.0
    disjoint_loss = soft_dice_loss(inverted_logits, target, num_classes=c)
    assert disjoint_loss.item() > 0.8, f"Disjoint loss should be high, got {disjoint_loss.item()}"


def test_focal_loss_gradient():
    b, c, h, w = 2, 4, 16, 16
    logits = torch.randn(b, c, h, w, requires_grad=True)
    target = torch.randint(0, c, (b, h, w))

    criterion = FocalLoss(gamma=2.0)
    loss = criterion(logits, target)

    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_combo_losses():
    b, c, h, w = 2, 4, 16, 16
    logits = torch.randn(b, c, h, w, requires_grad=True)
    target = torch.randint(0, c, (b, h, w))

    combo = ComboLoss(ce_weight=1.0, dice_weight=1.0)
    loss_combo = combo(logits, target)
    assert loss_combo.item() > 0.0

    dice_focal = DiceFocalLoss(focal_weight=1.0, dice_weight=1.0, gamma=2.0)
    loss_df = dice_focal(logits, target)
    assert loss_df.item() > 0.0
