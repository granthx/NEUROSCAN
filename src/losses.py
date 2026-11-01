"""
Loss functions for multi-modal brain tumor segmentation.
Includes Soft Dice Loss, Cross Entropy, Multi-class Focal Loss, and Combinations.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 4,
    eps: float = 1e-5,
    ignore_background: bool = True,
) -> torch.Tensor:
    """
    Compute multi-class Soft Dice loss over batch spatial dimensions.

    Args:
        logits: Raw predictions of shape (B, C, H, W).
        target: Ground truth indices of shape (B, H, W).
        num_classes: Number of segmentation classes (default: 4).
        eps: Small epsilon to avoid division by zero.
        ignore_background: If True, exclude class 0 (background) from the loss.

    Returns:
        Scalar Dice loss tensor.
    """
    probs = F.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)

    intersection = (probs * target_one_hot).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + target_one_hot.sum(dim=dims)
    dice_per_class = (2.0 * intersection + eps) / (cardinality + eps)

    if ignore_background:
        # Average over tumor classes (1, 2, 3)
        return 1.0 - dice_per_class[1:].mean()
    return 1.0 - dice_per_class.mean()


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss to address severe class imbalance between
    background and small tumor subregions.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-class Focal Loss.
        """
        # Cross entropy computes -log(pt)
        ce_loss = F.cross_entropy(logits, target, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class ComboLoss(nn.Module):
    """
    Combined Cross-Entropy and Soft Dice loss.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        num_classes: int = 4,
        ignore_background: bool = True,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target)
        dice = soft_dice_loss(
            logits,
            target,
            num_classes=self.num_classes,
            ignore_background=self.ignore_background,
        )
        return self.ce_weight * ce + self.dice_weight * dice


class DiceFocalLoss(nn.Module):
    """
    Combined Multi-class Focal Loss and Soft Dice loss.
    """

    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        gamma: float = 2.0,
        num_classes: int = 4,
        ignore_background: bool = True,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(gamma=gamma)
        self.num_classes = num_classes
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        focal = self.focal(logits, target)
        dice = soft_dice_loss(
            logits,
            target,
            num_classes=self.num_classes,
            ignore_background=self.ignore_background,
        )
        return self.focal_weight * focal + self.dice_weight * dice
