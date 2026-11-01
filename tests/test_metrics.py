"""
Unit tests for BraTS region Dice and HD95 metrics.
"""
import numpy as np
import pytest
import torch

from src.metrics import (
    compute_binary_dice,
    compute_binary_hd95,
    compute_region_dice,
    compute_region_hd95,
    get_brats_regions,
)


def test_brats_region_definitions():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1, 1] = 1  # NCR/NET
    mask[2, 2] = 2  # Edema
    mask[3, 3] = 3  # ET

    regions = get_brats_regions(mask)
    # WT should include 1, 2, 3 (3 voxels)
    assert regions["WT"].sum() == 3
    # TC should include 1, 3 (2 voxels)
    assert regions["TC"].sum() == 2
    # ET should include 3 (1 voxel)
    assert regions["ET"].sum() == 1


def test_binary_dice_metric():
    pred = np.zeros((20, 20), dtype=np.uint8)
    target = np.zeros((20, 20), dtype=np.uint8)

    # Empty masks -> 1.0 (true negative)
    assert compute_binary_dice(pred, target) == 1.0

    # Overlapping 4 pixels
    pred[5:7, 5:7] = 1
    target[5:7, 5:7] = 1
    assert abs(compute_binary_dice(pred, target) - 1.0) < 1e-4

    # Half overlap: pred 4 pixels, target 4 pixels, intersection 2
    target[5:7, 5:7] = 0
    target[5:7, 6:8] = 1
    # inter=2, card=4+4=8 -> 2*2 / 8 = 0.5
    assert abs(compute_binary_dice(pred, target) - 0.5) < 1e-2


def test_hd95_metric():
    pred = np.zeros((20, 20, 20), dtype=np.uint8)
    target = np.zeros((20, 20, 20), dtype=np.uint8)

    # Both empty -> 0.0 mm
    assert compute_binary_hd95(pred, target) == 0.0

    # Identical sphere -> 0.0 mm
    pred[8:12, 8:12, 8:12] = 1
    target[8:12, 8:12, 8:12] = 1
    hd = compute_binary_hd95(pred, target)
    assert hd == 0.0

    # One voxel offset in X
    target_shifted = np.zeros((20, 20, 20), dtype=np.uint8)
    target_shifted[9:13, 8:12, 8:12] = 1
    hd_shifted = compute_binary_hd95(pred, target_shifted)
    assert 0.5 <= hd_shifted <= 1.5
