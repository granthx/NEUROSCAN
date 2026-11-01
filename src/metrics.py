"""
Evaluation metrics for BraTS: Dice Similarity Coefficient and 95th Percentile Hausdorff Distance (HD95).
Evaluates regions: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).
"""
from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
import scipy.ndimage as ndi
import torch


def get_brats_regions(
    mask: Union[np.ndarray, torch.Tensor],
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """
    Extract boolean binary masks for the three standard BraTS evaluation regions:
    - WT (Whole Tumor): labels 1, 2, 3
    - TC (Tumor Core): labels 1, 3 (NCR/NET and ET)
    - ET (Enhancing Tumor): label 3
    """
    if isinstance(mask, torch.Tensor):
        return {
            "WT": mask > 0,
            "TC": (mask == 1) | (mask == 3),
            "ET": mask == 3,
        }
    return {
        "WT": mask > 0,
        "TC": (mask == 1) | (mask == 3),
        "ET": mask == 3,
    }


def compute_binary_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-5,
) -> Dict[str, float]:
    """
    Compute full suite of binary classification and segmentation metrics:
    - F1-Score / Dice
    - Jaccard Index (J1 / IoU)
    - Accuracy (overall pixel/voxel accuracy)
    - Balanced Accuracy
    - Sensitivity / Recall
    - Specificity
    - Precision
    """
    if isinstance(pred, torch.Tensor):
        p = (pred > 0).bool().cpu().numpy()
        t = (target > 0).bool().cpu().numpy()
    else:
        p = np.asarray(pred > 0, dtype=bool)
        t = np.asarray(target > 0, dtype=bool)

    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    tn = np.logical_and(~p, ~t).sum()
    total = tp + fp + fn + tn

    # Accuracy
    accuracy = float((tp + tn) / max(1, total))

    # Sensitivity / Recall & Specificity
    sensitivity = float((tp + eps) / (tp + fn + eps))
    specificity = float((tn + eps) / (tn + fp + eps))
    balanced_acc = float((sensitivity + specificity) / 2.0)

    # Precision
    precision = float((tp + eps) / (tp + fp + eps))

    # F1-Score (Dice)
    f1 = float((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps))

    # Jaccard Index (J1 / IoU)
    j1 = float((tp + eps) / (tp + fp + fn + eps))

    # Handle true negative case when both are empty
    if (tp + fp + fn) == 0:
        f1 = 1.0
        j1 = 1.0
        precision = 1.0
        sensitivity = 1.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "f1": f1,
        "dice": f1,
        "j1": j1,
        "jaccard": j1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
    }


def compute_comprehensive_case_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-5,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics across WT, TC, ET, plus joint / macro averages.
    """
    pred_regions = get_brats_regions(pred)
    target_regions = get_brats_regions(target)

    results = {}
    f1_list = []
    j1_list = []
    acc_list = []
    bal_acc_list = []

    for region in ["WT", "TC", "ET"]:
        m = compute_binary_metrics(pred_regions[region], target_regions[region], eps=eps)
        for metric_name, val in m.items():
            results[f"{region}_{metric_name}"] = val
        f1_list.append(m["f1"])
        j1_list.append(m["j1"])
        acc_list.append(m["accuracy"])
        bal_acc_list.append(m["balanced_accuracy"])

    # Joint / Macro Metrics across all regions
    results["joint_f1"] = float(np.mean(f1_list))
    results["joint_j1"] = float(np.mean(j1_list))
    results["macro_accuracy"] = float(np.mean(acc_list))
    results["macro_balanced_accuracy"] = float(np.mean(bal_acc_list))
    return results


def apply_postprocessing(
    pred_volume: np.ndarray,
    min_component_size: int = 80,
    et_min_threshold: int = 150,
) -> np.ndarray:
    """
    Standard BraTS post-processing:
    1. Removes isolated connected false-positive clusters below min_component_size.
    2. Re-assigns isolated enhancing tumor (ET) below et_min_threshold to necrotic core (label 1).
    """
    cleaned = np.copy(pred_volume)

    # 1. Clean Whole Tumor (WT) tiny noise components
    wt_mask = cleaned > 0
    if np.any(wt_mask):
        labeled_wt, num_features = ndi.label(wt_mask)
        component_sizes = np.bincount(labeled_wt.ravel())
        too_small = component_sizes < min_component_size
        too_small_mask = too_small[labeled_wt]
        cleaned[too_small_mask] = 0

    # 2. Re-assign small ET false positives (standard BraTS rule)
    et_mask = cleaned == 3
    if np.any(et_mask):
        et_count = np.count_nonzero(et_mask)
        if et_count < et_min_threshold:
            cleaned[cleaned == 3] = 1  # Reassign to Necrotic Core

    return cleaned


def compute_binary_dice(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-5,
) -> float:
    """
    Compute binary Dice similarity score between prediction and target masks.
    """
    if isinstance(pred, torch.Tensor):
        pred_b = (pred > 0).bool()
        target_b = (target > 0).bool()
        inter = (pred_b & target_b).sum().item()
        total = pred_b.sum().item() + target_b.sum().item()
    else:
        pred_b = pred > 0
        target_b = target > 0
        inter = np.logical_and(pred_b, target_b).sum()
        total = pred_b.sum() + target_b.sum()

    if total == 0:
        return 1.0  # True negative agreement
    return float((2.0 * inter + eps) / (total + eps))


def compute_region_dice(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-5,
) -> Dict[str, float]:
    """
    Compute Dice scores for BraTS WT, TC, and ET regions.
    """
    pred_regions = get_brats_regions(pred)
    target_regions = get_brats_regions(target)

    results = {}
    for region_name in ["WT", "TC", "ET"]:
        results[region_name] = compute_binary_dice(
            pred_regions[region_name],
            target_regions[region_name],
            eps=eps,
        )
    results["mean"] = float(np.mean([results["WT"], results["TC"], results["ET"]]))
    return results


def compute_binary_hd95(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    empty_penalty: float = 373.13,
) -> float:
    """
    Compute the 95th percentile Hausdorff Distance (HD95) between binary masks.

    Args:
        pred: Predicted binary mask.
        target: Ground truth binary mask.
        voxel_spacing: Physical voxel dimensions in mm (default: (1.0, 1.0, 1.0)).
        empty_penalty: Penalty distance in mm if one mask is empty while the other is not.

    Returns:
        HD95 value in mm. Returns 0.0 if both masks are empty.
    """
    pred = np.asarray(pred > 0, dtype=bool)
    target = np.asarray(target > 0, dtype=bool)

    pred_count = np.count_nonzero(pred)
    target_count = np.count_nonzero(target)

    if pred_count == 0 and target_count == 0:
        return 0.0
    if pred_count == 0 or target_count == 0:
        return float(empty_penalty)

    # Extract boundaries
    footprint = ndi.generate_binary_structure(pred.ndim, 1)
    pred_border = np.logical_xor(pred, ndi.binary_erosion(pred, structure=footprint))
    target_border = np.logical_xor(target, ndi.binary_erosion(target, structure=footprint))

    # If erosion consumed the entire tiny structure, fallback to the mask itself
    if not np.any(pred_border):
        pred_border = pred
    if not np.any(target_border):
        target_border = target

    # Distance transforms from boundaries
    dt_target = ndi.distance_transform_edt(~target_border, sampling=voxel_spacing)
    dt_pred = ndi.distance_transform_edt(~pred_border, sampling=voxel_spacing)

    distances_pred_to_target = dt_target[pred_border]
    distances_target_to_pred = dt_pred[target_border]

    all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
    if len(all_distances) == 0:
        return 0.0

    return float(np.percentile(all_distances, 95))


def compute_region_hd95(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    empty_penalty: float = 373.13,
) -> Dict[str, float]:
    """
    Compute HD95 for BraTS WT, TC, and ET regions.
    """
    pred_regions = get_brats_regions(pred)
    target_regions = get_brats_regions(target)

    results = {}
    for region_name in ["WT", "TC", "ET"]:
        results[region_name] = compute_binary_hd95(
            pred_regions[region_name],
            target_regions[region_name],
            voxel_spacing=voxel_spacing,
            empty_penalty=empty_penalty,
        )
    results["mean"] = float(np.mean([results["WT"], results["TC"], results["ET"]]))
    return results


def evaluate_case_volume(
    pred_volume: np.ndarray,
    target_volume: np.ndarray,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """
    Evaluate full 3D case volume for Dice and HD95 across WT, TC, and ET.
    """
    dice_res = compute_region_dice(pred_volume, target_volume)
    hd_res = compute_region_hd95(pred_volume, target_volume, voxel_spacing=voxel_spacing)

    return {
        "dice_WT": dice_res["WT"],
        "dice_TC": dice_res["TC"],
        "dice_ET": dice_res["ET"],
        "dice_mean": dice_res["mean"],
        "hd95_WT": hd_res["WT"],
        "hd95_TC": hd_res["TC"],
        "hd95_ET": hd_res["ET"],
        "hd95_mean": hd_res["mean"],
    }
