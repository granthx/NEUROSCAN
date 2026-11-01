"""
Evaluate comprehensive validation metrics:
- Overall Voxel Accuracy (>99%)
- Balanced Accuracy
- F1-Score / Dice (Whole Tumor, Tumor Core, Enhancing Tumor)
- Joint F1 (Macro Average)
- J1 (Jaccard Index / IoU)
- Sensitivity / Recall & Specificity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.ndimage as ndi
import torch

from src.model import build_model
from src.predict import predict_volume
from src.utils import get_device


def postprocess_volume(
    pred: np.ndarray,
    min_component_size: int = 80,
    et_min_threshold: int = 150,
) -> np.ndarray:
    """Removes small isolated false-positive noise and reassigns tiny ET."""
    cleaned = np.copy(pred)
    wt = cleaned > 0
    if np.any(wt):
        labeled, _ = ndi.label(wt)
        counts = np.bincount(labeled.ravel())
        cleaned[counts[labeled] < min_component_size] = 0

    et = cleaned == 3
    if np.any(et) and np.count_nonzero(et) < et_min_threshold:
        cleaned[cleaned == 3] = 1
    return cleaned


def compute_metrics(p: np.ndarray, t: np.ndarray, eps: float = 1e-5) -> Dict[str, float]:
    p_b = p > 0
    t_b = t > 0
    tp = np.logical_and(p_b, t_b).sum()
    fp = np.logical_and(p_b, ~t_b).sum()
    fn = np.logical_and(~p_b, t_b).sum()
    tn = np.logical_and(~p_b, ~t_b).sum()
    total = max(1, (tp + tn + fp + fn))

    acc = float((tp + tn) / total)
    sens = float((tp + eps) / (tp + fn + eps))
    spec = float((tn + eps) / (tn + fp + eps))
    bal_acc = float((sens + spec) / 2.0)
    f1 = float((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps))
    j1 = float((tp + eps) / (tp + fp + fn + eps))
    precision = float((tp + eps) / (tp + fp + eps))

    if (tp + fp + fn) == 0:
        f1 = 1.0
        j1 = 1.0
        sens = 1.0
        precision = 1.0

    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "f1": f1,
        "j1": j1,
        "sens": sens,
        "spec": spec,
        "prec": precision,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate 95%+ Comprehensive Metrics")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_attention_unet.pt")
    parser.add_argument("--splits", type=str, default="data/splits.json")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--split_name", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cases (0 = all)")
    args = parser.parse_args()

    splits_path = Path(args.splits)
    if not splits_path.is_file():
        raise FileNotFoundError(f"Missing {splits_path}")

    with open(splits_path) as f:
        splits = json.load(f)

    case_ids = splits.get(args.split_name, [])
    if args.limit > 0:
        case_ids = case_ids[: args.limit]

    print(f"Loading checkpoint: {args.ckpt}")
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("config", {})
    m_cfg = cfg.get("model", {})

    model = build_model(
        name=m_cfg.get("name", "attention_unet"),
        in_channels=m_cfg.get("in_channels", 4),
        num_classes=4,
        base_filters=m_cfg.get("base_filters", 16),
    ).to(device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    print(f"Evaluating {len(case_ids)} patients from '{args.split_name}' split with post-processing...")

    all_res = {
        "acc": [],
        "bal_acc": [],
        "wt_f1": [],
        "tc_f1": [],
        "et_f1": [],
        "wt_j1": [],
        "tc_j1": [],
        "et_j1": [],
        "sens": [],
        "spec": [],
        "prec": [],
    }

    raw_path = Path(args.raw_dir)
    for cid in case_ids:
        case_dir = raw_path / cid
        if not case_dir.exists():
            continue

        pred_vol, _, gt, _ = predict_volume(case_dir, model, device, crop_size=192)
        if gt is None:
            continue

        # Apply postprocessing
        pred_vol = postprocess_volume(pred_vol)

        wt_m = compute_metrics(pred_vol > 0, gt > 0)
        tc_m = compute_metrics((pred_vol == 1) | (pred_vol == 3), (gt == 1) | (gt == 3))
        et_m = compute_metrics(pred_vol == 3, gt == 3)

        all_res["acc"].append(wt_m["acc"])
        all_res["bal_acc"].append(wt_m["bal_acc"])
        all_res["wt_f1"].append(wt_m["f1"])
        all_res["tc_f1"].append(tc_m["f1"])
        all_res["et_f1"].append(et_m["f1"])
        all_res["wt_j1"].append(wt_m["j1"])
        all_res["tc_j1"].append(tc_m["j1"])
        all_res["et_j1"].append(et_m["j1"])
        all_res["sens"].append(wt_m["sens"])
        all_res["spec"].append(wt_m["spec"])
        all_res["prec"].append(wt_m["prec"])

    joint_f1 = float(np.mean([all_res["wt_f1"], all_res["tc_f1"], all_res["et_f1"]]))
    joint_j1 = float(np.mean([all_res["wt_j1"], all_res["tc_j1"], all_res["et_j1"]]))

    print("\n" + "=" * 62)
    print("        COMPREHENSIVE VALIDATION METRICS MATRIX        ")
    print("=" * 62)
    print(f"  Overall Voxel Accuracy:        {np.mean(all_res['acc']) * 100:.2f}%")
    print(f"  Balanced Accuracy:             {np.mean(all_res['bal_acc']) * 100:.2f}%")
    print(f"  Whole Tumor (WT) F1-Score:     {np.mean(all_res['wt_f1']) * 100:.2f}%")
    print(f"  Tumor Core (TC) F1-Score:      {np.mean(all_res['tc_f1']) * 100:.2f}%")
    print(f"  Enhancing Tumor (ET) F1-Score: {np.mean(all_res['et_f1']) * 100:.2f}%")
    print(f"  Joint F1 (Macro Average):      {joint_f1 * 100:.2f}%")
    print(f"  Joint J1 (Jaccard Index / IoU):{joint_j1 * 100:.2f}%")
    print(f"  Sensitivity / Recall:          {np.mean(all_res['sens']) * 100:.2f}%")
    print(f"  Specificity:                   {np.mean(all_res['spec']) * 100:.2f}%")
    print(f"  Precision (PPV):               {np.mean(all_res['prec']) * 100:.2f}%")
    print("=" * 62)


if __name__ == "__main__":
    main()
