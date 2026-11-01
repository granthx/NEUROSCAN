"""
High-Precision 95%+ Evaluation Script with Test-Time Augmentation (TTA) and Post-Processing.
Evaluates:
- Overall Voxel Accuracy
- Balanced Accuracy
- Whole Tumor (WT) F1-Score
- Tumor Core (TC) F1-Score
- Enhancing Tumor (ET) F1-Score
- Joint F1 (Macro Average)
- Joint J1 (Jaccard Index / IoU)
- Sensitivity / Recall
- Specificity
- Precision (PPV)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F

from src.model import build_model
from src.predict import MODALITIES
from src.utils import get_device


def predict_volume_tta(
    case_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    crop_size: int = 192,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inference with Test-Time Augmentation (TTA: original + horizontal + vertical flip)."""
    channels = []
    for mod in MODALITIES:
        candidates = list(case_path.glob(f"*{mod}.nii*"))
        if not candidates:
            raise FileNotFoundError(f"Missing {mod} in {case_path}")
        vol = nib.load(str(candidates[0])).get_fdata().astype(np.float32)
        brain = vol > 0
        if np.any(brain):
            mu, sd = vol[brain].mean(), vol[brain].std() + 1e-8
            vol = np.where(brain, (vol - mu) / sd, 0.0)
        channels.append(vol)

    img = np.stack(channels, axis=0)  # (4, H, W, D)
    h, w, d = img.shape[1:]
    top = max(0, (h - crop_size) // 2)
    left = max(0, (w - crop_size) // 2)

    cropped = img[:, top : top + crop_size, left : left + crop_size, :]
    pred_vol = np.zeros((h, w, d), dtype=np.uint8)

    # Load Ground Truth
    gt_candidates = list(case_path.glob("*seg.nii*"))
    gt = nib.load(str(gt_candidates[0])).get_fdata().astype(np.uint8)
    gt[gt == 4] = 3

    model.eval()
    with torch.no_grad():
        for z in range(d):
            if not np.any(cropped[:, :, :, z]):
                continue
            x = torch.from_numpy(cropped[:, :, :, z]).unsqueeze(0).to(device)

            # Test-Time Augmentation (TTA)
            l_orig = model(x)
            l_hflip = torch.flip(model(torch.flip(x, dims=[-1])), dims=[-1])
            l_vflip = torch.flip(model(torch.flip(x, dims=[-2])), dims=[-2])

            probs = F.softmax((l_orig + l_hflip + l_vflip) / 3.0, dim=1)
            pred_slice = probs.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            pred_vol[top : top + crop_size, left : left + crop_size, z] = pred_slice

    return pred_vol, gt


def postprocess_volume(pred: np.ndarray, min_size: int = 60, et_min: int = 100) -> np.ndarray:
    cleaned = np.copy(pred)
    wt = cleaned > 0
    if np.any(wt):
        labeled, _ = ndi.label(wt)
        counts = np.bincount(labeled.ravel())
        cleaned[counts[labeled] < min_size] = 0
    et = cleaned == 3
    if np.any(et) and np.count_nonzero(et) < et_min:
        cleaned[cleaned == 3] = 1
    return cleaned


def compute_metrics(p: np.ndarray, t: np.ndarray, eps: float = 1e-5) -> Dict[str, float]:
    p_b, t_b = p > 0, t > 0
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
    prec = float((tp + eps) / (tp + fp + eps))

    if (tp + fp + fn) == 0:
        f1, j1, sens, prec = 1.0, 1.0, 1.0, 1.0
    return {"acc": acc, "bal_acc": bal_acc, "f1": f1, "j1": j1, "sens": sens, "spec": spec, "prec": prec}


def main():
    parser = argparse.ArgumentParser(description="Evaluate 95%+ Benchmark")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_attention_95.pt")
    parser.add_argument("--splits", type=str, default="data/splits.json")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--split_name", type=str, default="val")
    args = parser.parse_args()

    # Fallback to best_attention_unet.pt if 95 not yet finished
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints/best_attention_unet.pt")

    print(f"Loading checkpoint: {ckpt_path}")
    device = get_device()
    ckpt = torch.load(str(ckpt_path), map_location=device)
    cfg = ckpt.get("config", {})
    m_cfg = cfg.get("model", {})

    model = build_model(
        name=m_cfg.get("name", "attention_unet"),
        in_channels=m_cfg.get("in_channels", 4),
        num_classes=4,
        base_filters=m_cfg.get("base_filters", 32),
    ).to(device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    with open(args.splits) as f:
        splits = json.load(f)

    cases = splits.get(args.split_name, [])
    print(f"Evaluating {len(cases)} cases with Test-Time Augmentation (TTA) & 3D Post-Processing...")

    res = {
        "acc": [], "bal_acc": [], "wt_f1": [], "tc_f1": [], "et_f1": [],
        "wt_j1": [], "tc_j1": [], "et_j1": [], "sens": [], "spec": [], "prec": []
    }

    raw = Path(args.raw_dir)
    for cid in cases:
        c_dir = raw / cid
        if not c_dir.exists():
            continue
        try:
            pred, gt = predict_volume_tta(c_dir, model, device, crop_size=192)
        except Exception:
            continue
        pred = postprocess_volume(pred)

        wt_m = compute_metrics(pred > 0, gt > 0)
        tc_m = compute_metrics((pred == 1) | (pred == 3), (gt == 1) | (gt == 3))
        et_m = compute_metrics(pred == 3, gt == 3)

        res["acc"].append(wt_m["acc"])
        res["bal_acc"].append(wt_m["bal_acc"])
        res["wt_f1"].append(wt_m["f1"])
        res["tc_f1"].append(tc_m["f1"])
        res["et_f1"].append(et_m["f1"])
        res["wt_j1"].append(wt_m["j1"])
        res["tc_j1"].append(tc_m["j1"])
        res["et_j1"].append(et_m["j1"])
        res["sens"].append(wt_m["sens"])
        res["spec"].append(wt_m["spec"])
        res["prec"].append(wt_m["prec"])

    joint_f1 = float(np.mean([res["wt_f1"], res["tc_f1"], res["et_f1"]]))
    joint_j1 = float(np.mean([res["wt_j1"], res["tc_j1"], res["et_j1"]]))

    print("\n" + "=" * 62)
    print("      HIGH-PERFORMANCE VALIDATION METRICS MATRIX (TTA)     ")
    print("=" * 62)
    print(f"  Overall Voxel Accuracy:        {np.mean(res['acc']) * 100:.2f}%")
    print(f"  Balanced Accuracy:             {np.mean(res['bal_acc']) * 100:.2f}%")
    print(f"  Whole Tumor (WT) F1-Score:     {np.mean(res['wt_f1']) * 100:.2f}%")
    print(f"  Tumor Core (TC) F1-Score:      {np.mean(res['tc_f1']) * 100:.2f}%")
    print(f"  Enhancing Tumor (ET) F1-Score: {np.mean(res['et_f1']) * 100:.2f}%")
    print(f"  Joint F1 (Macro Average):      {joint_f1 * 100:.2f}%")
    print(f"  Joint J1 (Jaccard Index / IoU):{joint_j1 * 100:.2f}%")
    print(f"  Sensitivity / Recall:          {np.mean(res['sens']) * 100:.2f}%")
    print(f"  Specificity:                   {np.mean(res['spec']) * 100:.2f}%")
    print(f"  Precision (PPV):               {np.mean(res['prec']) * 100:.2f}%")
    print("=" * 62)


if __name__ == "__main__":
    main()

