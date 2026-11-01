"""
Evaluation pipeline for held-out test set.
Computes volume-level Dice and HD95 for WT, TC, and ET across all test patients.
Exports per-patient CSV results and generates a failure-case gallery.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.metrics import evaluate_case_volume
from src.model import build_model
from src.predict import BRATS_CMAP, predict_volume
from src.utils import get_device


def render_failure_case(
    flair: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    case_id: str,
    metrics: Dict[str, float],
    out_png: Path,
) -> None:
    """
    Render visual failure analysis displaying FLAIR, Ground Truth, Model Prediction,
    and False Positive / False Negative discrepancy maps.
    """
    d = flair.shape[2]
    # Find slice with maximum tumor or largest discrepancy
    discrepancy = np.abs((pred > 0).astype(int) - (gt > 0).astype(int))
    disc_counts = [discrepancy[:, :, z].sum() for z in range(d)]
    slice_idx = int(np.argmax(disc_counts)) if np.max(disc_counts) > 0 else d // 2

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#181818")

    # 1. FLAIR
    axes[0].imshow(flair[:, :, slice_idx].T, cmap="gray", origin="lower")
    axes[0].set_title(f"FLAIR (Slice {slice_idx})", color="white", fontsize=11)

    # 2. Ground Truth
    axes[1].imshow(flair[:, :, slice_idx].T, cmap="gray", origin="lower")
    axes[1].imshow(gt[:, :, slice_idx].T, cmap=BRATS_CMAP, vmin=0, vmax=3, origin="lower", alpha=0.6)
    axes[1].set_title("Ground Truth", color="white", fontsize=11)

    # 3. Prediction
    axes[2].imshow(flair[:, :, slice_idx].T, cmap="gray", origin="lower")
    axes[2].imshow(pred[:, :, slice_idx].T, cmap=BRATS_CMAP, vmin=0, vmax=3, origin="lower", alpha=0.6)
    axes[2].set_title(
        f"Prediction\n(Dice: WT={metrics['dice_WT']:.2f}, TC={metrics['dice_TC']:.2f}, ET={metrics['dice_ET']:.2f})",
        color="#ff7675",
        fontsize=10,
    )

    # 4. Error Map (Green=True Pos, Red=False Pos, Blue=False Neg)
    gt_bin = gt[:, :, slice_idx].T > 0
    pred_bin = pred[:, :, slice_idx].T > 0
    tp = gt_bin & pred_bin
    fp = pred_bin & (~gt_bin)
    fn = (~pred_bin) & gt_bin

    err_map = np.zeros((*gt_bin.shape, 3), dtype=np.float32)
    err_map[tp] = [0.2, 0.8, 0.2]  # True Positives (Green)
    err_map[fp] = [0.9, 0.2, 0.2]  # False Positives (Red)
    err_map[fn] = [0.2, 0.4, 0.9]  # False Negatives (Blue)

    axes[3].imshow(flair[:, :, slice_idx].T, cmap="gray", origin="lower")
    axes[3].imshow(err_map, origin="lower", alpha=0.7)
    axes[3].set_title("Discrepancy (Green=TP, Red=FP, Blue=FN)", color="white", fontsize=10)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"Failure Case Inspection: {case_id} (Mean Dice: {metrics['dice_mean']:.3f})", color="white", fontsize=13)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()


def evaluate_test_set(
    ckpt_path: Path,
    splits_file: Path,
    raw_dir: Path,
    out_dir: Path,
    num_failure_cases: int = 3,
) -> Dict[str, Any]:
    """
    Run full test set evaluation, save CSV, and render failure gallery.
    """
    if not splits_file.is_file():
        raise FileNotFoundError(f"Splits file not found: {splits_file}")

    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_cases = splits.get("test", [])
    if not test_cases:
        # If test cases empty, fallback to val cases for validation evaluation
        test_cases = splits.get("val", [])
        print("Note: 'test' split was empty; evaluating on 'val' cases.")

    device = get_device()
    checkpoint = torch.load(ckpt_path, map_location=device)

    cfg = checkpoint.get("config", {})
    m_cfg = cfg.get("model", {})
    model_name = m_cfg.get("name", "unet")
    in_channels = m_cfg.get("in_channels", 4)
    base_filters = m_cfg.get("base_filters", 16)
    is_25d = model_name == "unet_25d"

    model = build_model(
        name=model_name,
        in_channels=in_channels,
        num_classes=4,
        base_filters=base_filters,
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "test_evaluation_results.csv"

    records: List[Dict[str, Any]] = []
    case_predictions: List[Tuple[str, np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, float]]] = []

    print(f"Evaluating {len(test_cases)} test cases with model: {model_name}...")

    for cid in tqdm(test_cases, desc="Evaluating Test Cases"):
        case_folder = raw_dir / cid
        if not case_folder.exists():
            print(f"Warning: case folder {case_folder} does not exist, skipping.")
            continue

        pred_vol, flair, gt_seg, _ = predict_volume(case_folder, model, device, is_25d=is_25d)
        if gt_seg is None:
            print(f"Skipping {cid}: no ground truth segmentation mask found.")
            continue

        metrics = evaluate_case_volume(pred_vol, gt_seg)
        rec = {"case_id": cid, **metrics}
        records.append(rec)
        case_predictions.append((cid, flair, pred_vol, gt_seg, metrics))

    if not records:
        raise RuntimeError("No test cases could be evaluated.")

    # Save CSV
    fieldnames = [
        "case_id",
        "dice_WT",
        "dice_TC",
        "dice_ET",
        "dice_mean",
        "hd95_WT",
        "hd95_TC",
        "hd95_ET",
        "hd95_mean",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()})

    # Summary statistics
    avg_metrics = {
        col: float(np.mean([r[col] for r in records]))
        for col in fieldnames
        if col != "case_id"
    }

    print("\n" + "=" * 60)
    print("Test Set Evaluation Summary")
    print("=" * 60)
    print(f"Dice WT:    {avg_metrics['dice_WT']:.4f}")
    print(f"Dice TC:    {avg_metrics['dice_TC']:.4f}")
    print(f"Dice ET:    {avg_metrics['dice_ET']:.4f}")
    print(f"Dice Mean:  {avg_metrics['dice_mean']:.4f}")
    print(f"HD95 WT:    {avg_metrics['hd95_WT']:.2f} mm")
    print(f"HD95 TC:    {avg_metrics['hd95_TC']:.2f} mm")
    print(f"HD95 ET:    {avg_metrics['hd95_ET']:.2f} mm")
    print(f"HD95 Mean:  {avg_metrics['hd95_mean']:.2f} mm")
    print(f"Results CSV saved to: {csv_path}")

    # Generate failure gallery (lowest dice_mean)
    gallery_dir = out_dir / "failure_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    case_predictions.sort(key=lambda item: item[4]["dice_mean"])

    num_failures = min(num_failure_cases, len(case_predictions))
    for rank in range(num_failures):
        cid, flair, pred_vol, gt_seg, m = case_predictions[rank]
        fail_img = gallery_dir / f"rank{rank+1}_{cid}_failure.png"
        render_failure_case(flair, pred_vol, gt_seg, cid, m, fail_img)

    print(f"Rendered {num_failures} failure cases in gallery: {gallery_dir}")
    print("=" * 60)

    return {"summary": avg_metrics, "csv_path": str(csv_path), "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="BraTS Test Set Evaluation")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_baseline.pt", help="Path to checkpoint")
    parser.add_argument("--splits", type=str, default="data/splits.json", help="Path to splits JSON")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw NIfTI cases")
    parser.add_argument("--out_dir", type=str, default="outputs/evaluation", help="Evaluation output folder")
    parser.add_argument("--num_failures", type=int, default=3, help="Number of failure cases to save")
    args = parser.parse_args()

    evaluate_test_set(
        ckpt_path=Path(args.ckpt),
        splits_file=Path(args.splits),
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        num_failure_cases=args.num_failures,
    )


if __name__ == "__main__":
    main()
