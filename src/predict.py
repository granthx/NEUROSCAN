"""
Inference CLI and visualization tool for BraTS brain tumor segmentation.
Loads a trained checkpoint, processes a multi-modal NIfTI case directory,
and exports predicted segmentation as .nii.gz plus high-resolution PNG slice overlays.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
import torch

from src.data import MODALITIES
from src.model import build_model
from src.utils import get_device

# Color mapping: 0=Transparent, 1=Red (Necrotic), 2=Green (Edema), 3=Yellow (Enhancing)
BRATS_CMAP = ListedColormap([(0, 0, 0, 0), "#e74c3c", "#2ecc71", "#f1c40f"])


def predict_volume(
    case_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    is_25d: bool = False,
    crop_size: int = 192,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Run inference on a 3D multi-modal NIfTI case.

    Returns:
        (pred_volume, raw_flair, ground_truth_seg, affine)
    """
    # 1. Load modalities
    channels = []
    affine = np.eye(4)
    raw_flair = None

    for mod in MODALITIES:
        candidates = list(case_path.glob(f"*{mod}.nii*"))
        if not candidates:
            raise FileNotFoundError(f"Missing {mod} modality in {case_path}")
        nii = nib.load(str(candidates[0]))
        if mod == "flair":
            affine = nii.affine
            raw_flair = nii.get_fdata().astype(np.float32)
        vol = nii.get_fdata().astype(np.float32)

        # Brain z-score
        brain_mask = vol > 0
        if np.any(brain_mask):
            mu = vol[brain_mask].mean()
            sd = vol[brain_mask].std() + 1e-8
            vol = np.where(brain_mask, (vol - mu) / sd, 0.0)
        channels.append(vol)

    img = np.stack(channels, axis=0)  # (4, H, W, D)

    # Load ground truth if exists
    seg_candidates = list(case_path.glob("*seg.nii*"))
    gt_seg = None
    if seg_candidates:
        gt_seg = nib.load(str(seg_candidates[0])).get_fdata().astype(np.uint8)
        gt_seg[gt_seg == 4] = 3

    # Shape dimensions
    c, h, w, d = img.shape
    top = max(0, (h - crop_size) // 2)
    left = max(0, (w - crop_size) // 2)

    # Crop H, W
    cropped_img = img[:, top : top + crop_size, left : left + crop_size, :]  # (4, CROP, CROP, D)
    pred_volume = np.zeros((h, w, d), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for z in range(d):
            if is_25d:
                z_prev = max(0, z - 1)
                z_next = min(d - 1, z + 1)
                sl_prev = cropped_img[:, :, :, z_prev]
                sl_curr = cropped_img[:, :, :, z]
                sl_next = cropped_img[:, :, :, z_next]
                x_slice = np.concatenate([sl_prev, sl_curr, sl_next], axis=0)  # (12, CROP, CROP)
            else:
                x_slice = cropped_img[:, :, :, z]  # (4, CROP, CROP)

            # Skip prediction if slice is empty brain tissue
            if not np.any(cropped_img[:, :, :, z]):
                continue

            tensor_slice = torch.from_numpy(x_slice).unsqueeze(0).to(device)  # (1, C, CROP, CROP)
            logits = model(tensor_slice)
            pred_slice = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

            # Place back into full volume shape
            pred_volume[top : top + crop_size, left : left + crop_size, z] = pred_slice

    return pred_volume, raw_flair, gt_seg, affine


def save_prediction_overlay(
    flair: np.ndarray,
    pred: np.ndarray,
    gt: Optional[np.ndarray],
    out_png: Path,
    slice_idx: Optional[int] = None,
) -> None:
    """
    Render 3-panel visualization: FLAIR | Ground Truth | Prediction.
    """
    d = flair.shape[2]
    if slice_idx is None:
        # Choose slice with largest tumor area in ground truth, or middle slice
        if gt is not None and np.any(gt > 0):
            tumor_counts = [(gt[:, :, z] > 0).sum() for z in range(d)]
            slice_idx = int(np.argmax(tumor_counts))
        elif np.any(pred > 0):
            tumor_counts = [(pred[:, :, z] > 0).sum() for z in range(d)]
            slice_idx = int(np.argmax(tumor_counts))
        else:
            slice_idx = d // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#1e1e1e")

    # Panel 1: FLAIR
    axes[0].imshow(flair[:, :, slice_idx].T, cmap="gray", origin="lower")
    axes[0].set_title(f"FLAIR (Axial Slice {slice_idx})", color="white", fontsize=12)

    # Panel 2: Ground Truth Overlay
    axes[1].imshow(flair[:, :, slice_idx].T, cmap="gray", origin="lower")
    if gt is not None:
        axes[1].imshow(gt[:, :, slice_idx].T, cmap=BRATS_CMAP, vmin=0, vmax=3, origin="lower", alpha=0.6)
        axes[1].set_title("Ground Truth Segmentation", color="white", fontsize=12)
    else:
        axes[1].set_title("Ground Truth (N/A)", color="white", fontsize=12)

    # Panel 3: Prediction Overlay
    axes[2].imshow(flair[:, :, slice_idx].T, cmap="gray", origin="lower")
    axes[2].imshow(pred[:, :, slice_idx].T, cmap=BRATS_CMAP, vmin=0, vmax=3, origin="lower", alpha=0.6)
    axes[2].set_title("Model Prediction", color="white", fontsize=12)

    for ax in axes:
        ax.axis("off")

    fig.text(
        0.5,
        0.02,
        "Legend:  Red = Necrotic Core (1)  |  Green = Peritumoral Edema (2)  |  Yellow = Enhancing Tumor (3)",
        ha="center",
        color="lightgray",
        fontsize=11,
    )
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Saved prediction overlay to: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BraTS NIfTI Case Inference")
    parser.add_argument("--case_dir", type=str, required=True, help="Path to raw case folder containing NIfTI files")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_baseline.pt", help="Path to model checkpoint")
    parser.add_argument("--out_dir", type=str, default="outputs/predictions", help="Directory to save output files")
    parser.add_argument("--slice", type=int, default=None, help="Specific slice index for PNG overlay")
    args = parser.parse_args()

    case_path = Path(args.case_dir)
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = get_device()
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Extract model configuration if saved, else default
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

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    print(f"Running inference for {case_path.name} using {model_name}...")
    pred_vol, flair, gt, affine = predict_volume(case_path, model, device, is_25d=is_25d)

    # Save predicted NIfTI
    out_dir.mkdir(parents=True, exist_ok=True)
    nii_out = out_dir / f"{case_path.name}_pred_seg.nii.gz"
    nib.save(nib.Nifti1Image(pred_vol, affine), str(nii_out))
    print(f"Saved predicted segmentation NIfTI to: {nii_out}")

    # Save PNG overlay
    png_out = out_dir / f"{case_path.name}_overlay.png"
    save_prediction_overlay(flair, pred_vol, gt, png_out, slice_idx=args.slice)


if __name__ == "__main__":
    main()
