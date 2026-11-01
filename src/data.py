"""
Data pipeline for BraTS multi-modal MRI brain tumor segmentation.
Includes NIfTI preprocessing, patient-level splits (zero leakage),
and PyTorch Dataset implementations for 2D and 2.5D slice sampling.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

MODALITIES = ["flair", "t1", "t1ce", "t2"]
CROP = 192
NUM_CLASSES = 4


def preprocess_case(
    case_path: Union[str, Path],
    out_path: Union[str, Path],
    modalities: List[str] = MODALITIES,
    crop_size: int = CROP,
) -> bool:
    """
    Preprocess a single raw BraTS case directory containing NIfTI files.

    Performs:
    1. Per-modality z-score normalization on brain voxels (intensity > 0).
    2. Stacking 4 modalities into (4, H, W, D).
    3. Remapping ground truth label 4 -> 3 (0=BG, 1=NCR, 2=ED, 3=ET).
    4. Transposing to (4, D, H, W) and center-cropping H, W to crop_size.
    5. Saving img.npy (float16) and seg.npy (uint8).
    """
    case_dir = Path(case_path)
    out_dir = Path(out_path)
    cid = case_dir.name

    img_dst = out_dir / "img.npy"
    seg_dst = out_dir / "seg.npy"
    if img_dst.exists() and seg_dst.exists():
        return True

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load modalities and z-score normalize
    channels = []
    for mod in modalities:
        # Match case_id_modality.nii.gz or case_id_modality.nii
        candidates = list(case_dir.glob(f"*{mod}.nii*"))
        if not candidates:
            raise FileNotFoundError(f"Missing {mod} modality in {case_dir}")
        nii = nib.load(str(candidates[0]))
        vol = nii.get_fdata().astype(np.float32)

        # Brain mask: non-zero voxels
        brain_mask = vol > 0
        if np.any(brain_mask):
            mu = vol[brain_mask].mean()
            sd = vol[brain_mask].std() + 1e-8
            vol = np.where(brain_mask, (vol - mu) / sd, 0.0)
        channels.append(vol)

    img = np.stack(channels, axis=0)  # (4, H, W, D)

    # 2. Load segmentation and remap labels
    seg_candidates = list(case_dir.glob("*seg.nii*"))
    if not seg_candidates:
        raise FileNotFoundError(f"Missing segmentation mask in {case_dir}")
    seg = nib.load(str(seg_candidates[0])).get_fdata().astype(np.uint8)
    seg[seg == 4] = 3  # remap label 4 (enhancing tumor) -> 3

    # 3. Transpose to (C, D, H, W) and center crop H, W
    img = img.transpose(0, 3, 1, 2)  # (4, D, H, W)
    seg = seg.transpose(2, 0, 1)     # (D, H, W)

    h, w = img.shape[2:]
    top = max(0, (h - crop_size) // 2)
    left = max(0, (w - crop_size) // 2)

    img = img[:, :, top : top + crop_size, left : left + crop_size]
    seg = seg[:, top : top + crop_size, left : left + crop_size]

    # Save to disk
    np.save(img_dst, img.astype(np.float16))
    np.save(seg_dst, seg.astype(np.uint8))
    return True


def preprocess_dataset(
    raw_dir: Union[str, Path],
    out_dir: Union[str, Path],
    modalities: List[str] = MODALITIES,
    crop_size: int = CROP,
) -> List[str]:
    """
    Batch preprocess all cases in raw_dir into out_dir.
    """
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    cases = sorted([p for p in raw_path.iterdir() if p.is_dir()])

    processed_ids = []
    for case in tqdm(cases, desc="Preprocessing BraTS dataset"):
        cid = case.name
        dst = out_path / cid
        success = preprocess_case(case, dst, modalities=modalities, crop_size=crop_size)
        if success:
            processed_ids.append(cid)
    return processed_ids


def create_splits(
    processed_dir: Union[str, Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    splits_file: Optional[Union[str, Path]] = None,
) -> Dict[str, List[str]]:
    """
    Create a deterministic patient-level train/val/test split.
    Ensures that ALL slices from a given patient remain exclusively within
    a single split, preventing data leakage.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    processed_path = Path(processed_dir)
    case_ids = sorted([p.name for p in processed_path.iterdir() if (p / "img.npy").exists()])

    if len(case_ids) == 0:
        raise ValueError(f"No processed cases found in {processed_dir}")

    # Deterministic shuffle
    rng = random.Random(seed)
    shuffled = list(case_ids)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))

    # Adjust to handle small sample counts safely
    if n_train + n_val >= n_total and n_total >= 3:
        n_train = n_total - 2
        n_val = 1

    train_cases = sorted(shuffled[:n_train])
    val_cases = sorted(shuffled[n_train : n_train + n_val])
    test_cases = sorted(shuffled[n_train + n_val :])

    splits = {
        "train": train_cases,
        "val": val_cases,
        "test": test_cases,
    }

    if splits_file:
        sf_path = Path(splits_file)
        sf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sf_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
        logging.info(f"Saved patient-level splits to {sf_path}")

    return splits


def apply_nnunet_augmentations(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply nnU-Net style data augmentations:
    - Gaussian noise
    - Gaussian blur
    - Gamma transformation
    - Random rotation (90 degree increments)
    """
    # 1. Random 90 deg rotation
    k = random.choice([0, 1, 2, 3])
    if k > 0:
        x = np.rot90(x, k=k, axes=(-2, -1)).copy()
        y = np.rot90(y, k=k, axes=(-2, -1)).copy()

    # 2. Additive Gaussian noise (30% prob)
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.05, size=x.shape).astype(np.float32)
        x = x + noise

    # 3. Gaussian blur (20% prob)
    if random.random() < 0.2:
        sigma = random.uniform(0.5, 1.2)
        for c in range(x.shape[0]):
            x[c] = ndi.gaussian_filter(x[c], sigma=sigma)

    # 4. Gamma transformation (30% prob)
    if random.random() < 0.3:
        gamma = random.uniform(0.7, 1.5)
        # Shift to positive range for gamma, then shift back
        min_val = x.min()
        x_shifted = x - min_val + 1e-5
        x_gamma = np.power(x_shifted, gamma)
        x = x_gamma + min_val

    return x, y


class BraTSSliceDataset(Dataset):
    """
    PyTorch Dataset for BraTS 2D or 2.5D slice sampling.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        cases: List[str],
        mode: str = "2d",
        augment: bool = False,
        nnunet_aug: bool = False,
        empty_keep: float = 0.2,
        seed: int = 42,
    ) -> None:
        """
        Args:
            data_dir: Root directory of preprocessed cases.
            cases: List of case IDs for this split.
            mode: "2d" (4 channels) or "2.5d" (12 channels, 3 adjacent slices).
            augment: Enable standard flips and intensity scaling.
            nnunet_aug: Enable advanced nnU-Net style augmentations.
            empty_keep: Probability of keeping a non-tumor slice (0.0 to 1.0).
            seed: Seed for empty slice selection.
        """
        super().__init__()
        self.root = Path(data_dir)
        self.cases = cases
        self.mode = mode.lower()
        self.augment = augment
        self.nnunet_aug = nnunet_aug
        self.vols: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.index: List[Tuple[str, int]] = []

        rng = random.Random(seed)
        for cid in cases:
            img_path = self.root / cid / "img.npy"
            seg_path = self.root / cid / "seg.npy"
            if not img_path.exists() or not seg_path.exists():
                continue

            img = np.load(img_path, mmap_mode="r")  # (4, D, H, W)
            seg = np.load(seg_path, mmap_mode="r")  # (D, H, W)
            self.vols[cid] = (img, seg)

            num_slices = seg.shape[0]
            for z in range(num_slices):
                # Skip blank slices with no brain signal
                if not np.any(img[:, z]):
                    continue
                has_tumor = np.any(seg[z] > 0)
                if has_tumor or rng.random() < empty_keep:
                    self.index.append((cid, z))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cid, z = self.index[idx]
        img, seg = self.vols[cid]

        if self.mode == "2.5d":
            # Stack slice z-1, z, z+1
            d = img.shape[1]
            z_prev = max(0, z - 1)
            z_next = min(d - 1, z + 1)
            slice_prev = np.array(img[:, z_prev], dtype=np.float32)
            slice_curr = np.array(img[:, z], dtype=np.float32)
            slice_next = np.array(img[:, z_next], dtype=np.float32)
            x_np = np.concatenate([slice_prev, slice_curr, slice_next], axis=0)  # (12, H, W)
        else:
            x_np = np.array(img[:, z], dtype=np.float32)  # (4, H, W)

        y_np = np.array(seg[z], dtype=np.int64)  # (H, W)

        # Standard augmentations
        if self.augment:
            if random.random() < 0.5:
                x_np = np.flip(x_np, axis=-1).copy()
                y_np = np.flip(y_np, axis=-1).copy()
            if random.random() < 0.5:
                x_np = np.flip(x_np, axis=-2).copy()
                y_np = np.flip(y_np, axis=-2).copy()
            if random.random() < 0.5:
                scale = random.uniform(0.9, 1.1)
                shift = random.uniform(-0.1, 0.1)
                x_np = x_np * scale + shift

        # nnU-Net style augmentations
        if self.nnunet_aug:
            x_np, y_np = apply_nnunet_augmentations(x_np, y_np)

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        return x, y
