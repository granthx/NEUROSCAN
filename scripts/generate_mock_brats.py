"""
Generate synthetic BraTS-compliant multi-modal NIfTI cases for testing, CI, and CPU benchmarking.
Creates realistic brain phantoms with multi-modal contrasts and 3D tumor lesions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def create_mock_case(
    case_id: str,
    out_dir: Path,
    shape: tuple = (240, 240, 60),
    seed: int = 42,
) -> None:
    """
    Generate synthetic FLAIR, T1, T1ce, T2, and segmentation NIfTI volumes.
    """
    rng = np.random.default_rng(seed)
    h, w, d = shape

    # Create 3D coordinate grid
    y, x, z = np.meshgrid(np.arange(h), np.arange(w), np.arange(d), indexing="ij")
    cy, cx, cz = h / 2.0, w / 2.0, d / 2.0

    # 1. Ellipsoidal brain phantom
    brain_mask = (
        ((x - cx) / (w * 0.38)) ** 2
        + ((y - cy) / (h * 0.42)) ** 2
        + ((z - cz) / (d * 0.40)) ** 2
    ) <= 1.0

    # Internal brain structures (GM/WM variations)
    tissue_base = np.zeros(shape, dtype=np.float32)
    variation = 100.0 + 30.0 * np.sin(x / 15.0) + 20.0 * np.cos(y / 15.0)
    tissue_base[brain_mask] = variation[brain_mask]

    # 2. Simulated multi-compartment tumor
    # Tumor center offset from brain center
    offset_y = int(rng.uniform(-h * 0.12, h * 0.12))
    offset_x = int(rng.uniform(-w * 0.12, w * 0.12))
    offset_z = int(rng.uniform(-d * 0.10, d * 0.10))
    tcy, tcx, tcz = cy + offset_y, cx + offset_x, cz + offset_z

    # Distances from tumor center
    tumor_r_sq = (
        ((x - tcx) / 24.0) ** 2
        + ((y - tcy) / 20.0) ** 2
        + ((z - tcz) / 12.0) ** 2
    )

    edema_mask = (tumor_r_sq <= 1.0) & brain_mask
    core_mask = (tumor_r_sq <= 0.45) & brain_mask
    enhancing_mask = (tumor_r_sq > 0.15) & (tumor_r_sq <= 0.45) & brain_mask
    necrotic_mask = (tumor_r_sq <= 0.15) & brain_mask

    # Ground truth segmentation (BraTS original conventions: 1=NCR, 2=ED, 4=ET)
    seg = np.zeros(shape, dtype=np.uint8)
    seg[edema_mask] = 2
    seg[necrotic_mask] = 1
    seg[enhancing_mask] = 4

    # 3. Multimodal contrasts
    # Noise field
    noise = rng.normal(0, 3.0, size=shape).astype(np.float32)

    # FLAIR: Edema is hyperintense, brain tissue moderate, background zero
    flair = np.zeros(shape, dtype=np.float32)
    flair[brain_mask] = tissue_base[brain_mask] * 1.2
    flair[edema_mask] += 120.0
    flair[enhancing_mask] += 60.0
    flair[necrotic_mask] += 30.0
    flair[brain_mask] += noise[brain_mask]
    flair = np.clip(flair, 0, None)

    # T1: Brain tissue standard, edema hypointense, necrotic hypointense
    t1 = np.zeros(shape, dtype=np.float32)
    t1[brain_mask] = tissue_base[brain_mask] * 1.5
    t1[edema_mask] -= 40.0
    t1[necrotic_mask] -= 50.0
    t1[brain_mask] += noise[brain_mask]
    t1 = np.clip(t1, 0, None)

    # T1ce: Enhancing rim is bright hyperintense, necrotic core dark
    t1ce = t1.copy()
    t1ce[enhancing_mask] += 150.0
    t1ce = np.clip(t1ce, 0, None)

    # T2: Edema and necrotic core bright hyperintense
    t2 = np.zeros(shape, dtype=np.float32)
    t2[brain_mask] = tissue_base[brain_mask] * 1.3
    t2[edema_mask] += 100.0
    t2[necrotic_mask] += 90.0
    t2[brain_mask] += noise[brain_mask]
    t2 = np.clip(t2, 0, None)

    # Save NIfTI files
    affine = np.eye(4)
    case_folder = out_dir / case_id
    case_folder.mkdir(parents=True, exist_ok=True)

    nib.save(nib.Nifti1Image(flair, affine), str(case_folder / f"{case_id}_flair.nii.gz"))
    nib.save(nib.Nifti1Image(t1, affine), str(case_folder / f"{case_id}_t1.nii.gz"))
    nib.save(nib.Nifti1Image(t1ce, affine), str(case_folder / f"{case_id}_t1ce.nii.gz"))
    nib.save(nib.Nifti1Image(t2, affine), str(case_folder / f"{case_id}_t2.nii.gz"))
    nib.save(nib.Nifti1Image(seg, affine), str(case_folder / f"{case_id}_seg.nii.gz"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic BraTS dataset for testing")
    parser.add_argument("--num_cases", type=int, default=8, help="Number of synthetic cases to generate")
    parser.add_argument("--out", type=str, default="data/raw", help="Target output directory")
    parser.add_argument("--depth", type=int, default=50, help="Axial slice depth")
    args = parser.parse_args()

    out_path = Path(args.out)
    print(f"Generating {args.num_cases} synthetic BraTS cases in {out_path}...")
    for i in range(1, args.num_cases + 1):
        cid = f"BraTS20_Mock_{i:03d}"
        create_mock_case(cid, out_path, shape=(240, 240, args.depth), seed=100 + i)
        print(f"  Created {cid}")
    print("Done generating synthetic BraTS dataset.")


if __name__ == "__main__":
    main()
