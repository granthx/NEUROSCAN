"""
Helper script to organize or verify the BraTS dataset.
Provides guidance on downloading BraTS and checks folder compliance.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REQUIRED_MODALITIES = ["flair", "t1", "t1ce", "t2", "seg"]


def print_download_instructions(target_path: Path) -> None:
    print("=" * 70)
    print("BraTS Dataset Acquisition Instructions")
    print("=" * 70)
    print("BraTS data is not yet found at the specified location:")
    print(f"  Target: {target_path.resolve()}\n")
    print("Where to obtain BraTS:")
    print("1. Kaggle (BraTS 2020 / BraTS 2021):")
    print("   - BraTS 2020: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation")
    print("   - BraTS 2021: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1")
    print("2. Synapse BraTS Challenge Official Portal:")
    print("   - https://www.synapse.org/#!Synapse:syn25829424 (BraTS 2021 Task 1)")
    print("3. RSNA-ASNR-MICCAI BraTS:")
    print("   - https://www.med.upenn.edu/cbica/brats/\n")
    print("Expected layout:")
    print("  <data_dir>/<CASE_ID>/<CASE_ID>_flair.nii.gz")
    print("  <data_dir>/<CASE_ID>/<CASE_ID>_t1.nii.gz")
    print("  <data_dir>/<CASE_ID>/<CASE_ID>_t1ce.nii.gz")
    print("  <data_dir>/<CASE_ID>/<CASE_ID>_t2.nii.gz")
    print("  <data_dir>/<CASE_ID>/<CASE_ID>_seg.nii.gz\n")
    print("Tip: If you want to test and develop immediately on your CPU without")
    print("downloading tens of gigabytes, run:")
    print("  python scripts/generate_mock_brats.py --num_cases 8 --out data/raw")
    print("=" * 70)


def verify_and_organize(source_path: Path, dest_path: Path) -> int:
    source_path = source_path.resolve()
    dest_path = dest_path.resolve()

    if not source_path.exists():
        print_download_instructions(source_path)
        return 1

    dest_path.mkdir(parents=True, exist_ok=True)
    candidate_cases = sorted([p for p in source_path.iterdir() if p.is_dir()])

    valid_cases = 0
    for case_dir in candidate_cases:
        cid = case_dir.name
        files = {m: list(case_dir.glob(f"*{m}.nii*")) for m in REQUIRED_MODALITIES}
        missing = [m for m, found in files.items() if not found]
        if missing:
            continue

        target_case_dir = dest_path / cid
        target_case_dir.mkdir(exist_ok=True)
        for m, found in files.items():
            src_file = found[0]
            dst_file = target_case_dir / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
        valid_cases += 1

    print(f"Verified and organized {valid_cases} compliant BraTS cases into: {dest_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BraTS Dataset Verification and Organization Helper")
    parser.add_argument("--src", type=str, default="data/raw", help="Path to raw or downloaded BraTS directory")
    parser.add_argument("--dest", type=str, default="data/raw", help="Destination path for organized data")
    args = parser.parse_args()

    sys.exit(verify_and_organize(Path(args.src), Path(args.dest)))
