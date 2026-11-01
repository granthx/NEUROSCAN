"""
Unit tests for data preprocessing and patient-level splitting.
"""
from pathlib import Path
import numpy as np
import pytest

from src.data import create_splits


def test_zero_leakage_patient_splits(tmp_path):
    # Create 10 dummy patient folders
    for i in range(10):
        case_dir = tmp_path / f"Patient_{i:03d}"
        case_dir.mkdir()
        # Create empty img.npy
        np.save(case_dir / "img.npy", np.zeros((4, 10, 16, 16), dtype=np.float16))

    splits = create_splits(
        processed_dir=tmp_path,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    # Verify no overlap between splits (zero patient leakage)
    assert train_set.isdisjoint(val_set), "Train and Val splits share patients!"
    assert train_set.isdisjoint(test_set), "Train and Test splits share patients!"
    assert val_set.isdisjoint(test_set), "Val and Test splits share patients!"

    # Total cases preserved
    all_cases = train_set | val_set | test_set
    assert len(all_cases) == 10


def test_preprocess_and_dataset(tmp_path):
    from scripts.generate_mock_brats import create_mock_case
    from src.data import preprocess_case, BraTSSliceDataset

    raw_case_dir = tmp_path / "raw" / "BraTS_Test_001"
    create_mock_case("BraTS_Test_001", tmp_path / "raw", shape=(240, 240, 10), seed=42)

    proc_dir = tmp_path / "processed" / "BraTS_Test_001"
    success = preprocess_case(raw_case_dir, proc_dir, crop_size=192)
    assert success is True
    assert (proc_dir / "img.npy").exists()
    assert (proc_dir / "seg.npy").exists()

    img = np.load(proc_dir / "img.npy")
    seg = np.load(proc_dir / "seg.npy")

    # Shape: (4, D, 192, 192) and (D, 192, 192)
    assert img.shape == (4, 10, 192, 192)
    assert seg.shape == (10, 192, 192)
    # Check label remapping: 4 should not exist, 3 should exist
    assert 4 not in np.unique(seg), "Original BraTS label 4 was not remapped to 3!"
    assert 3 in np.unique(seg), "Remapped label 3 (ET) not found!"

    # Test BraTSSliceDataset in 2D mode
    ds_2d = BraTSSliceDataset(tmp_path / "processed", cases=["BraTS_Test_001"], mode="2d", empty_keep=1.0)
    assert len(ds_2d) > 0
    x, y = ds_2d[0]
    assert x.shape == (4, 192, 192)
    assert y.shape == (192, 192)

    # Test BraTSSliceDataset in 2.5D mode (12 channels)
    ds_25d = BraTSSliceDataset(tmp_path / "processed", cases=["BraTS_Test_001"], mode="2.5d", empty_keep=1.0)
    x_25d, y_25d = ds_25d[0]
    assert x_25d.shape == (12, 192, 192)
    assert y_25d.shape == (192, 192)

