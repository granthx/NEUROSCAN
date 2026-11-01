"""
Brain Tumor Segmentation Interactive Explorer (Streamlit).
Allows interactive slice browsing, multi-modal MRI inspection,
ground truth vs model prediction overlay comparison, and metrics display.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
import streamlit as st
import torch

from src.metrics import evaluate_case_volume
from src.model import build_model
from src.predict import BRATS_CMAP, predict_volume
from src.utils import get_device

st.set_page_config(
    page_title="BraTS Brain Tumor Segmentation AI",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: #f1f2f6; }
    .stMetric { background-color: #1e222d; padding: 12px; border-radius: 8px; border: 1px solid #2d3436; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_segmentation_model(ckpt_path: str):
    device = get_device()
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint.get("config", {})
    m_cfg = cfg.get("model", {})
    model_name = m_cfg.get("name", "unet")
    in_channels = m_cfg.get("in_channels", 4)
    base_filters = m_cfg.get("base_filters", 16)

    model = build_model(
        name=model_name,
        in_channels=in_channels,
        num_classes=4,
        base_filters=base_filters,
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model, device, model_name


def main():
    st.title("🧠 BraTS Multi-Modal Brain Tumour Segmentation")
    st.caption("AI-assisted 3D Segmentation of Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET)")

    st.sidebar.header("Configuration & Data")

    # Checkpoint selection
    ckpt_dir = Path("checkpoints")
    ckpts = list(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
    if not ckpts:
        st.warning("No checkpoint found in `checkpoints/`. Train a model first!")
        return

    selected_ckpt = st.sidebar.selectbox("Model Checkpoint", [str(p) for p in ckpts])

    # Case selection
    raw_dir = Path("data/raw")
    cases = sorted([p.name for p in raw_dir.iterdir() if p.is_dir()]) if raw_dir.exists() else []
    if not cases:
        st.warning("No raw cases found in `data/raw/`.")
        return

    selected_case = st.sidebar.selectbox("BraTS Case", cases)
    case_folder = raw_dir / selected_case

    # Load Model
    model, device, model_name = load_segmentation_model(selected_ckpt)
    is_25d = model_name == "unet_25d"

    # Run / Cache Volume Prediction
    with st.spinner(f"Running inference for {selected_case}..."):
        pred_vol, flair, gt_seg, affine = predict_volume(case_folder, model, device, is_25d=is_25d)

    d = flair.shape[2]

    # Metrics summary if GT available
    if gt_seg is not None:
        metrics = evaluate_case_volume(pred_vol, gt_seg)
        st.subheader("Patient Volume Performance Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Whole Tumor (WT) Dice", f"{metrics['dice_WT']:.3f}", f"HD95: {metrics['hd95_WT']:.1f} mm")
        c2.metric("Tumor Core (TC) Dice", f"{metrics['dice_TC']:.3f}", f"HD95: {metrics['hd95_TC']:.1f} mm")
        c3.metric("Enhancing Tumor (ET) Dice", f"{metrics['dice_ET']:.3f}", f"HD95: {metrics['hd95_ET']:.1f} mm")
        c4.metric("Mean Dice", f"{metrics['dice_mean']:.3f}", f"Mean HD95: {metrics['hd95_mean']:.1f} mm")

    st.divider()

    # Slice Navigator
    st.sidebar.subheader("Slice Controls")
    slice_idx = st.sidebar.slider("Axial Slice", 0, d - 1, d // 2)
    alpha = st.sidebar.slider("Overlay Opacity", 0.1, 1.0, 0.6)

    # Visualization Display
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**1. Raw FLAIR MRI**")
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0e1117")
        ax.imshow(flair[:, :, slice_idx].T, cmap="gray", origin="lower")
        ax.axis("off")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**2. Ground Truth Segmentation**")
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0e1117")
        ax.imshow(flair[:, :, slice_idx].T, cmap="gray", origin="lower")
        if gt_seg is not None:
            ax.imshow(gt_seg[:, :, slice_idx].T, cmap=BRATS_CMAP, vmin=0, vmax=3, origin="lower", alpha=alpha)
        ax.axis("off")
        st.pyplot(fig)
        plt.close()

    with col3:
        st.markdown(f"**3. AI Prediction ({model_name})**")
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0e1117")
        ax.imshow(flair[:, :, slice_idx].T, cmap="gray", origin="lower")
        ax.imshow(pred_vol[:, :, slice_idx].T, cmap=BRATS_CMAP, vmin=0, vmax=3, origin="lower", alpha=alpha)
        ax.axis("off")
        st.pyplot(fig)
        plt.close()

    st.markdown(
        """
        <div style="background-color: #1e222d; padding: 10px; border-radius: 6px; text-align: center;">
            <span style="color: #e74c3c; font-weight: bold;">■ Necrotic Core (1)</span> &nbsp;&nbsp;|&nbsp;&nbsp;
            <span style="color: #2ecc71; font-weight: bold;">■ Peritumoral Edema (2)</span> &nbsp;&nbsp;|&nbsp;&nbsp;
            <span style="color: #f1c40f; font-weight: bold;">■ Enhancing Tumor (3)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Research and educational use only. Not approved for clinical diagnostics.")


if __name__ == "__main__":
    main()
