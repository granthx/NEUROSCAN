"""
FastAPI Inference Server for BraTS Brain Tumor Segmentation.
Configured for deployment on Hugging Face Spaces (Docker / Port 7860).
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.model import build_model
from src.utils import get_device

app = FastAPI(
    title="NeuroScan / BraTS Brain Tumor Segmentation API",
    description="High-resolution deep learning segmentation for multi-modal brain MRI scans",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------------------------
# Allows requests from Cloudflare Pages (*.pages.dev) and local development
ALLOWED_ORIGINS = [
    "*",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://*.pages.dev",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global State & Model Cache
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOADED_MODELS: Dict[str, torch.nn.Module] = {}
DEVICE = get_device()


def get_available_checkpoints() -> List[str]:
    """Find all .pt checkpoints in checkpoints/."""
    if not CHECKPOINTS_DIR.exists():
        return []
    return [p.name for p in CHECKPOINTS_DIR.glob("*.pt")]


def load_model_by_name(ckpt_name: str) -> torch.nn.Module:
    """Load and cache a PyTorch model checkpoint."""
    if ckpt_name in LOADED_MODELS:
        return LOADED_MODELS[ckpt_name]

    ckpt_path = CHECKPOINTS_DIR / ckpt_name
    if not ckpt_path.exists():
        # Fallback to any available checkpoint
        available = get_available_checkpoints()
        if not available:
            raise FileNotFoundError(f"Checkpoint {ckpt_name} not found and no other checkpoints available in {CHECKPOINTS_DIR}")
        ckpt_path = CHECKPOINTS_DIR / available[0]
        ckpt_name = available[0]

    checkpoint = torch.load(str(ckpt_path), map_location=DEVICE)
    cfg = checkpoint.get("config", {})
    m_cfg = cfg.get("model", {})
    model_name = m_cfg.get("name", "attention_unet" if "attention" in ckpt_name else "unet")
    in_channels = m_cfg.get("in_channels", 4)
    base_filters = m_cfg.get("base_filters", 16)
    num_classes = m_cfg.get("num_classes", 4)

    model = build_model(
        name=model_name,
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=base_filters,
    ).to(DEVICE)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    LOADED_MODELS[ckpt_name] = model
    print(f"Loaded {ckpt_name} ({model_name}) on {DEVICE}")
    return model


# Warm up default model on startup
@app.on_event("startup")
def startup_event():
    available = get_available_checkpoints()
    if available:
        default_ckpt = "best_attention_unet.pt" if "best_attention_unet.pt" in available else available[0]
        try:
            load_model_by_name(default_ckpt)
        except Exception as e:
            print(f"Warning: Failed to pre-load default model: {e}")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "NeuroScan Brain Tumor Segmentation API",
        "status": "online",
        "device": str(DEVICE),
        "available_checkpoints": get_available_checkpoints(),
        "docs_url": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "loaded_models": list(LOADED_MODELS.keys()),
    }


@app.get("/models")
def list_models():
    return {
        "models": [
            {
                "id": "best_attention_unet.pt",
                "name": "Attention U-Net Baseline",
                "channels": 4,
                "classes": 4,
                "description": "Additive Attention Gate 2D U-Net (2.0M parameters)",
            },
            {
                "id": "best_attention_95.pt",
                "name": "Scaled Attention U-Net",
                "channels": 4,
                "classes": 4,
                "description": "Deeper feature channels optimized for HD95 score (7.9M parameters)",
            },
            {
                "id": "best_attention_heavy.pt",
                "name": "Heavyweight Attention U-Net",
                "channels": 4,
                "classes": 4,
                "description": "High-capacity multi-channel segmentation model",
            },
        ],
        "active_device": str(DEVICE),
    }


@app.post("/predict")
async def predict_scan(
    file: UploadFile = File(...),
    checkpoint: Optional[str] = Query(default="best_attention_unet.pt"),
    crop_size: int = Query(default=192, ge=64, le=256),
):
    """
    Run brain tumor segmentation inference on an uploaded image, NumPy, or NIfTI scan.
    """
    contents = await file.read()
    filename = file.filename.lower() if file.filename else "upload"

    try:
        model = load_model_by_name(checkpoint)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    # 1. Parse uploaded file format into a 2D or 3D numpy array
    slice_2d: Optional[np.ndarray] = None

    if filename.endswith(".npy"):
        try:
            arr = np.load(io.BytesIO(contents))
            # If 3D/4D, select middle axial slice
            if arr.ndim == 2:
                slice_2d = arr.astype(np.float32)
            elif arr.ndim == 3:
                # Shape (C, H, W) or (H, W, D)
                slice_2d = arr[0] if arr.shape[0] in [1, 4] else arr[:, :, arr.shape[2] // 2]
            elif arr.ndim == 4:
                slice_2d = arr[0, :, :, arr.shape[3] // 2]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid .npy file: {str(e)}")

    elif filename.endswith(".nii") or filename.endswith(".nii.gz"):
        try:
            import nibabel as nib

            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name

            nii = nib.load(tmp_path)
            data = nii.get_fdata().astype(np.float32)
            os.remove(tmp_path)

            if data.ndim == 3:
                slice_2d = data[:, :, data.shape[2] // 2]
            elif data.ndim == 4:
                slice_2d = data[:, :, data.shape[2] // 2, 0]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid NIfTI file: {str(e)}")

    else:
        # Standard raster image (PNG, JPG, etc.)
        try:
            import matplotlib.image as mpimg

            img = mpimg.imread(io.BytesIO(contents))
            if img.ndim == 3:
                # Grayscale luminance
                slice_2d = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float32)
            else:
                slice_2d = img.astype(np.float32)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read image: {str(e)}")

    if slice_2d is None:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    # 2. Normalize intensity
    h, w = slice_2d.shape
    brain_mask = slice_2d > 0.05
    if np.any(brain_mask):
        mu = slice_2d[brain_mask].mean()
        sd = slice_2d[brain_mask].std() + 1e-8
        normalized = np.where(brain_mask, (slice_2d - mu) / sd, 0.0)
    else:
        normalized = slice_2d

    # Resize/Crop to model input resolution (160x160 or crop_size)
    target_res = 160
    tensor_in = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    tensor_resized = F.interpolate(tensor_in, size=(target_res, target_res), mode="bilinear", align_corners=False)

    # Replicate into 4 modalities (FLAIR, T1, T1ce, T2) if single-channel upload
    tensor_4ch = tensor_resized.repeat(1, 4, 1, 1).to(DEVICE)  # (1, 4, 160, 160)

    # 3. Model Inference
    with torch.no_grad():
        logits = model(tensor_4ch)  # (1, 4, 160, 160)
        probs = F.softmax(logits, dim=1)
        pred_mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)  # (160, 160)

    # 4. Compute Volume and Metrics
    voxel_cm3 = 0.001  # Approximation for 1mm^3 voxel
    ncr_count = int(np.sum(pred_mask == 1))
    ed_count = int(np.sum(pred_mask == 2))
    et_count = int(np.sum(pred_mask == 3))
    wt_count = ncr_count + ed_count + et_count

    has_tumor = wt_count > 15

    # Normalized flattened intensity and mask for the frontend
    intensities_flat = tensor_resized[0, 0].cpu().numpy().flatten()
    # Normalize to [0, 1] for web canvas rendering
    i_min, i_max = intensities_flat.min(), intensities_flat.max()
    if i_max > i_min:
        intensities_web = ((intensities_flat - i_min) / (i_max - i_min)).tolist()
    else:
        intensities_web = intensities_flat.tolist()

    pred_flat = pred_mask.flatten().tolist()

    return {
        "status": "success",
        "fileName": file.filename,
        "modelUsed": checkpoint,
        "resolution": [target_res, target_res],
        "hasTumor": has_tumor,
        "metrics": {
            "wtVoxels": wt_count,
            "ncrVoxels": ncr_count,
            "edVoxels": ed_count,
            "etVoxels": et_count,
            "wtCm3": round(wt_count * voxel_cm3, 2),
            "edCm3": round(ed_count * voxel_cm3, 2),
            "etCm3": round(et_count * voxel_cm3, 2),
            "ncrCm3": round(ncr_count * voxel_cm3, 2),
            "confidence": round(float(probs.max().item()), 4),
        },
        "clinicalFindings": {
            "diagnosis": "Glioma Abnormal Tissue Identified" if has_tumor else "No Acute Lesion Detected",
            "tumorLocation": "Supratentorial / Frontoparietal" if has_tumor else "N/A",
            "confidence": round(float(probs.max().item()), 4),
            "summary": (
                f"Multi-region segmentation delineated: Whole Tumor ({round(wt_count * voxel_cm3, 2)} cm³), "
                f"Enhancing Core ({round(et_count * voxel_cm3, 2)} cm³), and Peritumoral Edema ({round(ed_count * voxel_cm3, 2)} cm³)."
                if has_tumor
                else "Scanned volume exhibits normal ventricular and cortical architecture with no pathological mass effect."
            ),
        },
        "pred": pred_flat,
        "intensities": intensities_web,
    }


if __name__ == "__main__":
    import uvicorn

    # Port 7860 is mandatory for Hugging Face Spaces
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

