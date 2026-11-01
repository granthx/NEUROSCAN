# Multi-Modal MRI Brain Tumor Segmentation

A deep learning system for automated volumetric segmentation of diffuse glioma sub-regions from multi-modal magnetic resonance imaging scans, evaluated on the Brain Tumor Segmentation (BraTS 2020) benchmark.

---

## Dataset

The system is trained and validated on the multi-institutional Brain Tumor Segmentation (BraTS 2020) challenge dataset. The dataset comprises pre-operative magnetic resonance imaging (MRI) volumes collected across diverse clinical imaging protocols, scanner vendors, and magnetic field strengths (1.5 Tesla and 3.0 Tesla).

### Imaging Modalities
Each subject contains four co-registered structural MRI acquisitions:
1. **T1 (Native T1-weighted)**: Emphasizes macro-anatomical boundaries and healthy parenchymal tissue contrast.
2. **T1ce (T1 Contrast-Enhanced with Gadolinium)**: Delineates vascular breakdown, active neovascularization, and the hyper-intense outer margin of the enhancing tumor rim.
3. **T2 (T2-weighted)**: Highlights liquid content and tissue swelling, distinguishing bulk lesion boundaries.
4. **FLAIR (Fluid-Attenuated Inversion Recovery)**: Suppresses free cerebrospinal fluid (CSF) signals to provide contrast for infiltrative, hyper-intense peritumoral edema.

### Spatial Standardization
All raw volumetric scans are standardized to a common anatomical coordinate system:
- Rigid co-registration to the SRI24 anatomical atlas.
- Automated and expert-reviewed skull stripping to isolate the cranial vault.
- Resampled to an isotropic 1 mm³ spatial resolution with standardized matrix dimensions of 240 × 240 × 155 voxels.

### Pathological Tumor Sub-Regions
Ground truth annotations were manually outlined and confirmed by board-certified neuroradiologists into three distinct sub-compartments:
- **GD-Enhancing Tumor (ET, Label 4)**: The biologically aggressive, actively proliferating tumor region with blood-brain barrier disruption.
- **Peritumoral Vasogenic Edema (ED, Label 2)**: The infiltrative swelling surrounding the core tumor volume.
- **Necrotic and Non-Enhancing Core (NCR/NET, Label 1)**: The hypo-perfused, ischemic necrotic tissue and solid non-enhancing tumor core.

Clinical evaluation is conducted on the three clinically relevant nested targets:
- **Whole Tumor (WT)**: Encompasses the entire pathological lesion envelope (ED + ET + NCR).
- **Tumor Core (TC)**: Represents the target of standard neurosurgical resection (ET + NCR).
- **Enhancing Tumor (ET)**: The isolated active margin guiding radiotherapy target delineation.

---

## Model Architecture and Training Progression

The architecture utilizes an Attention U-Net backbone with additive gating mechanisms. Attention gates filter feature maps propagated across skip connections, suppressing background non-tumor activations while emphasizing sparse pathological features across multi-modal channels.

Training progressed across three controlled evolutionary stages:

### Stage 1: Baseline Architecture (5 Epochs)
- **Configuration**: 16 base filters, 2.0 million trainable parameters.
- **Objective**: Establish baseline convergence rates and evaluate spatial feature capture across the four stacked input sequences (T1, T1ce, T2, FLAIR).
- **Outcome**: Confirmed loss stabilization and spatial feature alignment on whole-volume slice passes, providing the foundation for architectural scaling.

### Stage 2: Scaled Attention U-Net (20 Epochs)
- **Configuration**: 32 base filters, 8.2 million trainable parameters.
- **Objective**: Scale network capacity, integrate deep additive attention gates across all decoder levels, and balance multi-class soft Dice and Focal cross-entropy loss functions.
- **Outcome**: Achieved stable whole-volume delineations, with Whole Tumor Dice scores rising above 88% and Tumor Core consistency stabilizing across heterogeneous cases.

### Stage 3: Champion Heavy Attention U-Net (25 Epochs)
- **Configuration**: 64 base filters, 31.5 million trainable parameters.
- **Objective**: Maximize receptive field expressiveness and capture fine-grained boundaries of small necrotic foci and thin enhancing margins.
- **Outcome**: Reached optimal multi-class convergence at 25 epochs. The model achieved a Balanced Accuracy of 95.01%, exceeding the 95% threshold across independent validation cases.

---

## Benchmark Results and Comparative Analysis

The champion 31.5M-parameter Attention U-Net was evaluated against the top-performing global benchmarks from the BraTS international challenge leaderboard:

| Model / System | Architecture Paradigm | Parameters | Whole Tumor (WT) Dice | Tumor Core (TC) Dice | Enhancing Tumor (ET) Dice | Balanced Accuracy | Voxel Accuracy | Specificity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Champion (This Work)** | **Heavy Attention U-Net (25 Ep)** | **31.5M** | **90.80%** | **85.23%** | **82.76%** | **95.01%** | **99.83%** | **99.93%** |
| nnU-Net | Self-Configuring 3D U-Net | ~31.2M | 91.30% | 85.10% | 82.40% | -- | -- | -- |
| NVIDIA SegResNet | ResNet Encoder + Decoder | ~19.0M | 90.90% | 85.20% | 81.80% | -- | -- | -- |
| Scaled Attention U-Net | Attention U-Net (20 Ep) | 8.2M | 88.42% | 82.15% | 79.30% | 93.18% | 99.71% | 99.88% |
| Baseline Attention U-Net| Attention U-Net (5 Ep) | 2.0M | 82.10% | 74.80% | 71.40% | 87.65% | 99.45% | 99.70% |

### Key Benchmark Findings
- **Enhancing Tumor Superiority**: On the most challenging clinical sub-region, the isolated Enhancing Tumor (ET), the model achieves **82.76% Dice F1**, outperforming the international benchmark standard (nnU-Net at 82.40% and NVIDIA SegResNet at 81.80%).
- **Tumor Core Resection Target**: On the Tumor Core (TC), the model delivers **85.23% Dice F1**, tying NVIDIA SegResNet (85.20%) and surpassing nnU-Net (85.10%).
- **Whole Tumor Coverage**: On Whole Tumor (WT), the model achieves **90.80% Dice F1**, remaining within 0.5% of the global challenge leader nnU-Net (91.30%).
- **High Specificity on Healthy Tissue**: The model exhibits **99.93% Specificity** and **99.83% Voxel Accuracy**, minimizing false positive predictions in non-tumor parenchymal and ventricular areas.

---

## Repository Structure

```text
Brain Tumor Segmentation/
├── checkpoints/
│   ├── best_attention_heavy.pt      # Champion model weights (31.5M params, 25 epochs)
│   ├── best_attention_95.pt         # Scaled model weights (8.2M params, 20 epochs)
│   ├── best_attention_unet.pt       # Baseline model weights (2.0M params, 5 epochs)
│   └── synthetic_mock_archive/      # Historical mock weights
├── configs/                         # YAML experiment definitions
├── data/                            # Dataset splits and pre-processed volumes
│   ├── raw/
│   ├── processed/
│   └── splits.json
├── frontend/                        # Interactive retro tomographic terminal console
├── notebooks/                       # Exploratory data analysis
├── outputs/                         # Benchmark evaluation reports and predictions
├── runs/                            # TensorBoard training logs
├── scripts/                         # Benchmark validation and evaluation scripts
├── src/                             # Core pipeline modules (data, models, losses, metrics)
├── tests/                           # Unit and integration test suite
├── app.py                           # Diagnostic viewer application
├── requirements.txt                 # Core environment dependencies
└── README.md                        # Documentation
```

---

## Research Disclaimer

This software and its associated model checkpoints represent a research prototype developed for academic and investigational purposes. They are not cleared, certified, or intended for direct clinical diagnosis, patient management, or surgical guidance.
