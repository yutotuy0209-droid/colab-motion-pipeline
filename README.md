# colab-motion-pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yutotuy0209-droid/colab-motion-pipeline/blob/main/notebooks/colab_motion_analysis_demo.ipynb)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Cloud GPU-Accelerated 2D Subpixel Motion Analysis Pipeline**  
> Scalable Phase-Only Correlation (POC), subpixel peak refinement, and linear motion error identification designed for Google Colab, local AI agents, and experimental verification.

---

## 1. Overview

`colab-motion-pipeline` is an independent, modular research toolkit designed to offload computationally intensive 2D image displacement analysis to Google Colab's cloud GPUs or local high-performance accelerators.

It integrates seamlessly with:
- **Antigravity (AGY)**: Mathematical modeling, documentation, and agent orchestration.
- **Google Colab (`colab-mcp`)**: Cloud GPU-accelerated batch FFT and matrix processing.
- **Codex**: Autonomous code optimization, C++ porting, and kernel acceleration.
- **GitHub**: Reproducible research repository and automated continuous tracking.

---

## 2. Mathematical Principles

### 2.1 2D Phase-Only Correlation (POC)
Given two consecutive image frames $f_1(x, y)$ and $f_2(x, y)$ of size $M \times N$:

1. **Windowing**: Apply 2D Hann window $w(x, y)$ to suppress spectral leakage:
   $$\tilde{f}_i(x, y) = f_i(x, y) \cdot w(x, y) \quad (i = 1, 2)$$

2. **Cross-Phase Spectrum**:
   $$F_i(u, v) = \mathcal{F}_{2D}\{\tilde{f}_i(x, y)\}$$
   $$R(u, v) = \frac{F_1(u, v) F_2^*(u, v)}{|F_1(u, v) F_2^*(u, v)| + \epsilon}$$

3. **Correlation Surface**:
   $$r(x, y) = \mathcal{F}_{2D}^{-1}\{R(u, v)\}$$

### 2.2 Subpixel Peak Refinement
Around the discrete integer peak $(x_p, y_p)$, 1D parabolic interpolation along each axis yields continuous subpixel shifts $(\delta_x, \delta_y)$:
$$\delta_x = \frac{r(x_p - 1, y_p) - r(x_p + 1, y_p)}{2 \left(r(x_p - 1, y_p) - 2r(x_p, y_p) + r(x_p + 1, y_p)\right)}$$

### 2.3 Linear Motion Error Decomposition
The relationship between commanded machine motion $L_{\text{cmd}}$ and optically measured displacement $L_{\text{meas}}$ is modeled as:
$$L_{\text{meas}} = \alpha L_{\text{cmd}} + \beta + \epsilon$$
- $\alpha$: Optical scale factor (calibrated µm/pixel slope).
- $\beta$: Systematic intercept and reversal backlash.
- $\epsilon$: Random repeatability uncertainty (evaluated via $3\sigma$).

---

## 3. Directory Structure

```
colab-motion-pipeline/
├── .gitignore
├── requirements.txt
├── CHANGELOG.md
├── README.md
├── notebooks/
│   └── colab_motion_analysis_demo.ipynb   # Interactive Colab notebook
└── src/
    ├── poc_core.py                         # Dual PyTorch (GPU) & NumPy (CPU) POC engine
    ├── evaluation.py                       # Linear motion model & 3-sigma error analysis
    └── synthetic_data.py                   # Continuous PSF dot grid generator
```

---

## 4. Quick Start

### Option A: Run Directly in Google Colab (Zero Local Setup)
Click the badge above or open:
[Open in Colab](https://colab.research.google.com/github/yutotuy0209-droid/colab-motion-pipeline/blob/main/notebooks/colab_motion_analysis_demo.ipynb)

Select **Runtime > Change runtime type > T4 GPU**, then execute the cells in sequence.

### Option B: Local Execution

```bash
# Clone the repository
git clone https://github.com/yutotuy0209-droid/colab-motion-pipeline.git
cd colab-motion-pipeline

# Install dependencies
pip install -r requirements.txt

# Run Python verification
python -c "from src.synthetic_data import generate_motion_sequence; from src.poc_core import compute_displacement; frames, tx, ty = generate_motion_sequence(5); dx, dy, p = compute_displacement(frames[0], frames[1]); print(f'Detected shift: dx={dx:.4f}px, dy={dy:.4f}px, peak={p:.4f}')"
```

---

## 5. Codex Handover Specification

All core computational kernels contain explicit handover headers adhering to the **AGY-Codex Cooperative Rules**:
- `[FOR CODEX REFACTOR]`
- `[INPUT/OUTPUT SPEC]`
- `[ALGORITHM INTENT]`
- `[TODO FOR CODEX]`

Codex agents can autonomously inspect, vectorize, and compile these kernels into high-throughput C++/CUDA shared libraries (`.so` / `.dll`).
