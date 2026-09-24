# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-24

### Added
- **Core POC Engine (`src/poc_core.py`)**:
  - Implemented dual-backend 2D Phase-Only Correlation (PyTorch CUDA acceleration with NumPy/SciPy CPU fallback).
  - 2D Hann spectral windowing to suppress boundary leakage.
  - Subpixel peak refinement using 1D parabolic interpolation around the correlation maximum.
  - Embedded Codex refactor handover tags (`[FOR CODEX REFACTOR]`, `[INPUT/OUTPUT SPEC]`, `[ALGORITHM INTENT]`, `[TODO FOR CODEX]`).
- **Linear Motion Error Decomposition (`src/evaluation.py`)**:
  - Mathematical modeling of $L_{\text{meas}} = \alpha L_{\text{cmd}} + \beta + \epsilon$.
  - Automated estimation of calibration slope ($\alpha$), intercept/backlash ($\beta$), RMSE, and random reproducibility uncertainty ($3\sigma$).
- **Synthetic Motion Benchmark Generator (`src/synthetic_data.py`)**:
  - Analytical PSF 2D Gaussian dot grid rendering with continuous subpixel translation and Gaussian sensor noise.
- **Google Colab Interactive Demo (`notebooks/colab_motion_analysis_demo.ipynb`)**:
  - End-to-end executable notebook with `Open in Colab` badge.
  - Automated GPU device detection, batch motion processing, and dual-panel academic visualization.
- **Repository Setup**:
  - Added `.gitignore` and `requirements.txt`.
