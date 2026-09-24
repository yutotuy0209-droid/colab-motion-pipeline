# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-24

### Added
- **Bidirectional Backlash & Hysteresis Fitter (`src/hysteresis_fitter.py`)**:
  - Independent forward and backward stroke regression.
  - Identification of backlash gap ($\Delta_{\text{backlash}}$), hysteresis area, and true 3-sigma repeatability.
- **Analytical G-Code Motion & Dwell Profiler (`src/gcode_analyzer.py`)**:
  - G-code parser to calculate exact machine travel, feed transitions, and G04 steady-state dwell windows.
- **Subpixel Grid Dot Centroid & Topology Extractor (`src/grid_extractor.py`)**:
  - Morphological Top-Hat illumination leveling and local intensity centroid (< 0.05 px repeatability).
- **Geometric QR Circle & Runout Estimator (`src/rotation_estimator.py`)**:
  - Householder QR linear least squares circle fitting, rotation center $(C_x, C_y)$, radius $R$, and peak-to-valley runout.
- **Autonomous Agent Experiment API (`src/agent_api.py`)**:
  - Standardized JSON facade for AI agents (Antigravity & Codex) to autonomously execute experiments in cloud/local.
- **Comprehensive Cloud Notebook (`notebooks/colab_autonomous_research_hub.ipynb`)**:
  - Interactive multi-engine demonstration notebook with `Open in Colab` badge.

## [0.1.0] - 2026-09-24

### Added
- Core POC Engine (`src/poc_core.py`) with GPU/CPU dual-backend.
- Linear Motion Error Decomposition (`src/evaluation.py`).
- Synthetic Motion Benchmark Generator (`src/synthetic_data.py`).
- Interactive Demo Notebook (`notebooks/colab_motion_analysis_demo.ipynb`).
