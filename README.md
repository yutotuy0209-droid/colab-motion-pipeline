# colab-motion-pipeline

[![Open In Colab (Research Hub)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yutotuy0209-droid/colab-motion-pipeline/blob/main/notebooks/colab_autonomous_research_hub.ipynb)
[![Open In Colab (POC Demo)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yutotuy0209-droid/colab-motion-pipeline/blob/main/notebooks/colab_motion_analysis_demo.ipynb)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Autonomous Cloud & Local Motion Analysis Engines**  
> Scalable Phase-Only Correlation (POC), Bidirectional Backlash Hysteresis, Analytical G-Code Profiling, Subpixel Grid Extraction, and Geometric QR Rotation Center Estimation.

---

## 📱 Mobile & Remote Research Navigation (PC-less Access)

休日にPCを開かずにスマートフォンやタブレットから研究を進めるための直結リンク：

- 🏠 **[卒業研究メインリポジトリ (graduation-research-motion-analysis)](https://github.com/yutotuy0209-droid/graduation-research-motion-analysis)**
- 📖 **[休日モバイル作業完全ガイド](https://github.com/yutotuy0209-droid/graduation-research-motion-analysis/blob/main/docs/MOBILE_WORK_GUIDE.md)**
- 📐 **[来週実験計画＆シンプルイズベストUI設計書](https://github.com/yutotuy0209-droid/graduation-research-motion-analysis/blob/main/docs/2026-09-26_%E6%9D%A5%E9%80%B1%E5%AE%9F%E9%A8%93%E8%A8%88%E7%94%BB%E3%81%A8%E3%82%B7%E3%83%B3%E3%83%97%E3%83%AB%E3%82%A4%E3%82%BA%E3%83%99%E3%82%B9%E3%83%88UI%E6%94%B9%E4%BF%AE%E8%A8%AD%E8%A8%88%E6%9B%B8.md)**
- ✅ **[来週実験チェックリスト (Day 1〜5)](https://github.com/yutotuy0209-droid/graduation-research-motion-analysis/blob/main/docs/EXPERIMENT_CHECKLIST.md)**
- 🔬 **[アルゴリズムコア数理モデルリファレンス](https://github.com/yutotuy0209-droid/graduation-research-motion-analysis/blob/main/docs/ALGORITHM_CORE_REFERENCE.md)**

---

## 1. Cloud-Executable Notebooks

Run directly on Google Colab with free Cloud GPU acceleration (zero setup required):

| Notebook | Purpose | Link |
| :--- | :--- | :--- |
| **Autonomous Research Hub** | Comprehensive 5-engine cloud execution (Backlash, G-Code, QR Circle, Grid) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yutotuy0209-droid/colab-motion-pipeline/blob/main/notebooks/colab_autonomous_research_hub.ipynb) |
| **2D POC Subpixel Demo** | High-throughput Phase-Only Correlation & linear error analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yutotuy0209-droid/colab-motion-pipeline/blob/main/notebooks/colab_motion_analysis_demo.ipynb) |

---

## 2. Core Computational Engines

```
colab-motion-pipeline/
├── notebooks/
│   ├── colab_autonomous_research_hub.ipynb   # Comprehensive 5-engine cloud demonstration
│   └── colab_motion_analysis_demo.ipynb       # 2D GPU POC interactive notebook
└── src/
    ├── poc_core.py            # Dual PyTorch (CUDA) & NumPy (CPU) POC engine (<0.05px)
    ├── hysteresis_fitter.py   # Forward/backward stroke separation & backlash gap (ISO 230-2)
    ├── gcode_analyzer.py      # Analytical G-code simulation & G04 camera dwell timing
    ├── grid_extractor.py      # Morphological Top-Hat leveling & subpixel centroid
    ├── rotation_estimator.py  # Householder QR linear least squares circle & runout
    ├── evaluation.py          # Linear error modeling (L_meas = alpha * L_cmd + beta + eps)
    ├── agent_api.py           # Machine-readable JSON API for AI agent autonomous execution
    └── synthetic_data.py      # Exact Fourier phase shift benchmark generator
```

---

## 3. Agent Autonomous API (`src/agent_api.py`)

AI agents (Antigravity and Codex) can execute actions programmatically with clean JSON I/O:

```bash
# Analyze G-code travel distance & camera dwell rest windows
python src/agent_api.py analyze_gcode --params '{"gcode": "G01 X0.010 F1.0\nG04 X1.5"}' --json

# Fit bidirectional backlash from motion coordinates
python src/agent_api.py fit_backlash --params '{"cmd_positions": [0, 10, 20, 10, 0], "meas_positions": [0, 9.8, 19.5, 9.1, -0.4]}' --json

# Estimate table rotation center via QR decomposition
python src/agent_api.py estimate_rotation --params '{"points": [[100, 200], [150, 220], [200, 200], [150, 180]]}' --json
```

---

## 4. Codex Handover Specification

All computational kernels are tagged with standardized Codex handover blocks:
- `[FOR CODEX REFACTOR]`
- `[INPUT/OUTPUT SPEC]`
- `[ALGORITHM INTENT]`
- `[TODO FOR CODEX]`
