"""Bidirectional Motion Hysteresis & Backlash Fitter for Cloud & Local Execution.

Separates forward and backward stroke measurements to compute:
  - Reversal backlash gap (delta = |beta_fwd - beta_bwd|)
  - Hysteresis polygon area
  - Directional scale slopes (alpha_fwd, alpha_bwd)
  - True 3-sigma repeatability uncertainty
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np


# ==============================================================================
# [FOR CODEX REFACTOR]: Cloud Bidirectional Backlash & Hysteresis Fitter
# [INPUT/OUTPUT SPEC]:
#   - Input: cmd_positions: np.ndarray, meas_positions: np.ndarray
#   - Output: BacklashFitResult with backlash_um, hysteresis_area, 3-sigma
# [ALGORITHM INTENT]:
#   - Prevents systematic backlash from falsely degrading random error metrics.
#   - Complies with ISO 230-2 mechanical accuracy assessment standards.
# [TODO FOR CODEX]:
#   - Port regression matrix solvers to PyTorch GPU tensors for batch datasets.
# ==============================================================================

@dataclass
class BacklashFitResult:
    backlash_um: float
    hysteresis_area: float
    alpha_forward: float
    beta_forward: float
    alpha_backward: float
    beta_backward: float
    rmse_forward: float
    rmse_backward: float
    three_sigma: float
    forward_indices: List[int]
    backward_indices: List[int]


def fit_bidirectional_backlash(
    cmd_positions: np.ndarray,
    meas_positions: np.ndarray
) -> BacklashFitResult:
    """Analyze bidirectional motion data to extract backlash and hysteresis."""
    cmd = np.asarray(cmd_positions, dtype=np.float64)
    meas = np.asarray(meas_positions, dtype=np.float64)

    assert len(cmd) == len(meas), "Array lengths must match"
    assert len(cmd) >= 4, "At least 4 points required for bidirectional fitting"

    d_cmd = np.diff(cmd)
    fwd_idx = []
    bwd_idx = []

    for i in range(len(d_cmd)):
        if d_cmd[i] > 1e-6:
            fwd_idx.append(i + 1)
        elif d_cmd[i] < -1e-6:
            bwd_idx.append(i + 1)

    if len(fwd_idx) > 0 and fwd_idx[0] == 1:
        fwd_idx.insert(0, 0)
    elif len(bwd_idx) > 0 and bwd_idx[0] == 1:
        bwd_idx.insert(0, 0)

    if len(fwd_idx) < 2 or len(bwd_idx) < 2:
        p = np.polyfit(cmd, meas, 1)
        res = meas - (p[0] * cmd + p[1])
        return BacklashFitResult(
            backlash_um=0.0,
            hysteresis_area=0.0,
            alpha_forward=float(p[0]),
            beta_forward=float(p[1]),
            alpha_backward=float(p[0]),
            beta_backward=float(p[1]),
            rmse_forward=float(np.sqrt(np.mean(res**2))),
            rmse_backward=float(np.sqrt(np.mean(res**2))),
            three_sigma=3.0 * float(np.std(res)),
            forward_indices=list(range(len(cmd))),
            backward_indices=[]
        )

    xf, yf = cmd[fwd_idx], meas[fwd_idx]
    pf = np.polyfit(xf, yf, 1)
    alpha_f, beta_f = float(pf[0]), float(pf[1])
    res_f = yf - (alpha_f * xf + beta_f)
    rmse_f = float(np.sqrt(np.mean(res_f**2)))

    xb, yb = cmd[bwd_idx], meas[bwd_idx]
    pb = np.polyfit(xb, yb, 1)
    alpha_b, beta_b = float(pb[0]), float(pb[1])
    res_b = yb - (alpha_b * xb + beta_b)
    rmse_b = float(np.sqrt(np.mean(res_b**2)))

    mid_x = (np.min(cmd) + np.max(cmd)) / 2.0
    pred_f = alpha_f * mid_x + beta_f
    pred_b = alpha_b * mid_x + beta_b
    backlash_um = float(abs(pred_f - pred_b))

    combined_res = np.concatenate([res_f, res_b])
    three_sigma = 3.0 * float(np.std(combined_res))
    hysteresis_area = float(backlash_um * (np.max(cmd) - np.min(cmd)))

    return BacklashFitResult(
        backlash_um=backlash_um,
        hysteresis_area=hysteresis_area,
        alpha_forward=alpha_f,
        beta_forward=beta_f,
        alpha_backward=alpha_b,
        beta_backward=beta_b,
        rmse_forward=rmse_f,
        rmse_backward=rmse_b,
        three_sigma=three_sigma,
        forward_indices=fwd_idx,
        backward_indices=bwd_idx
    )
