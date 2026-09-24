"""Motion Analysis Evaluation and Linear Error Identification Module.

Decomposes machine command vs image measurement into:
  - Scale factor (alpha)
  - Reversal backlash (beta)
  - Random reproducibility error (epsilon / 3-sigma)
"""

from typing import Dict, Any, Tuple
import numpy as np


# ==============================================================================
# [FOR CODEX REFACTOR]: Motion Command vs Measurement Error Decomposition
# [INPUT/OUTPUT SPEC]:
#   - Input:
#       cmd_positions: np.ndarray, 1D array of commanded positions [µm]
#       meas_displacements: np.ndarray, 1D array of measured displacements [µm]
#   - Output:
#       metrics: Dict[str, Any] containing:
#           - 'alpha': float, scale calibration slope
#           - 'beta': float, intercept / backlash offset [µm]
#           - 'residuals': np.ndarray, point-by-point residual error [µm]
#           - 'rmse': float, root mean square error [µm]
#           - 'max_error': float, maximum absolute error [µm]
#           - 'three_sigma': float, 3 * standard deviation of residuals [µm]
#           - 'r_squared': float, coefficient of determination
# [ALGORITHM INTENT]:
#   - Separation of systematic calibration error (alpha, beta) from random noise (3*sigma).
#   - Fits: L_meas = alpha * L_cmd + beta + epsilon
# [TODO FOR CODEX]:
#   - Add separate bidirectional fitting for hysteresis loop / forward-backward backlash.
#   - Export results to standardized JSON and formatted Markdown report.
# ==============================================================================

def fit_linear_motion_model(
    cmd_positions: np.ndarray,
    meas_displacements: np.ndarray
) -> Dict[str, Any]:
    """Fit linear regression model and compute error statistics."""
    assert len(cmd_positions) == len(meas_displacements), "Array lengths must match"
    assert len(cmd_positions) >= 2, "At least 2 points are required for fitting"

    x = np.asarray(cmd_positions, dtype=np.float64)
    y = np.asarray(meas_displacements, dtype=np.float64)

    # 1. First-order polynomial fit (y = alpha * x + beta)
    coeffs = np.polyfit(x, y, 1)
    alpha = float(coeffs[0])
    beta = float(coeffs[1])

    # 2. Predicted values and residuals
    y_pred = alpha * x + beta
    residuals = y - y_pred

    # 3. Statistical metrics
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    max_error = float(np.max(np.abs(residuals)))
    std_residuals = float(np.std(residuals))
    three_sigma = float(3.0 * std_residuals)

    # 4. R-squared (Coefficient of Determination)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum(residuals ** 2)
    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 1.0

    return {
        "alpha": alpha,
        "beta": beta,
        "rmse": rmse,
        "max_error": max_error,
        "std_residuals": std_residuals,
        "three_sigma": three_sigma,
        "r_squared": r_squared,
        "y_pred": y_pred,
        "residuals": residuals,
    }
