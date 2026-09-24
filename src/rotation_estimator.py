"""Geometric QR-Decomposition Rotation Center & Runout Estimator.

Solves circular arc trajectory via Householder QR least squares:
  - Non-iterative global optimum circle center (Cx, Cy) & radius R
  - Peak-to-valley eccentricity runout & RMS circularity error
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ==============================================================================
# [FOR CODEX REFACTOR]: QR Decomposition Circle Fitting & Coaxiality Estimator
# [INPUT/OUTPUT SPEC]:
#   - Input: points: np.ndarray, shape (N, 2)
#   - Output: RotationCenterResult with center_x, center_y, radius, runout
# ==============================================================================

@dataclass
class RotationCenterResult:
    center_x: float
    center_y: float
    radius: float
    radial_residuals: np.ndarray
    peak_to_valley_runout: float
    circularity_rms: float
    angular_positions_rad: np.ndarray


def estimate_rotation_center_qr(points: np.ndarray) -> RotationCenterResult:
    """Fit circle using linear least-squares with QR decomposition."""
    pts = np.asarray(points, dtype=np.float64)
    assert pts.ndim == 2 and pts.shape[1] == 2, "Points array must have shape (N, 2)"
    N = pts.shape[0]
    assert N >= 3, "At least 3 points are required to define a circle"

    x = pts[:, 0]
    y = pts[:, 1]

    A = np.column_stack([2.0 * x, 2.0 * y, np.ones(N, dtype=np.float64)])
    b = x**2 + y**2

    Q, R = np.linalg.qr(A)
    p = np.linalg.solve(R[:3, :3], np.dot(Q.T, b)[:3])

    cx = float(p[0])
    cy = float(p[1])
    c_const = float(p[2])

    radius_sq = c_const + cx**2 + cy**2
    radius = float(np.sqrt(max(1e-12, radius_sq)))

    actual_radii = np.sqrt((x - cx)**2 + (y - cy)**2)
    residuals = actual_radii - radius

    peak_to_valley = float(np.max(residuals) - np.min(residuals))
    circularity_rms = float(np.sqrt(np.mean(residuals**2)))
    angles = np.arctan2(y - cy, x - cx)

    return RotationCenterResult(
        center_x=cx,
        center_y=cy,
        radius=radius,
        radial_residuals=residuals,
        peak_to_valley_runout=peak_to_valley,
        circularity_rms=circularity_rms,
        angular_positions_rad=angles
    )
