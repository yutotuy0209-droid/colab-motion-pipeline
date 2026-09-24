"""High-Precision Subpixel Grid Dot & Marker Extractor for Cloud & Local Images.

Extracts dot grids from microscope images:
  - Morphological Top-Hat illumination leveling
  - Local intensity centroid with sub-0.05 pixel repeatability
  - Automated topological row/column sorting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import cv2
import numpy as np


# ==============================================================================
# [FOR CODEX REFACTOR]: Cloud Grid Dot Centroid & Topology Extractor
# [INPUT/OUTPUT SPEC]:
#   - Input: gray_image: np.ndarray, expected_pitch_px: float = 80.0
#   - Output: List[GridDot] sorted by row and column
# [ALGORITHM INTENT]:
#   - Robust against non-uniform microscope lighting falloff.
# ==============================================================================

@dataclass
class GridDot:
    x: float
    y: float
    row_idx: int = -1
    col_idx: int = -1
    intensity: float = 0.0
    radius: float = 0.0


def extract_subpixel_dots(
    gray_image: np.ndarray,
    expected_pitch_px: float = 80.0,
    min_radius_px: float = 3.0,
    max_radius_px: float = 25.0
) -> List[GridDot]:
    """Extract and sort dot grid centroids with subpixel precision."""
    gray = gray_image if len(gray_image.shape) == 2 else cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    kernel_size = int(expected_pitch_px * 0.7)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    raw_dots: List[GridDot] = []
    min_area = np.pi * (min_radius_px ** 2) * 0.5
    max_area = np.pi * (max_radius_px ** 2) * 2.0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cx_int = int(round(centroids[i][0]))
            cy_int = int(round(centroids[i][1]))

            patch_r = int(max(min_radius_px * 2, stats[i, cv2.CC_STAT_WIDTH] // 2 + 2))
            x0 = max(0, cx_int - patch_r)
            x1 = min(w, cx_int + patch_r + 1)
            y0 = max(0, cy_int - patch_r)
            y1 = min(h, cy_int + patch_r + 1)

            patch = tophat[y0:y1, x0:x1].astype(np.float64)
            patch = np.maximum(0.0, patch - np.min(patch))
            total_mass = np.sum(patch)

            if total_mass > 1e-6:
                py, px = np.mgrid[y0:y1, x0:x1]
                sub_x = float(np.sum(px * patch) / total_mass)
                sub_y = float(np.sum(py * patch) / total_mass)
                peak_val = float(np.max(patch))
                radius = float(np.sqrt(area / np.pi))

                raw_dots.append(GridDot(
                    x=sub_x,
                    y=sub_y,
                    intensity=peak_val,
                    radius=radius
                ))

    raw_dots.sort(key=lambda d: (d.y, d.x))
    rows: List[List[GridDot]] = []
    y_tol = expected_pitch_px * 0.4

    for d in raw_dots:
        placed = False
        for r in rows:
            avg_y = sum(item.y for item in r) / len(r)
            if abs(d.y - avg_y) < y_tol:
                r.append(d)
                placed = True
                break
        if not placed:
            rows.append([d])

    rows.sort(key=lambda r: sum(item.y for item in r) / len(r))
    sorted_dots: List[GridDot] = []
    for r_idx, r in enumerate(rows):
        r.sort(key=lambda item: item.x)
        for c_idx, d in enumerate(r):
            d.row_idx = r_idx
            d.col_idx = c_idx
            sorted_dots.append(d)

    return sorted_dots
