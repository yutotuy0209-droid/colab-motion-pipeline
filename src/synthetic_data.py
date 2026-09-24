"""Synthetic Motion Sequence Generator for Subpixel POC Verification.

Uses Fourier Shift Theorem to render mathematically exact subpixel translations
with controllable sensor noise and background illumination.
"""

from typing import Tuple, List
import numpy as np


def generate_subpixel_shifted_image(
    base_image: np.ndarray,
    shift_x: float,
    shift_y: float
) -> np.ndarray:
    """Apply exact continuous subpixel shift using Fourier Shift Theorem."""
    h, w = base_image.shape[:2]
    f = base_image.astype(np.float64)

    F = np.fft.fft2(f)

    # Frequency grids
    kx = np.fft.fftfreq(w)
    ky = np.fft.fftfreq(h)
    KX, KY = np.meshgrid(kx, ky)

    # Phase shift kernel: exp(-2pi*i * (u*dx/W + v*dy/H))
    phase_shift = np.exp(-2j * np.pi * (KX * shift_x + KY * shift_y))
    shifted_F = F * phase_shift

    shifted_f = np.real(np.fft.ifft2(shifted_F))
    return np.clip(shifted_f, 0, 255).astype(np.uint8)


def generate_grid_base_pattern(
    height: int = 512,
    width: int = 512,
    pitch: float = 40.0,
    radius: float = 6.0,
    bg_level: float = 30.0,
    dot_level: float = 220.0
) -> np.ndarray:
    """Generate reference base dot grid pattern."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    img = np.full((height, width), bg_level, dtype=np.float64)

    xs = np.arange(pitch, width - pitch, pitch)
    ys = np.arange(pitch, height - pitch, pitch)
    sigma = radius / 2.0

    for cy in ys:
        for cx in xs:
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            spot = np.exp(-d2 / (2.0 * (sigma ** 2)))
            img += (dot_level - bg_level) * spot

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_motion_sequence(
    num_frames: int = 20,
    step_dx: float = 0.20,
    step_dy: float = 0.08,
    height: int = 512,
    width: int = 512,
    noise_sigma: float = 1.0
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Generate a sequence of continuous subpixel shifted frames.

    Returns:
        frames: list of images (np.ndarray, uint8)
        true_dx: ground truth cumulative x shifts [pixel]
        true_dy: ground truth cumulative y shifts [pixel]
    """
    base = generate_grid_base_pattern(height=height, width=width)
    frames = []
    true_dx = np.zeros(num_frames, dtype=np.float64)
    true_dy = np.zeros(num_frames, dtype=np.float64)

    for i in range(num_frames):
        sx = i * step_dx
        sy = i * step_dy
        true_dx[i] = sx
        true_dy[i] = sy

        frame = generate_subpixel_shifted_image(base, sx, sy)
        if noise_sigma > 0:
            noise = np.random.normal(0, noise_sigma, frame.shape)
            frame = np.clip(frame.astype(np.float64) + noise, 0, 255).astype(np.uint8)

        frames.append(frame)

    return frames, true_dx, true_dy
