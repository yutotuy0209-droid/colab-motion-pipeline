"""Phase-Only Correlation (POC) Core Engine for GPU/CPU.

Supports subpixel 2D translation estimation between consecutive image frames.
Optimized for Google Colab GPU (PyTorch) with automatic fallback to CPU (NumPy/SciPy).
"""

from typing import Tuple, Optional
import numpy as np

try:
    import torch
    import torch.fft
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==============================================================================
# [FOR CODEX REFACTOR]: GPU/CPU Dual-Backend 2D Phase-Only Correlation Engine
# [INPUT/OUTPUT SPEC]:
#   - Input:
#       img1: np.ndarray or torch.Tensor, shape (H, W), dtype uint8 or float32 (reference frame)
#       img2: np.ndarray or torch.Tensor, shape (H, W), dtype uint8 or float32 (target frame)
#       subpixel: bool, whether to apply parabolic peak refinement (default True)
#       eps: float, numerical stability epsilon to prevent division by zero (default 1e-12)
#   - Output:
#       dx: float, subpixel translation in X direction [pixel]
#       dy: float, subpixel translation in Y direction [pixel]
#       peak_val: float, normalized correlation peak value (0.0 to 1.0)
# [ALGORITHM INTENT]:
#   - POC removes amplitude variation and isolates spatial phase differences:
#       cross_power = conj(F1) * F2
#       R(u, v) = cross_power / (|cross_power| + eps)
#       correlation_surface = real(ifft2(R))
#   - Discrete peak (px, py) is extracted directly from circular cross-correlation.
#   - 1D parabolic interpolation along X and Y axes refines subpixel offset (|delta| <= 0.5 px).
#   - Wrapping boundary is handled via standard modulo arithmetic.
# [TODO FOR CODEX]:
#   - Implement full batch-dimension processing (B, H, W) for PyTorch tensor input.
#   - Add CUDA stream synchronization if integrating into real-time C++ DLL bindings.
#   - Port peak refinement kernel to Triton or custom CUDA for microsecond throughput.
# ==============================================================================

def compute_poc_numpy(
    img1: np.ndarray,
    img2: np.ndarray,
    subpixel: bool = True,
    eps: float = 1e-12
) -> Tuple[float, float, float]:
    """Compute subpixel displacement using NumPy on CPU."""
    assert img1.shape == img2.shape, "Images must have identical dimensions"
    height, width = img1.shape[:2]

    f1 = img1.astype(np.float32)
    f2 = img2.astype(np.float32)

    # 1. 2D FFT
    F1 = np.fft.fft2(f1)
    F2 = np.fft.fft2(f2)

    # 2. Normalized cross-phase spectrum
    cross_power = np.conj(F1) * F2
    magnitude = np.abs(cross_power)
    norm = np.where(magnitude > eps, cross_power / (magnitude + eps), 0.0)

    # 3. Correlation surface via inverse 2D FFT
    surface = np.real(np.fft.ifft2(norm))

    # 4. Discrete peak search
    peak_idx = int(np.argmax(surface))
    peak_y = peak_idx // width
    peak_x = peak_idx % width
    peak_val = float(surface[peak_y, peak_x])

    # 5. 1D Parabolic subpixel refinement
    dx_sub, dy_sub = 0.0, 0.0
    if subpixel:
        # X direction
        xm = (peak_x - 1) % width
        xp = (peak_x + 1) % width
        c_x0 = float(surface[peak_y, peak_x])
        c_xm = float(surface[peak_y, xm])
        c_xp = float(surface[peak_y, xp])
        denom_x = 2.0 * (c_xm - 2.0 * c_x0 + c_xp)
        if abs(denom_x) > 1e-12:
            dx_sub = float((c_xm - c_xp) / denom_x)
            dx_sub = max(-0.5, min(0.5, dx_sub))

        # Y direction
        ym = (peak_y - 1) % height
        yp = (peak_y + 1) % height
        c_y0 = float(surface[peak_y, peak_x])
        c_ym = float(surface[ym, peak_x])
        c_yp = float(surface[yp, peak_x])
        denom_y = 2.0 * (c_ym - 2.0 * c_y0 + c_yp)
        if abs(denom_y) > 1e-12:
            dy_sub = float((c_ym - c_yp) / denom_y)
            dy_sub = max(-0.5, min(0.5, dy_sub))

    refined_x = float(peak_x) + dx_sub
    refined_y = float(peak_y) + dy_sub

    # Periodic wrapping handling
    dx = float(refined_x if peak_x <= width // 2 else refined_x - width)
    dy = float(refined_y if peak_y <= height // 2 else refined_y - height)

    return dx, dy, peak_val


def compute_poc_torch(
    img1: "torch.Tensor",
    img2: "torch.Tensor",
    subpixel: bool = True,
    eps: float = 1e-12
) -> Tuple[float, float, float]:
    """Compute subpixel displacement using PyTorch on GPU/CPU."""
    assert HAS_TORCH, "PyTorch is required for compute_poc_torch"
    assert img1.shape == img2.shape, "Tensors must have identical dimensions"

    height, width = img1.shape[-2], img1.shape[-1]
    f1 = img1.to(torch.float32)
    f2 = img2.to(torch.float32)

    F1 = torch.fft.fft2(f1)
    F2 = torch.fft.fft2(f2)

    cross_power = torch.conj(F1) * F2
    magnitude = torch.abs(cross_power)
    norm = torch.where(magnitude > eps, cross_power / (magnitude + eps), torch.zeros_like(cross_power))

    surface = torch.real(torch.fft.ifft2(norm))

    peak_idx = int(torch.argmax(surface).item())
    peak_y = peak_idx // width
    peak_x = peak_idx % width
    peak_val = float(surface[peak_y, peak_x].item())

    dx_sub, dy_sub = 0.0, 0.0
    if subpixel:
        xm = (peak_x - 1) % width
        xp = (peak_x + 1) % width
        c_x0 = float(surface[peak_y, peak_x].item())
        c_xm = float(surface[peak_y, xm].item())
        c_xp = float(surface[peak_y, xp].item())
        denom_x = 2.0 * (c_xm - 2.0 * c_x0 + c_xp)
        if abs(denom_x) > 1e-12:
            dx_sub = float((c_xm - c_xp) / denom_x)
            dx_sub = max(-0.5, min(0.5, dx_sub))

        ym = (peak_y - 1) % height
        yp = (peak_y + 1) % height
        c_y0 = float(surface[peak_y, peak_x].item())
        c_ym = float(surface[ym, peak_x].item())
        c_yp = float(surface[yp, peak_x].item())
        denom_y = 2.0 * (c_ym - 2.0 * c_y0 + c_yp)
        if abs(denom_y) > 1e-12:
            dy_sub = float((c_ym - c_yp) / denom_y)
            dy_sub = max(-0.5, min(0.5, dy_sub))

    refined_x = float(peak_x) + dx_sub
    refined_y = float(peak_y) + dy_sub

    dx = float(refined_x if peak_x <= width // 2 else refined_x - width)
    dy = float(refined_y if peak_y <= height // 2 else refined_y - height)

    return dx, dy, peak_val


def compute_displacement(
    img1: np.ndarray,
    img2: np.ndarray,
    use_gpu: bool = True,
    subpixel: bool = True
) -> Tuple[float, float, float]:
    """Universal displacement estimator automatically selecting GPU or CPU."""
    if use_gpu and HAS_TORCH and torch.cuda.is_available():
        t1 = torch.from_numpy(img1).cuda()
        t2 = torch.from_numpy(img2).cuda()
        return compute_poc_torch(t1, t2, subpixel=subpixel)
    else:
        return compute_poc_numpy(img1, img2, subpixel=subpixel)
