"""Fixed linear sensing operator for y = A x with 64x64 RGB images (n=12288)."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

N_PIX = 3 * 64 * 64
DEFAULT_SAMPLING = 0.1
DEFAULT_A_PATH = Path("RESULTS/baselineA.npy")
FALLBACK_FIXED_RGB = Path("/home/hdsp/Desktop/codexPre3/fixed_A_rgb.npy")


def load_fixed_A(path: Path | str = DEFAULT_A_PATH, sampling: float = DEFAULT_SAMPLING) -> torch.Tensor:
    """Load sensing matrix A [m,n]; prefer provided path, then fixed_A_rgb.npy, else generate Rademacher."""
    path = Path(path)
    if not path.exists() and FALLBACK_FIXED_RGB.exists():
        path = FALLBACK_FIXED_RGB
    n = N_PIX
    m = int(max(1, sampling * n))
    if path.exists():
        A = np.load(path)
    else:
        rng = np.random.default_rng(42)
        A = rng.choice([-1.0, 1.0], size=(m, n), replace=True).astype(np.float32) / np.sqrt(m)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, A)
    return torch.from_numpy(A).float()


def Ax(x_flat: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Apply A to flattened x. x_flat: [B,n] -> [B,m]."""
    return x_flat @ A.t()


def ATz(z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Apply A^T to z. z: [B,m] -> [B,n]."""
    return z @ A


def grad_f(x_img: torch.Tensor, A: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gradient of data fidelity 0.5||Ax - y||^2."""
    B = x_img.shape[0]
    x_flat = x_img.view(B, -1)
    r = Ax(x_flat, A) - y
    g_flat = ATz(r, A)
    return g_flat.view(B, 3, 64, 64)


def estimate_spectral_norm(A: torch.Tensor, iters: int = 10) -> float:
    """Power iteration estimate of ||A||_2."""
    device = A.device
    n = A.shape[1]
    v = torch.randn(n, device=device)
    v = v / (v.norm() + 1e-8)
    for _ in range(iters):
        Av = torch.mv(A, v)
        v = torch.mv(A.t(), Av)
        v = v / (v.norm() + 1e-8)
    Av = torch.mv(A, v)
    norm = torch.norm(Av) / (torch.norm(v) + 1e-8)
    return norm.item()
