"""A-dependent teacher codelets constructed analytically from B_r and predictions y_r_hat.

Codelets implemented:
- teacher_fid:       B_r^T (B_r x - y_r_hat)
- teacher_meas_smooth: B_r^T W (B_r x - y_r_hat), W from row norms of B_r
- teacher_null:      B_r^T psi(B_r x - y_r_hat), psi = soft threshold
- teacher_graph:     L(A'_r) x, where L is a precomputed Laplacian
- base graph/null/meas variants using A only are in codelets.py
"""

from __future__ import annotations

import torch


def teacher_fid(x: torch.Tensor, B: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    Bx = torch.matmul(x.view(x.shape[0], -1), B.t())
    r = Bx - y_hat
    g = torch.matmul(r, B)  # [B,n]
    return g.view_as(x)


def teacher_meas_smooth(x: torch.Tensor, B: torch.Tensor, y_hat: torch.Tensor, W_diag: torch.Tensor) -> torch.Tensor:
    Bx = torch.matmul(x.view(x.shape[0], -1), B.t())
    r = (Bx - y_hat) * W_diag.unsqueeze(0)
    g = torch.matmul(r, B)  # [B,n]
    return g.view_as(x)


def teacher_null(x: torch.Tensor, B: torch.Tensor, y_hat: torch.Tensor, tau: float = 0.01) -> torch.Tensor:
    Bx = torch.matmul(x.view(x.shape[0], -1), B.t())
    r = Bx - y_hat
    r_shrunk = torch.sign(r) * torch.clamp(torch.abs(r) - tau, min=0.0)
    g = torch.matmul(r_shrunk, B)
    return g.view_as(x)


def teacher_graph(x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    x_flat = x.view(x.shape[0], -1)
    if L.is_sparse:
        g_flat = torch.sparse.mm(L.t(), x_flat.t()).t()
    else:
        g_flat = torch.matmul(x_flat, L.t())
    return g_flat.view_as(x)


def row_norm_weights(B: torch.Tensor, gamma: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """Compute W_diag for measurement smoothing: w_i = 1 / (||row_i|| + eps)^gamma."""
    norms = torch.norm(B, dim=1)
    return 1.0 / (norms + eps) ** gamma
