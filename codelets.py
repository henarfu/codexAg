"""Codelets for PnP-COD agent (small regularizer directions without denoisers)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def codelet_none(x: torch.Tensor, _A, _y) -> torch.Tensor:
    return torch.zeros_like(x)


def codelet_tv(x: torch.Tensor, alpha_tv: float) -> torch.Tensor:
    """Isotropic TV gradient (smoothed with alpha_tv)."""
    dx = torch.diff(x, dim=3, append=x[:, :, :, -1:].clone())
    dy = torch.diff(x, dim=2, append=x[:, :, -1:, :].clone())
    mag = torch.sqrt(dx * dx + dy * dy + alpha_tv * alpha_tv)
    dx_norm = dx / mag
    dy_norm = dy / mag
    dx_back = torch.diff(dx_norm, dim=3, prepend=dx_norm[:, :, :, :1].clone())
    dy_back = torch.diff(dy_norm, dim=2, prepend=dy_norm[:, :, :1, :].clone())
    return dx_back + dy_back


def codelet_graph(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Graph Laplacian direction via depthwise conv (approx Laplacian)."""
    c = x.shape[1]
    weight = kernel.to(x.device).expand(c, 1, *kernel.shape)
    return F.conv2d(x, weight, padding=kernel.shape[-1] // 2, groups=c)


def codelet_nullspace(x: torch.Tensor, U: torch.Tensor, tau_n: float) -> torch.Tensor:
    """Nullspace shrinkage in a low-rank basis U."""
    B = x.shape[0]
    x_flat = x.view(B, -1)
    coeff = torch.matmul(x_flat, U)  # [B,q]
    coeff_shrunk = torch.sign(coeff) * torch.clamp(torch.abs(coeff) - tau_n, min=0.0)
    diff = coeff - coeff_shrunk
    out_flat = torch.matmul(diff, U.t())
    return out_flat.view_as(x)


class LinearTransform:
    """Simple linear transform with explicit forward/inverse mats on flattened x."""

    def __init__(self, W: torch.Tensor):
        # W: [p,n]
        self.W = W
        self.WT = W.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        xf = x.view(B, -1)
        return torch.matmul(xf, self.W.t())

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        B = u.shape[0]
        out_flat = torch.matmul(u, self.W)
        return out_flat.view(B, 3, 64, 64)


def codelet_sparseW(x: torch.Tensor, W, tau_sp: float) -> torch.Tensor:
    """Sparse prior in transform domain W (must provide forward/inverse)."""
    u = W.forward(x)
    u_soft = torch.sign(u) * torch.clamp(torch.abs(u) - tau_sp, min=0.0)
    diff = u_soft - u
    return W.inverse(diff)


def codelet_null_A(x: torch.Tensor, A: torch.Tensor, _y: torch.Tensor, params: dict) -> torch.Tensor:
    """A-dependent nullspace shrinkage using a basis U built from the nullspace of A."""
    B = x.shape[0]
    x_flat = x.view(B, -1)  # [B, n]
    U = params["U"]  # [n, q]
    tau_n = params.get("tau_n", 0.01)
    c = torch.matmul(x_flat, U)  # [B, q]
    c_shrunk = torch.sign(c) * torch.clamp(torch.abs(c) - tau_n, min=0.0)
    diff = c - c_shrunk
    g_flat = torch.matmul(diff, U.t())  # [B, n]
    return g_flat.view_as(x)


def codelet_meas_smooth(x: torch.Tensor, A: torch.Tensor, y: torch.Tensor, params: dict) -> torch.Tensor:
    """Smooth residual in measurement space then backproject with A^T."""
    B = x.shape[0]
    x_flat = x.view(B, -1)  # [B, n]
    r = torch.matmul(x_flat, A.t()) - y  # [B, m]
    W_diag = params["W_diag"]  # [m]
    r_smooth = r * W_diag.unsqueeze(0)  # [B, m]
    g_flat = torch.matmul(r_smooth, A)  # [B, n]
    return g_flat.view_as(x)


def codelet_graph_A(x: torch.Tensor, A: torch.Tensor, _y: torch.Tensor, params: dict) -> torch.Tensor:
    """Apply an A-dependent graph Laplacian L(A) to x (L is sparse or dense)."""
    B = x.shape[0]
    x_flat = x.view(B, -1)  # [B, n]
    L = params["L"]  # [n, n] sparse or dense
    if L.is_sparse:
        g_flat = torch.sparse.mm(L.t(), x_flat.t()).t()  # [B, n]
    else:
        g_flat = torch.matmul(x_flat, L.t())  # [B, n]
    return g_flat.view_as(x)


def normalize_to_data(g_code: torch.Tensor, g_data: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize codelet direction RMS to match data grad RMS."""
    rms_code = torch.sqrt((g_code * g_code).mean(dim=[1, 2, 3], keepdim=True) + eps)
    rms_data = torch.sqrt((g_data * g_data).mean(dim=[1, 2, 3], keepdim=True) + eps)
    return g_code / rms_code * rms_data


def apply_codelet_batch(idx: torch.Tensor, x: torch.Tensor, A: torch.Tensor, y: torch.Tensor, params: dict) -> torch.Tensor:
    """Dispatch per-sample codelet."""
    outs = []
    for i in range(x.shape[0]):
        c = int(idx[i].item())
        if c == 0:
            outs.append(codelet_none(x[i:i+1], A, y[i:i+1]))
        elif c == 1:
            outs.append(codelet_tv(x[i:i+1], params["alpha_tv"]))
        elif c == 2:
            outs.append(codelet_graph(x[i:i+1], params["kernel_graph"]))
        elif c == 3:
            outs.append(codelet_nullspace(x[i:i+1], params["U"], params["tau_n"]))
        elif c == 4:
            outs.append(codelet_sparseW(x[i:i+1], params["W"], params["tau_sp"]))
        else:
            outs.append(codelet_none(x[i:i+1], A, y[i:i+1]))
    return torch.cat(outs, dim=0)


CODELET_FUNCS = {
    0: lambda x, A, y, p: codelet_none(x, A, y),
    1: lambda x, A, y, p: codelet_tv(x, p["alpha_tv"]),
    2: lambda x, A, y, p: codelet_graph(x, p["kernel_graph"]),
    3: lambda x, A, y, p: codelet_nullspace(x, p["U"], p["tau_n"]),
    4: lambda x, A, y, p: codelet_sparseW(x, p["W"], p["tau_sp"]),
    # Named entries for A-dependent codelets
    "NULL_A": lambda x, A, y, p: codelet_null_A(x, A, y, p),
    "MEAS_SMOOTH_A": lambda x, A, y, p: codelet_meas_smooth(x, A, y, p),
    "GRAPH_A": lambda x, A, y, p: codelet_graph_A(x, A, y, p),
}
