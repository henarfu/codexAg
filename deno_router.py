"""Router that only selects the denoiser (no codelets/lambda)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from linear_ops import Ax, ATz


def extract_scalar_features(x: torch.Tensor, x_prev: torch.Tensor, A: torch.Tensor, y: torch.Tensor, k: int, K: int) -> torch.Tensor:
    """Cheap scalar features for denoiser selection."""
    B = x.shape[0]
    x_flat = x.view(B, -1)
    x_prev_flat = x_prev.view(B, -1)
    t = k / max(1, K - 1)
    r = Ax(x_flat, A) - y  # [B,m]
    R = torch.norm(r, dim=1) / (torch.norm(y, dim=1) + 1e-8)
    g_flat = ATz(r, A)  # [B,n]
    g_norm = torch.log(torch.norm(g_flat, dim=1) + 1e-8)
    dx = torch.norm(x_flat - x_prev_flat, dim=1) / (torch.norm(x_prev_flat, dim=1) + 1e-8)
    grad_rms = torch.sqrt((g_flat * g_flat).mean(dim=1) + 1e-8)
    return torch.stack([torch.full_like(R, t), R, dx, g_norm, grad_rms], dim=1)


class DenoiserRouter(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 64, num_deno: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head_deno = nn.Linear(hidden, num_deno)

    def forward(self, state: torch.Tensor):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        h = self.net(state)
        return {"logits_deno": self.head_deno(h)}


def st_gumbel_onehot(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Straight-through Gumbel-softmax."""
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    y_soft = F.softmax((logits + g) / tau, dim=-1)
    idx = y_soft.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, idx, 1.0)
    return y_hard + (y_soft - y_soft.detach())
