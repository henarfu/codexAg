"""Teacher networks that predict missing measurements y_r = B_r x* (not full gradients).

Each TeacherNet takes (x_k, y, state) and outputs y_r_hat of shape [B, m_r].
This is a lightweight MLP+pool encoder; replace as needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherNet(nn.Module):
    def __init__(self, m_r: int, embed_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, m_r),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # x: [B,3,64,64], y: [B,m], state: [B,4] (e.g., t, residual norm, grad norm, dx)
        B = x.shape[0]
        h = self.img_encoder(x).view(B, -1)
        s = torch.cat([h, state], dim=1)
        return self.mlp(s)


def build_teacher_nets(m_list, device: torch.device, embed_dim: int = 64, hidden: int = 128):
    """Instantiate one TeacherNet per m_r in m_list."""
    nets = []
    for m_r in m_list:
        nets.append(TeacherNet(m_r, embed_dim=embed_dim, hidden=hidden).to(device))
    return nets
