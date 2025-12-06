"""Small CNN encoder that embeds x_k for the router."""

from __future__ import annotations

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, embed_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return h.mean(dim=[2, 3])
