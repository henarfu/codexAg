"""Router agent skeleton for choosing denoiser, codelet, teacher, and lambda.

Action heads:
- denoiser_idx ∈ {0..4}
- codelet_type ∈ {NONE, BASE_NULL_A, BASE_MEAS_A, BASE_GRAPH_A, TEACHER_FID, TEACHER_MS, TEACHER_NULL, TEACHER_GRAPH}
- teacher_idx ∈ {-1,0,1,2,3} (encoded as 0..4 with 0 meaning base/-1)
- lambda_idx ∈ {0..4} for λ ∈ {0, 0.03, 0.1, 0.3, 1.0}
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LAMBDA_BINS = [0.0, 0.03, 0.1, 0.3, 1.0]
CODELET_TYPES = [
    "NONE",
    "BASE_NULL_A",
    "BASE_MEAS_A",
    "BASE_GRAPH_A",
    "TEACHER_FID",
    "TEACHER_MS",
    "TEACHER_NULL",
    "TEACHER_GRAPH",
]


class AgentRouter(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128, num_deno: int = 5, num_teachers: int = 4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head_deno = nn.Linear(hidden, num_deno)
        self.head_codelet = nn.Linear(hidden, len(CODELET_TYPES))
        # teacher_idx: base(-1) + num_teachers -> encoded as 0..num_teachers (0 means base)
        self.head_teacher = nn.Linear(hidden, num_teachers + 1)
        self.head_lambda = nn.Linear(hidden, len(LAMBDA_BINS))

    def forward(self, state: torch.Tensor):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        h = self.trunk(state)
        return {
            "logits_deno": self.head_deno(h),
            "logits_codelet": self.head_codelet(h),
            "logits_teacher": self.head_teacher(h),
            "logits_lambda": self.head_lambda(h),
        }


def st_gumbel_onehot(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    y_soft = F.softmax((logits + g) / tau, dim=-1)
    idx = y_soft.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, idx, 1.0)
    return y_hard + (y_soft - y_soft.detach())
