"""Load a bank of pretrained denoisers for PnP-COD."""

from __future__ import annotations

from pathlib import Path
from typing import List, Callable

import torch
import sys

sys.path.append("/home/hdsp/Documents/Henry/pnp")
import deepinv as dinv  # type: ignore


def _load_dncnn(path: Path | None, device: torch.device, sigma: float) -> Callable[[torch.Tensor], torch.Tensor]:
    model = dinv.models.DnCNN(in_channels=3, out_channels=3, pretrained="download_lipschitz").to(device).eval()
    if path is not None and path.exists():
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded denoiser from {path}")
    sigma_tensor = torch.tensor([[sigma]], device=device, dtype=torch.float32)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return model(x, sigma=sigma_tensor)

    return fn


def make_general_only(device: torch.device) -> List[Callable[[torch.Tensor], torch.Tensor]]:
    """Return a bank with only the general denoiser (baseline restriction)."""
    base = Path("RESULTS/generaldenoiser.pth")
    return [_load_dncnn(base, device, 0.02)]


def make_denoiser_bank(device: torch.device, use_targeted: bool = True) -> List[Callable[[torch.Tensor], torch.Tensor]]:
    """Return list of denoisers in high→low sigma order. If use_targeted, use the sigma-targeted set."""
    base = Path("RESULTS")
    if use_targeted:
        paths = [
            base / "denoiser_sigma20.pth",
            base / "denoiser_sigma15.pth",
            base / "denoiser_sigma10.pth",
            base / "denoiser_sigma05.pth",
            base / "denoiser_sigma03.pth",
            base / "denoiser_sigma025.pth",
            base / "generaldenoiser.pth",
        ]
        sigmas = [0.20, 0.15, 0.10, 0.05, 0.03, 0.025, 0.02]
    else:
        paths = [
            base / "generaldenoiser.pth",
            base / "denoiser5.pth",
            base / "denoiser4.pth",
            base / "denoiser3.pth",
            base / "denoiser2.pth",
            base / "denoiser1.pth",
        ]
        sigmas = [0.02, 0.13, 0.08, 0.045, 0.02, 0.005]
    denos = []
    for p, s in zip(paths, sigmas):
        denos.append(_load_dncnn(p, device, s))
    return denos


def apply_denoisers(x_pred: torch.Tensor, d_idx: torch.Tensor, denos) -> torch.Tensor:
    """Apply selected denoiser per sample."""
    out = []
    for i in range(x_pred.shape[0]):
        out.append(denos[d_idx[i].item()](x_pred[i : i + 1]))
    return torch.cat(out, dim=0)
