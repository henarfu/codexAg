"""Heuristic denoiser selection driven by residual magnitude (PnP-PSG).

- Uses the fixed sensing matrix `RESULTS/baselineA.npy` (0.1x, 64x64 RGB).
- Denoiser bank: sigma20/15/10/05/03/025 plus the general model.
- Rule: higher relative residual -> higher-sigma denoiser; lower residual -> lower sigma.
- Prints per-iteration residuals and chosen denoiser for one image, plus final PSNR.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
import torch

from linear_ops import load_fixed_A, Ax, ATz, grad_f, estimate_spectral_norm
from denoisers_bank import _load_dncnn


# Residual cutoffs that promote to the next denoiser (larger index = lower sigma).
# Combined with a minimum-iteration schedule to ensure every denoiser is visited at least once.
RESIDUAL_THRESHOLDS: List[float] = [5.0, 1.0, 0.5, 0.2, 0.1, 0.05]
STAGE_ITERS: List[int] = [0, 1, 2, 3, 5, 8, 12]


def load_image(path: Path, device: torch.device, size: int = 64) -> torch.Tensor:
    """Load RGB image, resize, and return tensor [1,3,H,W] in [0,1]."""
    arr = np.array(Image.open(path).convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def build_bank(device: torch.device):
    base = Path("RESULTS")
    entries = [
        ("sigma20", 0.20, base / "denoiser_sigma20.pth"),
        ("sigma15", 0.15, base / "denoiser_sigma15.pth"),
        ("sigma10", 0.10, base / "denoiser_sigma10.pth"),
        ("sigma05", 0.05, base / "denoiser_sigma05.pth"),
        ("sigma03", 0.03, base / "denoiser_sigma03.pth"),
        ("sigma025", 0.025, base / "denoiser_sigma025.pth"),
        ("general", 0.02, base / "generaldenoiser.pth"),
    ]
    bank = []
    for name, sigma, path in entries:
        bank.append({"name": name, "sigma": sigma, "fn": _load_dncnn(path, device, sigma)})
    return bank


def pick_denoiser(rel_residual: float, name_to_idx: Dict[str, int], k: int) -> int:
    """Promote to lower-sigma denoisers as residual drops or iterations progress."""
    idx = max(i for i, it in enumerate(STAGE_ITERS) if k >= it)
    for i, thr in enumerate(RESIDUAL_THRESHOLDS, start=1):
        if rel_residual <= thr:
            idx = max(idx, i)
    return min(idx, max(name_to_idx.values()))


def psnr(x: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((x - gt) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 10 * math.log10(1.0 / mse)


def run_single(img_path: Path, A: torch.Tensor, bank, eta: float, iters: int, device: torch.device, size: int):
    x_true = load_image(img_path, device, size=size)
    B = x_true.shape[0]
    y = Ax(x_true.reshape(B, -1), A)
    x = ATz(y, A).reshape_as(x_true)
    logs = []
    name_to_idx = {b["name"]: i for i, b in enumerate(bank)}
    last_idx = 0

    with torch.no_grad():
        for k in range(iters):
            r = Ax(x.reshape(B, -1), A) - y
            rel = (torch.norm(r) / (torch.norm(y) + 1e-8)).item()
            base_idx = pick_denoiser(rel, name_to_idx, k)
            idx = max(base_idx, last_idx)  # enforce monotonic move toward lower sigma
            last_idx = idx
            g = grad_f(x, A, y)
            x = bank[idx]["fn"](x - eta * g).clamp(0.0, 1.0)
            logs.append((k, rel, bank[idx]["name"]))
    return x, logs, psnr(x, x_true)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/test")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    A = load_fixed_A().to(device)
    eta = 0.9 / (estimate_spectral_norm(A) ** 2 + 1e-8)
    bank = build_bank(device)

    paths = sorted([p for p in Path(args.img_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if len(paths) == 0:
        raise RuntimeError(f"No images found in {args.img_dir}")

    x_rec, logs, score = run_single(paths[0], A, bank, eta, args.iters, device, size=args.size)
    print(f"Image: {paths[0].name}")
    print(f"PSNR after {args.iters} iters: {score:.2f} dB")
    print("Residual (rel) -> denoiser per iteration:")
    for k, rel, name in logs:
        print(f"{k:03d}: rel_res={rel:.6f} -> {name}")


if __name__ == "__main__":
    main()
