"""Online fine-tuning demo: start from general denoiser, update it on-the-fly with x_k."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim

from linear_ops import load_fixed_A, Ax, ATz, grad_f, estimate_spectral_norm

# DeepInv DnCNN lives outside this repo.
import sys

sys.path.append("/home/hdsp/Documents/Henry/pnp")
import deepinv as dinv  # type: ignore


def load_image(path: Path, device: torch.device, size: int = 64) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def load_general_model(device: torch.device, sigma: float = 0.02, ckpt: Path = Path("RESULTS/generaldenoiser.pth")):
    model = dinv.models.DnCNN(in_channels=3, out_channels=3, pretrained="download_lipschitz").to(device)
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded general denoiser from {ckpt}")
    sigma_tensor = torch.tensor([[sigma]], device=device, dtype=torch.float32)
    return model, sigma_tensor


def psnr(x: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((x - gt) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 10 * math.log10(1.0 / mse)


def main():
    p = argparse.ArgumentParser(description="Online fine-tune a probe denoiser during PnP iterations.")
    p.add_argument("--img", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/test/Places365_val_00000001.jpg")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    x_true = load_image(Path(args.img), device, size=args.size)
    A = load_fixed_A().to(device)
    eta = 0.9 / (estimate_spectral_norm(A) ** 2 + 1e-8)

    model, sigma_tensor = load_general_model(device, sigma=args.sigma)
    model.train()  # enable gradients
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    y = Ax(x_true.reshape(1, -1), A)
    x = ATz(y, A).reshape_as(x_true)
    psnrs = []

    for k in range(args.iters):
        g = grad_f(x, A, y)
        x_pred = x - eta * g

        # One online fine-tuning step using x_true as supervision.
        opt.zero_grad()
        out = model(x_pred, sigma=sigma_tensor)
        loss = loss_fn(out, x_true)
        loss.backward()
        opt.step()

        with torch.no_grad():
            x = out.clamp(0.0, 1.0)
            psnrs.append(psnr(x, x_true))
        if (k + 1) % 10 == 0 or k == 0:
            print(f"Iter {k+1}/{args.iters}: loss={loss.item():.4e}, PSNR={psnrs[-1]:.2f} dB")

    print(f"Final PSNR after {args.iters} iters: {psnrs[-1]:.2f} dB")


if __name__ == "__main__":
    main()
