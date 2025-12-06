"""Inference using denoiser-only router (or uniform schedule)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from linear_ops import load_fixed_A, Ax, ATz, grad_f, estimate_spectral_norm
from denoisers_bank import make_denoiser_bank, apply_denoisers
from image_encoder import ImageEncoder
from deno_router import DenoiserRouter, extract_scalar_features


def load_image(path: Path, device: torch.device, size: int = 64) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/test")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--router", type=str, default="RESULTS/deno_router.pt")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--use-uniform", action="store_true", help="Ignore router and use fixed denoiser schedule.")
    parser.add_argument("--single-general", action="store_true", help="Use only the general denoiser as baseline.")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    A = load_fixed_A().to(device)
    denos = make_denoiser_bank(device, use_targeted=True)
    if args.single_general:
        denos = denos[:1]

    if not args.use_uniform:
        ckpt = torch.load(args.router, map_location=device)
        encoder = ImageEncoder(embed_dim=64).to(device)
        router = DenoiserRouter(state_dim=5 + 64, hidden=128, num_deno=len(denos)).to(device)
        encoder.load_state_dict(ckpt["encoder"])
        router.load_state_dict(ckpt["router"])
        eta = ckpt.get("eta", 0.9 / (estimate_spectral_norm(A) ** 2 + 1e-8))
        encoder.eval()
        router.eval()
    else:
        eta = 0.9 / (estimate_spectral_norm(A) ** 2 + 1e-8)
        encoder = None
        router = None

    paths = sorted([p for p in Path(args.img_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if args.max_images is not None:
        paths = paths[: args.max_images]
    if len(paths) == 0:
        raise RuntimeError(f"No images found in {args.img_dir}")

    psnrs = []
    block = max(1, args.iters // len(denos))
    for p in paths:
        x_true = load_image(p, device, size=args.size)
        B = x_true.shape[0]
        y = Ax(x_true.reshape(B, -1), A)
        x = ATz(y, A).reshape_as(x_true)
        x_prev = x.clone()
        for k in range(args.iters):
            g_data = grad_f(x, A, y)
            x_prev = x
            x_pred = x - eta * g_data
            if args.use_uniform:
                idx = min(k // block, len(denos) - 1)
                d_idx = torch.tensor([idx] * B, device=device, dtype=torch.long)
            else:
                scalars = extract_scalar_features(x, x_prev, A, y, k, args.iters)
                img_emb = encoder(x)
                state = torch.cat([scalars, img_emb], dim=1)
                out = router(state)
                d_idx = out["logits_deno"].argmax(dim=-1)
            x = apply_denoisers(x_pred, d_idx, denos).clamp(0.0, 1.0)
        mse = torch.mean((x - x_true) ** 2).item()
        psnr = 10 * math.log10(1.0 / (mse + 1e-8))
        psnrs.append(psnr)
        print(f"{p.name}: {psnr:.2f} dB")
    print(f"Average PSNR over {len(psnrs)} images: {sum(psnrs)/len(psnrs):.2f} dB")


if __name__ == "__main__":
    main()
