"""Inference script for PnP-COD router."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import math

from linear_ops import load_fixed_A, Ax, ATz, grad_f, estimate_spectral_norm
from denoisers_bank import make_denoiser_bank, apply_denoisers
from codelets import apply_codelet_batch, normalize_to_data, LinearTransform
from image_encoder import ImageEncoder
from router import CodeletDenoiserRouter, extract_scalar_features

LAMBDA_BINS = [0.0, 0.03, 0.1, 0.3, 1.0]


def load_image(path: Path, device: torch.device, size: int = 64) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def build_codelet_params(device: torch.device, n: int):
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device)
    q = 64
    U = torch.linalg.qr(torch.randn(n, q, device=device), mode="reduced").Q
    p = 256
    W_mat = torch.randn(p, n, device=device) / math.sqrt(n)
    W = LinearTransform(W_mat)
    return {
        "alpha_tv": 1e-3,
        "kernel_graph": kernel,
        "U": U,
        "tau_n": 0.01,
        "W": W,
        "tau_sp": 0.01,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/test")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--router", type=str, default="RESULTS/router.pt")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-images", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    A = load_fixed_A().to(device)
    denos = make_denoiser_bank(device)
    codelet_params = build_codelet_params(device, n=A.shape[1])

    ckpt = torch.load(args.router, map_location=device)
    encoder = ImageEncoder(embed_dim=64).to(device)
    router = CodeletDenoiserRouter(state_dim=5 + 64, hidden=128).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    router.load_state_dict(ckpt["router"])
    eta = ckpt.get("eta", 0.9 / (estimate_spectral_norm(A) ** 2 + 1e-8))
    encoder.eval()
    router.eval()

    paths = sorted([p for p in Path(args.img_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if len(paths) == 0:
        raise RuntimeError(f"No images found in {args.img_dir}")

    psnrs = []
    for p in paths:
        x_true = load_image(p, device, size=args.size)
        B = x_true.shape[0]
        y = Ax(x_true.reshape(B, -1), A)
        x = ATz(y, A).reshape_as(x_true)
        x_prev = x.clone()
        for k in range(args.iters):
            scalars = extract_scalar_features(x, x_prev, A, y, k, args.iters)
            img_emb = encoder(x)
            state = torch.cat([scalars, img_emb], dim=1)
            out = router(state)
            c_idx = out["logits_codelet"].argmax(dim=-1)
            l_idx = out["logits_lambda"].argmax(dim=-1)
            d_idx = out["logits_deno"].argmax(dim=-1)
            lam = torch.tensor(LAMBDA_BINS, device=device)[l_idx]
            g_data = grad_f(x, A, y)
            g_code = apply_codelet_batch(c_idx, x, A, y, codelet_params)
            g_code_used = normalize_to_data(g_code, g_data)
            step_dir = g_data + lam.view(-1, 1, 1, 1) * g_code_used
            x_prev = x
            x_pred = x - eta * step_dir
            x = apply_denoisers(x_pred, d_idx, denos).clamp(0.0, 1.0)
        mse = torch.mean((x - x_true) ** 2).item()
        psnr = 10 * math.log10(1.0 / (mse + 1e-8))
        psnrs.append(psnr)
        print(f"{p.name}: {psnr:.2f} dB")
    print(f"Average PSNR over {len(psnrs)} images: {sum(psnrs)/len(psnrs):.2f} dB")


if __name__ == "__main__":
    main()
