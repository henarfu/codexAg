"""Compare baseline PnP-PSG vs teacher02-corrected PnP-PSG on a small set of images."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import numpy as np
import torch

from linear_ops import load_fixed_A, Ax, ATz, grad_f, estimate_spectral_norm
from denoisers_bank import make_general_only
from codelets import normalize_to_data
from codeletsA.train_teacher02 import UNetTeacher


def load_image(path: Path, device: torch.device, size: int = 64) -> torch.Tensor:
    from PIL import Image
    arr = np.array(Image.open(path).convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def psnr(x: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((x - gt) ** 2).item()
    return 10 * math.log10(1.0 / (mse + 1e-8))


def build_state(x, x_prev, A, y, k, K):
    B = x.shape[0]
    x_flat = x.view(B, -1)
    r = Ax(x_flat, A) - y
    g = grad_f(x, A, y)
    res_norm = torch.norm(r, dim=1) / (torch.norm(y, dim=1) + 1e-8)
    grad_norm = torch.log(torch.norm(g.view(B, -1), dim=1) + 1e-8)
    dx = torch.norm(x_flat - x_prev.view(B, -1), dim=1) / (torch.norm(x_prev.view(B, -1), dim=1) + 1e-8)
    t = torch.full_like(res_norm, k / max(1, K - 1))
    return torch.stack([t, res_norm, grad_norm, dx], dim=1)


def run_once(img_path: Path, A: torch.Tensor, B: torch.Tensor, teacher_ckpt: Path, device: torch.device, iters: int, lam: float):
    deno = make_general_only(device)[0]
    eta = 0.9 / (estimate_spectral_norm(A) ** 2 + 1e-8)

    state_dict = torch.load(teacher_ckpt, map_location=device)
    net = UNetTeacher(m_out=B.shape[0], m_in=A.shape[0], state_dim=4).to(device)
    net.load_state_dict(state_dict["model"])
    net.eval()

    x_true = load_image(img_path, device)
    Bsz = x_true.shape[0]
    y = Ax(x_true.reshape(Bsz, -1), A)
    y0 = torch.matmul(x_true.reshape(Bsz, -1), B.t())

    # Baseline
    xb = ATz(y, A).reshape_as(x_true)
    for _ in range(iters):
        g = grad_f(xb, A, y)
        xb = deno(xb - eta * g).clamp(0, 1)
    psnr_base = psnr(xb, x_true)

    # Teacher-corrected
    xt = ATz(y, A).reshape_as(x_true)
    x_prev = xt.clone()
    for k in range(iters):
        g_data = -eta * grad_f(xt, A, y)
        state = build_state(xt, x_prev, A, y, k, iters)
        with torch.no_grad():
            y_hat = net(xt, state, y)
        # teacher fidelity codelet
        Bx = torch.matmul(xt.view(Bsz, -1), B.t())
        r = Bx - y_hat
        g_code = torch.matmul(r, B).view_as(xt)
        g_code_used = normalize_to_data(g_code, -g_data)
        x_prev = xt
        xt = deno(xt + g_data + lam * g_code_used).clamp(0, 1)
    psnr_teacher = psnr(xt, x_true)

    return psnr_base, psnr_teacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/test")
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--teacher", type=str, default="RESULTS/teacher02.pt")
    ap.add_argument("--B", type=str, default="RESULTS/B_teacher02.npy")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    A = load_fixed_A().to(device)
    B = torch.from_numpy(np.load(args.B)).to(device)
    teacher_ckpt = Path(args.teacher)

    paths = [p for p in sorted(Path(args.img_dir).iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    paths = paths[: args.max_images]
    base_scores = []
    teacher_scores = []
    for p in paths:
        pb, pt = run_once(p, A, B, teacher_ckpt, device, args.iters, args.lam)
        base_scores.append(pb)
        teacher_scores.append(pt)
        print(f"{p.name}: base {pb:.2f} dB, teacher {pt:.2f} dB")
    print(f"Average over {len(paths)} images: base {np.mean(base_scores):.2f} dB, teacher {np.mean(teacher_scores):.2f} dB")


if __name__ == "__main__":
    main()
