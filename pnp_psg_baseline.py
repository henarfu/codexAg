"""PnP-PSG baseline (0.1 compression, 64x64) using the fixed saved matrix A (AA.npy).

- Forward model: y = A x with A loaded from RESULTS/AA.npy (1228 x 12288).
- Algorithm: PnP proximal gradient (data gradient step + pretrained DnCNN).
- Denoiser: generic DeepInv DnCNN (downloaded weights), optionally fine-tuned checkpoint.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Tuple, List, Callable

import numpy as np
from PIL import Image
import torch

sys.path.append("/home/hdsp/Documents/Henry/pnp")
import deepinv as dinv  # type: ignore

from linear_ops import load_fixed_A, Ax, ATz, estimate_spectral_norm

def load_image(path: Path, device: torch.device, size: int = 64) -> torch.Tensor:
    """Load RGB image, resize, and return tensor [1,3,H,W] in [0,1]."""
    arr = np.array(Image.open(path).convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def list_images(folder: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in exts:
            yield p


def make_denoiser(device: torch.device, sigma: float, ckpt_path: Path | None = None):
    model = dinv.models.DnCNN(in_channels=3, out_channels=3, pretrained="download_lipschitz").to(device).eval()
    sigma_tensor = torch.tensor([[sigma]], device=device, dtype=torch.float32)
    if ckpt_path is not None and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded finetuned denoiser from {ckpt_path}")

    def denoise_fn(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(x, sigma=sigma_tensor)

    return denoise_fn


def make_denoiser_bank(device: torch.device, base_dir: Path, sigmas: List[float]) -> List[Callable[[torch.Tensor], torch.Tensor]]:
    paths = [base_dir / "generaldenoiser.pth"]
    for i in range(1, 6):
        paths.append(base_dir / f"denoiser{i}.pth")
    denos = []
    for p, s in zip(paths, sigmas):
        denos.append(make_denoiser(device, s, p))
    return denos


def psnr(x: torch.Tensor, x_gt: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((x - x_gt) ** 2).item()
    if mse <= eps:
        return 99.0
    return 10 * math.log10(1.0 / mse)


def pnp_psg_matrix(x_true: torch.Tensor, A: torch.Tensor, denoise_fn, eta: float, iters: int) -> Tuple[torch.Tensor, float]:
    """PnP-PSG: gradient step on data fidelity, then denoise, using explicit A."""
    B = x_true.shape[0]
    y = Ax(x_true.reshape(B, -1), A)
    x = ATz(y, A).reshape_as(x_true)
    with torch.no_grad():
        for _ in range(iters):
            r = Ax(x.reshape(B, -1), A) - y
            g = ATz(r, A).reshape_as(x)
            x = denoise_fn(x - eta * g).clamp(0.0, 1.0)
    return x, psnr(x, x_true)


def main():
    parser = argparse.ArgumentParser(description="PnP-PSG single-pixel baseline (0.1x, 64x64).")
    parser.add_argument("--img-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/test", help="Folder with input images.")
    parser.add_argument("--size", type=int, default=64, help="Resize images to this square size.")
    parser.add_argument("--iters", type=int, default=4000, help="Number of PnP-PSG iterations.")
    parser.add_argument("--sigma", type=float, default=0.02, help="Sigma passed to pretrained DnCNN.")
    parser.add_argument("--eta", type=float, default=1.0, help="Step size (fixed).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device.")
    parser.add_argument("--denoiser-path", type=str, default="RESULTS/generaldenoiser.pth", help="Optional finetuned denoiser checkpoint.")
    parser.add_argument("--use-denoiser-bank", action="store_true", help="Use general + denoiser1-5 with fixed schedule across iterations.")
    parser.add_argument("--bank-sigmas", type=str, default="0.02,0.005,0.02,0.045,0.08,0.13", help="Comma-separated sigma for bank (general,1..5).")
    parser.add_argument("--save-dir", type=str, default=None, help="Optional output folder for reconstructions.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images.")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    img_dir = Path(args.img_dir)
    img_paths = list(list_images(img_dir))
    if args.max_images is not None:
        img_paths = img_paths[: args.max_images]
    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in {img_dir}")

    # Load fixed matrix A (shape [m,n] with m=0.1*n=3684).
    A = load_fixed_A().to(device)
    norm = estimate_spectral_norm(A)
    eta = args.eta if args.eta is not None else 0.9 / (norm**2 + 1e-8)

    if args.use_denoiser_bank:
        base_dir = Path("RESULTS")
        sigma_bank = [float(s) for s in args.bank_sigmas.split(",")]
        if len(sigma_bank) != 6:
            raise ValueError("bank-sigmas must have 6 comma-separated values (general + 5 denoisers).")
        deno_paths = [base_dir / "generaldenoiser.pth"]
        for i in range(1, 6):
            deno_paths.append(base_dir / f"denoiser{i}.pth")
        # general first, then remaining denoisers sorted by sigma descending
        general_fn = make_denoiser(device, sigma_bank[0], deno_paths[0])
        rest = []
        for p, s in zip(deno_paths[1:], sigma_bank[1:]):
            rest.append((s, make_denoiser(device, s, p)))
        rest_sorted = [fn for _, fn in sorted(rest, key=lambda t: t[0], reverse=True)]
        deno_bank = [general_fn] + rest_sorted
        deno_desc = "bank(general then high→low sigma)"
        denoise_fn = None
    else:
        ckpt_path = Path(args.denoiser_path) if args.denoiser_path else None
        denoise_fn = make_denoiser(device, args.sigma, ckpt_path)
        deno_desc = ckpt_path.name if ckpt_path is not None else "vanilla"

    if args.save_dir:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    scores = []
    for p in img_paths:
        x_true = load_image(p, device, size=args.size)
        y_true = Ax(x_true.reshape(1, -1), A)
        x0 = ATz(y_true, A).reshape_as(x_true)
        if args.use_denoiser_bank:
            block = max(1, math.ceil(args.iters / len(deno_bank)))
            x = x0.clone()
            with torch.no_grad():
                for k in range(args.iters):
                    r = Ax(x.reshape(1, -1), A) - y_true
                    g = ATz(r, A).reshape_as(x)
                    idx = min(k // block, len(deno_bank) - 1)
                    x = deno_bank[idx](x - eta * g).clamp(0.0, 1.0)
            x_rec = x
            score = psnr(x_rec, x_true)
        else:
            x_rec, score = pnp_psg_matrix(x_true, A, denoise_fn, eta=eta, iters=args.iters)
        scores.append(score)
        extra = f"bank" if args.use_denoiser_bank else f"sigma={args.sigma}"
        print(f"{p.name}: {score:.2f} dB (eta={eta:.4e}, {extra})")
        if out_dir is not None:
            arr = (x_rec.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(out_dir / p.name)

    mean_psnr = sum(scores) / len(scores)
    print(f"Average PSNR over {len(scores)} images: {mean_psnr:.2f} dB ({deno_desc})")


if __name__ == "__main__":
    main()
