"""Train router via short-horizon BPTT to pick codelet/lambda/denoiser."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from linear_ops import load_fixed_A, Ax, ATz, grad_f, estimate_spectral_norm
from denoisers_bank import make_denoiser_bank, apply_denoisers
from codelets import apply_codelet_batch, normalize_to_data, LinearTransform
from image_encoder import ImageEncoder
from router import CodeletDenoiserRouter, extract_scalar_features, st_gumbel_onehot

LAMBDA_BINS = [0.0, 0.03, 0.1, 0.3, 1.0]


class FlatImageDataset(Dataset):
    """Loads all images in a folder (no class subdirs needed)."""

    def __init__(self, root: str, size: int, max_images: int | None = None):
        paths = sorted([p for p in Path(root).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if max_images is not None:
            paths = paths[:max_images]
        if len(paths) == 0:
            raise RuntimeError(f"No images found in {root}")
        self.paths = paths
        self.tfm = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.tfm(img)


def make_loaders(data_dir: str, size: int, batch_size: int, workers: int, max_images: int | None):
    ds = FlatImageDataset(data_dir, size=size, max_images=max_images)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)


def build_codelet_params(device: torch.device, n: int):
    # Graph kernel (depthwise Laplacian)
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


def train(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = make_loaders(args.data_dir, args.size, args.batch_size, args.num_workers, args.max_images)

    A = load_fixed_A().to(device)
    norm = estimate_spectral_norm(A)
    eta = 0.9 / (norm * norm + 1e-8)
    denos = make_denoiser_bank(device)
    codelet_params = build_codelet_params(device, n=A.shape[1])

    encoder = ImageEncoder(embed_dim=64).to(device)
    state_dim = 5 + 64  # scalars + embedding
    router = CodeletDenoiserRouter(state_dim=state_dim, hidden=args.hidden).to(device)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(router.parameters()), lr=args.lr)

    steps_per_epoch = math.ceil(len(loader))
    tau_start, tau_end = 1.0, 0.3

    for epoch in range(args.epochs):
        for step, imgs in enumerate(loader):
            imgs = imgs.to(device)
            B = imgs.shape[0]
            # forward model
            y = Ax(imgs.view(B, -1), A)
            x = ATz(y, A).view_as(imgs)
            x_prev = x.clone()
            tau = tau_start + (tau_end - tau_start) * (epoch / max(1, args.epochs - 1))
            for k in range(args.horizon):
                scalars = extract_scalar_features(x, x_prev, A, y, k, args.horizon)
                img_emb = encoder(x)
                state = torch.cat([scalars, img_emb], dim=1)
                out = router(state)
                c_onehot = st_gumbel_onehot(out["logits_codelet"], tau)
                l_onehot = st_gumbel_onehot(out["logits_lambda"], tau)
                d_onehot = st_gumbel_onehot(out["logits_deno"], tau)
                c_idx = c_onehot.argmax(dim=-1)
                l_idx = l_onehot.argmax(dim=-1)
                d_idx = d_onehot.argmax(dim=-1)
                lam = torch.tensor(LAMBDA_BINS, device=device)[l_idx]

                g_data = grad_f(x, A, y)
                g_code = apply_codelet_batch(c_idx, x, A, y, codelet_params)
                g_code_used = normalize_to_data(g_code, g_data)
                step_dir = g_data + lam.view(-1, 1, 1, 1) * g_code_used
                x_prev = x
                x_pred = x - eta * step_dir
                x = apply_denoisers(x_pred, d_idx, denos).clamp(0.0, 1.0)

            loss = ((x - imgs) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % args.log_every == 0:
                print(f"[epoch {epoch} step {step}/{steps_per_epoch}] loss={loss.item():.4e} tau={tau:.3f}")

    ckpt = {
        "encoder": encoder.state_dict(),
        "router": router.state_dict(),
        "eta": eta,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"Saved router to {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/train")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out", type=str, default="RESULTS/router.pt")
    p.add_argument("--max-images", type=int, default=128, help="Limit number of training images for speed.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
