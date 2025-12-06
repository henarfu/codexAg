"""Train Teacher 0.2 (predict missing measurements for compression ratio ~0.2).

- Builds/loads B_teacher02.npy (m0 ~ 0.2*n) orthogonal to A's rows.
- Uses UNet-style teacher net to map (x, optional state) -> y_hat (missing measurements).
- Saves checkpoint to RESULTS/teacher02.pt.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from linear_ops import load_fixed_A


class PlacesDataset(Dataset):
    def __init__(self, root: str, size: int = 64, max_images: int | None = None):
        exts = {".jpg", ".jpeg", ".png"}
        paths = [p for p in sorted(Path(root).iterdir()) if p.suffix.lower() in exts]
        if max_images is not None:
            paths = paths[:max_images]
        if len(paths) == 0:
            raise RuntimeError(f"No images found in {root}")
        self.paths = paths
        self.tfm = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tfm(img)


def build_B_teacher02(A: torch.Tensor, ratio: float = 0.2, seed: int = 0) -> torch.Tensor:
    """Build B with rows in nullspace of A, size ~ratio*n."""
    m, n = A.shape
    m0 = int(max(1, ratio * n))
    # Compute orthonormal basis for row(A)
    Q, _ = torch.linalg.qr(A.T)  # [n,m]
    rng = torch.Generator(device=A.device)
    rng.manual_seed(seed)
    rows = []
    for _ in range(m0):
        v = torch.randn(n, device=A.device, generator=rng)
        v = v - Q @ (Q.T @ v)
        v = v / (v.norm() + 1e-8)
        rows.append(v)
    B = torch.stack(rows, dim=0)
    return B


class UNetTeacher(nn.Module):
    def __init__(self, m_out: int, state_dim: int = 4, base_channels: int = 32, embed_dim: int = 128, hidden: int = 256):
        super().__init__()
        c = base_channels
        self.enc1 = nn.Sequential(nn.Conv2d(3, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(c, 2*c, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(2*c, 2*c, 3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(2*c, 4*c, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(4*c, 4*c, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(4*c + state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, m_out),
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        B = x.shape[0]
        g = self.pool(h3).view(B, -1)
        s = torch.cat([g, state], dim=1)
        return self.mlp(s)


def compute_state(x: torch.Tensor, A: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Simple 4-dim state: t=0, residual norm, grad norm, dx=0
    B = x.shape[0]
    x_flat = x.view(B, -1)
    r = x_flat @ A.T - y
    g = r @ A
    res_norm = torch.norm(r, dim=1)
    grad_norm = torch.log(torch.norm(g, dim=1) + 1e-8)
    t = torch.zeros_like(res_norm)
    dx = torch.zeros_like(res_norm)
    return torch.stack([t, res_norm, grad_norm, dx], dim=1)


def train(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    A = load_fixed_A().to(device)
    n = A.shape[1]
    m0 = int(max(1, args.ratio * n))
    B_path = Path(args.B_path)
    if B_path.exists():
        B = torch.from_numpy(np.load(B_path)).to(device)
        print(f"Loaded B from {B_path} shape {B.shape}")
    else:
        B = build_B_teacher02(A, ratio=args.ratio, seed=args.seed).cpu()
        B_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(B_path, B.numpy())
        B = B.to(device)
        print(f"Built and saved B to {B_path} shape {B.shape}")

    ds = PlacesDataset(args.data_dir, size=64, max_images=args.max_images)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    net = UNetTeacher(m_out=m0, state_dim=4, base_channels=32, embed_dim=128, hidden=256).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        for step, imgs in enumerate(loader):
            imgs = imgs.to(device)
            Bsz = imgs.shape[0]
            y = imgs.view(Bsz, -1) @ A.t()
            y0 = imgs.view(Bsz, -1) @ B.t()
            state = compute_state(imgs, A, y)
            pred = net(imgs, state)
            loss = loss_fn(pred, y0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % args.log_every == 0:
                print(f"[epoch {epoch} step {step}] loss={loss.item():.4e}")
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    ckpt = {
        "model": net.state_dict(),
        "B_path": str(B_path),
        "ratio": args.ratio,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"Saved teacher02 to {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/train")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-images", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--B-path", type=str, default="RESULTS/B_teacher02.npy")
    p.add_argument("--out", type=str, default="RESULTS/teacher02.pt")
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
