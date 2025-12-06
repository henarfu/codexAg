"""Train Teacher 0.2 predictor using only the base measurement y as input (net02p)."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from linear_ops import load_fixed_A, Ax, estimate_spectral_norm


class PlacesDataset(Dataset):
    def __init__(self, root: str, size: int = 64, max_images: int | None = None):
        exts = {".jpg", ".jpeg", ".png"}
        paths = [p for p in sorted(Path(root).iterdir()) if p.suffix.lower() in exts]
        if max_images is not None:
            paths = paths[:max_images]
        if len(paths) == 0:
            raise RuntimeError(f"No images found in {root}")
        self.paths = paths
        self.tfm = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tfm(img)


class YOnlyTeacher(nn.Module):
    def __init__(self, m_in: int, m_out: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(m_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, m_out),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


def train(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    A = load_fixed_A().to(device)
    n = A.shape[1]
    m = A.shape[0]
    m0 = int(max(1, args.ratio * n))
    B_path = Path(args.B_path)
    if B_path.exists():
        B = torch.from_numpy(np.load(B_path)).to(device)
        print(f"Loaded B from {B_path} shape {B.shape}")
    else:
        raise RuntimeError(f"B matrix not found at {B_path}")

    ds = PlacesDataset(args.data_dir, size=64, max_images=args.max_images)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    net = YOnlyTeacher(m_in=m, m_out=m0, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        for step, imgs in enumerate(loader):
            imgs = imgs.to(device)
            Bsz = imgs.shape[0]
            y = Ax(imgs.view(Bsz, -1), A)          # [B, m]
            y0 = torch.matmul(imgs.view(Bsz, -1), B.t())  # [B, m0]
            pred = net(y)
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

    ckpt = {"model": net.state_dict(), "B_path": str(B_path), "ratio": args.ratio, "arch": "YOnlyTeacher"}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"Saved net02p to {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/train")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-images", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--ratio", type=float, default=0.2)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--B-path", type=str, default="RESULTS/B_teacher02.npy")
    p.add_argument("--out", type=str, default="RESULTS/teacher02p.pt")
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
