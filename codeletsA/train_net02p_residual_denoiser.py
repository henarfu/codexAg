"""Train a residual denoiser on net02p outputs: input y0_pred, target residual = y0_true - y0_pred."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from linear_ops import load_fixed_A, Ax
from codeletsA.train_teacher02p_continued import YToImageUNet


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


class ResidualDenoiser(nn.Module):
    def __init__(self, m: int, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(m, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, m),
        )

    def forward(self, y0_pred: torch.Tensor) -> torch.Tensor:
        return self.net(y0_pred)


def evaluate(loader, net02p, deno, A, B, device):
    net02p.eval()
    deno.eval()
    sse = 0.0
    norm = 0.0
    peak = 0.0
    n_elem = 0
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(device)
            bsz = imgs.shape[0]
            x_flat = imgs.view(bsz, -1)
            y = Ax(x_flat, A)
            y0_true = torch.matmul(x_flat, B.t())
            y0_pred = net02p(y)
            res = deno(y0_pred)
            y0_corr = y0_pred + res
            err = y0_corr - y0_true
            sse += torch.sum(err ** 2).item()
            norm += torch.sum(y0_true ** 2).item()
            peak = max(peak, torch.max(torch.abs(y0_true)).item())
            n_elem += err.numel()
    mse = sse / max(1, n_elem)
    snr = 10 * math.log10(norm / (sse + 1e-12)) if sse > 0 else float("inf")
    psnr = 10 * math.log10((peak ** 2) / (mse + 1e-12)) if peak > 0 else float("inf")
    return mse, snr, psnr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/train")
    p.add_argument("--val-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/test")
    p.add_argument("--max-images", type=int, default=32500)
    p.add_argument("--val-max-images", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--net02p-path", type=str, default="RESULTS/teacher02p_unetproj.pt")
    p.add_argument("--base-channel", type=int, default=96)
    p.add_argument("--B-path", type=str, default="/home/hdsp/RESULTS/B_teacher02.npy")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out", type=str, default="RESULTS/net02p_res_denoiser.pt")
    p.add_argument("--log-every", type=int, default=200)
    args = p.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    A = load_fixed_A().to(device)
    B = torch.from_numpy(np.load(args.B_path)).float().to(device)

    net02p_ckpt = torch.load(args.net02p_path, map_location=device)
    net02p = YToImageUNet(m_in=A.shape[0], m_out=B.shape[0], base_channel=args.base_channel).to(device)
    net02p.load_state_dict(net02p_ckpt["model"], strict=False)
    net02p.eval()

    deno = ResidualDenoiser(m=B.shape[0], hidden=args.hidden).to(device)

    train_ds = PlacesDataset(args.data_dir, size=64, max_images=args.max_images)
    val_ds = PlacesDataset(args.val_dir, size=64, max_images=args.val_max_images)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    opt = torch.optim.Adam(deno.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * len(train_loader)))
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        deno.train()
        running = 0.0
        count = 0
        for step, imgs in enumerate(train_loader):
            imgs = imgs.to(device)
            bsz = imgs.shape[0]
            x_flat = imgs.view(bsz, -1)
            y = Ax(x_flat, A)
            y0_true = torch.matmul(x_flat, B.t())
            with torch.no_grad():
                y0_pred = net02p(y)
            res = deno(y0_pred)
            loss = loss_fn(res, y0_true - y0_pred)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            running += loss.item() * bsz
            count += bsz
            if step % args.log_every == 0:
                print(f"[epoch {epoch} step {step}] loss={loss.item():.4e}")
        if count > 0:
            print(f"[epoch {epoch}] avg_loss={running / count:.4e}")
        mse, snr, psnr = evaluate(val_loader, net02p, deno, A, B, device)
        print(f"[val epoch {epoch}] mse={mse:.4e} snr={snr:.2f}dB psnr={psnr:.2f}dB")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": deno.state_dict(), "hidden": args.hidden, "B_path": args.B_path}, Path(args.out))
    print(f"Saved residual denoiser to {args.out}")


if __name__ == "__main__":
    main()
