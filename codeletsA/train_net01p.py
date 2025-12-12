"""Train net01p: predict A-measurement y (size mA) from itself (denoising/regression), using UNet projection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import csv
import numpy as np
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from linear_ops import load_fixed_A, Ax
from codeletsA.unet_leon import UNetLeon


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


class YToImageUNet(nn.Module):
    """Project y to 3x64x64 then apply UNetLeon, output to m_out."""

    def __init__(self, m_in: int, m_out: int, base_channel: int = 64):
        super().__init__()
        self.proj = nn.Linear(m_in, 3 * 64 * 64)
        self.net = UNetLeon(n_channels=3, base_channel=base_channel)
        self.m_out = m_out

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        z = self.proj(y).view(y.shape[0], 3, 64, 64)
        out = self.net(z).reshape(y.shape[0], -1)
        return out[:, : self.m_out]


def train(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    A = load_fixed_A().to(device)
    mA = A.shape[0]
    # Data
    train_ds = PlacesDataset(args.data_dir, size=64, max_images=args.max_images)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = None
    if args.val_dir is not None:
        val_ds = PlacesDataset(args.val_dir, size=64, max_images=args.val_max_images)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    if args.arch == "unet_proj":
        net = YToImageUNet(m_in=mA, m_out=mA, base_channel=args.base_channel).to(device)
    else:
        net = UNetLeon(n_channels=3, base_channel=args.base_channel).to(device)

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["model"], strict=False)
        print(f"Resumed weights from {args.resume}")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * len(train_loader))) if args.scheduler == "cosine" else None
    loss_fn = nn.MSELoss()
    history = []

    def evaluate(loader: DataLoader):
        net.eval()
        sse = 0.0
        norm_y = 0.0
        count = 0
        n_elems = 0
        max_abs = 0.0
        with torch.no_grad():
            for imgs in loader:
                imgs = imgs.to(device)
                Bsz = imgs.shape[0]
                y = Ax(imgs.view(Bsz, -1), A)
                if args.arch == "unet_proj":
                    pred = net(y)
                else:
                    z_img = torch.zeros((Bsz, 3, 64, 64), device=device, dtype=y.dtype)
                    pad_len = 64 * 64
                    padded = torch.zeros((Bsz, pad_len), device=device, dtype=y.dtype)
                    padded[:, : y.shape[1]] = y
                    z_img[:, 0] = padded.reshape(Bsz, 64, 64)
                    pred = net(z_img).reshape(Bsz, -1)
                    pred = pred[:, : mA]
                err = pred - y
                sse += (err ** 2).sum().item()
                norm_y += (y ** 2).sum().item()
                n_elems += err.numel()
                count += Bsz
                max_abs = max(max_abs, torch.max(torch.abs(y)).item())
        mse = sse / max(1, n_elems)
        snr_db = 10 * math.log10(norm_y / (sse + 1e-12)) if sse > 0 else float("inf")
        peak = max_abs if max_abs > 0 else 1.0
        psnr_db = 10 * math.log10((peak ** 2) / (mse + 1e-12))
        net.train()
        return mse, snr_db, psnr_db

    for epoch in range(args.epochs):
        running = 0.0
        count = 0
        for step, imgs in enumerate(train_loader):
            imgs = imgs.to(device)
            Bsz = imgs.shape[0]
            y = Ax(imgs.view(Bsz, -1), A)
            if args.arch == "unet_proj":
                pred = net(y)
            else:
                z_img = torch.zeros((Bsz, 3, 64, 64), device=device, dtype=y.dtype)
                pad_len = 64 * 64
                padded = torch.zeros((Bsz, pad_len), device=device, dtype=y.dtype)
                padded[:, : y.shape[1]] = y
                z_img[:, 0] = padded.reshape(Bsz, 64, 64)
                pred = net(z_img).reshape(Bsz, -1)
                pred = pred[:, : mA]
            loss = loss_fn(pred, y)
            running += loss.item() * Bsz
            count += Bsz
            opt.zero_grad()
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()
            if step % args.log_every == 0:
                print(f"[epoch {epoch} step {step}] loss={loss.item():.4e}")
            if args.max_steps and step >= args.max_steps:
                break
        if count > 0:
            avg_loss = running / count
            print(f"[epoch {epoch}] avg_loss={avg_loss:.4e}")
        record = {"epoch": epoch, "train_loss": running / count if count > 0 else float("nan")}
        if val_loader is not None and ((epoch + 1) % args.val_every == 0):
            val_mse, val_snr, val_psnr = evaluate(val_loader)
            print(f"[val epoch {epoch}] mse={val_mse:.4e} snr={val_snr:.2f}dB psnr={val_psnr:.2f}dB")
            record.update({"val_mse": val_mse, "val_snr_db": val_snr, "val_psnr_db": val_psnr})
        history.append(record)
        if args.max_steps and step >= args.max_steps:
            break

    ckpt = {"model": net.state_dict(), "arch": args.arch}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"Saved net01p to {out_path}")

    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({k for r in history for k in r.keys()})
        csv_path = log_dir / f"{out_path.stem}_metrics.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in history:
                writer.writerow(r)
        print(f"Saved metrics to {csv_path}")
        try:
            import matplotlib.pyplot as plt
            if history:
                epochs = [r["epoch"] for r in history]
                train_loss_series = [r["train_loss"] for r in history]
                val_psnr_series = [r.get("val_psnr_db") for r in history]
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.plot(epochs, train_loss_series, label="train_loss")
                ax1.set_xlabel("epoch"); ax1.set_ylabel("MSE loss"); ax1.grid(True); ax1.legend()
                if any(v is not None for v in val_psnr_series):
                    ax2.plot([e for e,v in zip(epochs,val_psnr_series) if v is not None],
                             [v for v in val_psnr_series if v is not None], label="val_psnr")
                    ax2.set_xlabel("epoch"); ax2.set_ylabel("PSNR (dB)"); ax2.grid(True); ax2.legend()
                fig.tight_layout()
                plot_path = log_dir / f"{out_path.stem}_curves.png"
                fig.savefig(plot_path); plt.close(fig)
                print(f"Saved training curves to {plot_path}")
        except Exception as e:
            print(f"Plotting skipped: {e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/train")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--max-images", type=int, default=50000)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--arch", type=str, default="unet_proj", choices=["unet", "unet_proj"])
    p.add_argument("--base-channel", type=int, default=96)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out", type=str, default="RESULTS/net01p_unetproj.pt")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    p.add_argument("--val-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/test")
    p.add_argument("--val-max-images", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--log-dir", type=str, default="RESULTS/logs_net01p")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
