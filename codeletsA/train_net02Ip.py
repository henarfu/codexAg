"""Train net02Ip: map y0 = B x -> y = A x (inverse of net02p)."""

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


class Y0ToY(nn.Module):
    """Simple MLP: input y0=B x -> output y = A x."""

    def __init__(self, m_in: int, m_out: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(m_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, m_out),
        )

    def forward(self, y0: torch.Tensor) -> torch.Tensor:
        return self.net(y0)


class Y0ProjUNet(nn.Module):
    """Project y0 to pseudo-image -> UNet -> linear head to y."""

    def __init__(self, m_in: int, m_out: int, proj_dim: int = 4096, base_channel: int = 64):
        super().__init__()
        self.proj_in = nn.Linear(m_in, proj_dim)
        self.unet = UNetLeon(n_channels=3, base_channel=base_channel)
        self.head = nn.Linear(3 * 64 * 64, m_out)
        self.proj_dim = proj_dim

    def forward(self, y0: torch.Tensor) -> torch.Tensor:
        bsz = y0.size(0)
        # project to 3xHxW; if proj_dim < 4096 pad zeros
        proj = self.proj_in(y0)  # [bsz, proj_dim]
        if self.proj_dim < 4096:
            pad = torch.zeros((bsz, 4096 - self.proj_dim), device=y0.device, dtype=y0.dtype)
            proj_full = torch.cat([proj, pad], dim=1)
        else:
            proj_full = proj
        z = proj_full.view(bsz, 1, 64, 64)
        z_img = torch.zeros((bsz, 3, 64, 64), device=y0.device, dtype=y0.dtype)
        z_img[:, 0, :, :] = z.squeeze(1)
        feat = self.unet(z_img).view(bsz, -1)
        return self.head(feat)


def train(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    A = load_fixed_A().to(device)  # [m, n]
    B = torch.from_numpy(np.load(args.B_path)).float().to(device)  # [m0, n]
    if args.normalize_b:
        B = torch.nn.functional.normalize(B, dim=1)
    m = A.shape[0]
    m0 = B.shape[0]

    ds = PlacesDataset(args.data_dir, size=64, max_images=args.max_images)
    val_ds = PlacesDataset(args.val_dir, size=64, max_images=args.val_max_images)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.arch == "mlp":
        net = Y0ToY(m_in=m0, m_out=m, hidden=args.hidden).to(device)
    elif args.arch == "unetproj":
        net = Y0ProjUNet(m_in=m0, m_out=m, proj_dim=args.proj_dim, base_channel=args.base_channel).to(device)
    else:
        raise ValueError(f"Unknown arch {args.arch}")
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["model"])
        print(f"Resumed from {args.resume}")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * len(loader))) if args.scheduler == "cosine" else None
    loss_fn = nn.MSELoss()

    def forward_batch(imgs):
        imgs = imgs.to(device)
        bsz = imgs.shape[0]
        x_flat = imgs.view(bsz, -1)
        y = Ax(x_flat, A)              # [B, m]
        y0 = torch.matmul(x_flat, B.t())  # [B, m0]
        if args.scale_y0:
            rms = torch.sqrt(torch.mean(y0 * y0, dim=1, keepdim=True) + 1e-8)
            y0_scaled = y0 / rms
        else:
            y0_scaled = y0
        pred = net(y0_scaled)
        return pred, y

    def evaluate(loader_eval):
        net.eval()
        sse = norm = 0.0
        peak = 0.0
        n_elem = 0
        with torch.no_grad():
            for imgs in loader_eval:
                pred, y = forward_batch(imgs)
                err = pred - y
                sse += torch.sum(err ** 2).item()
                norm += torch.sum(y ** 2).item()
                peak = max(peak, torch.max(torch.abs(y)).item())
                n_elem += err.numel()
        mse = sse / max(1, n_elem)
        snr = 10 * math.log10(norm / (sse + 1e-12)) if sse > 0 else float("inf")
        psnr = 10 * math.log10((peak ** 2) / (mse + 1e-12)) if peak > 0 else float("inf")
        return mse, snr, psnr

    best_val = float("inf")

    for epoch in range(args.epochs):
        running = 0.0
        count = 0
        net.train()
        for step, imgs in enumerate(loader):
            pred, y = forward_batch(imgs)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            running += loss.item() * pred.size(0)
            count += pred.size(0)
            if step % args.log_every == 0:
                print(f"[epoch {epoch} step {step}] loss={loss.item():.4e}")
            if args.max_steps and step >= args.max_steps:
                break
        if count > 0:
            print(f"[epoch {epoch}] avg_loss={running / count:.4e}")
        if args.max_steps and step >= args.max_steps:
            break
        if (epoch + 1) % args.val_every == 0:
            mse, snr, psnr = evaluate(val_loader)
            print(f"[val epoch {epoch}] mse={mse:.4e} snr={snr:.2f}dB psnr={psnr:.2f}dB")
            if mse < best_val:
                best_val = mse
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model": net.state_dict(),
                    "B_path": args.B_path,
                    "arch": args.arch,
                    "hidden": args.hidden,
                    "proj_dim": args.proj_dim,
                    "base_channel": args.base_channel,
                }, Path(args.out))
                print(f"  saved new best to {args.out}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": net.state_dict(),
        "B_path": args.B_path,
        "arch": args.arch,
        "hidden": args.hidden,
        "proj_dim": args.proj_dim,
        "base_channel": args.base_channel,
    }, Path(args.out))
    print(f"Saved net02Ip to {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Train net02Ip: y0=B x -> y = A x.")
    p.add_argument("--data-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/train")
    p.add_argument("--val-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/test")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--max-images", type=int, default=32500)
    p.add_argument("--val-max-images", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--proj-dim", type=int, default=4096)
    p.add_argument("--base-channel", type=int, default=64)
    p.add_argument("--arch", type=str, default="unetproj", choices=["mlp", "unetproj"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--B-path", type=str, default="/home/hdsp/RESULTS/B_teacher02.npy")
    p.add_argument("--out", type=str, default="RESULTS/net02Ip.pt")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--normalize-b", action="store_true", help="L2-normalize rows of B.")
    p.add_argument("--scale-y0", action="store_true", help="Scale y0 to unit RMS per sample.")
    p.add_argument("--grad-clip", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
