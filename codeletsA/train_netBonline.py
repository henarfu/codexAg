"""Train netBonline: predict a single missing measurement given y and one row Bi.

- Inputs: base measurement y (A x) and a single row Bi from B (optionally normalized) with a positional embedding for the row index.
- Output: scalar prediction of Bi @ x.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from linear_ops import load_fixed_A, Ax
import csv


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


class NetBonline(nn.Module):
    """Predict a single measurement given y, Bi, and positional embedding."""

    def __init__(self, m_y: int, n: int, pos_size: int, y_dim: int = 192, b_dim: int = 96, pos_dim: int = 64):
        super().__init__()
        self.y_proj = nn.Sequential(nn.Linear(m_y, y_dim), nn.ReLU())
        self.b_proj = nn.Sequential(nn.Linear(n, b_dim), nn.ReLU())
        self.pos_emb = nn.Embedding(pos_size, pos_dim)
        hidden = y_dim + b_dim + pos_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, y: torch.Tensor, b_row: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        y_feat = self.y_proj(y)
        b_feat = self.b_proj(b_row)
        pos_feat = self.pos_emb(idx)
        h = torch.cat([y_feat, b_feat, pos_feat], dim=1)
        return self.mlp(h).squeeze(1)


def make_dataloaders(args, device, A: torch.Tensor, B: torch.Tensor) -> Tuple[DataLoader, DataLoader]:
    train_ds = PlacesDataset(args.data_dir, size=64, max_images=args.max_images)
    val_ds = PlacesDataset(args.val_dir, size=64, max_images=args.val_max_images)
    common = dict(num_workers=args.num_workers, pin_memory=True)
    # Each batch still pulls images only; Bi sampling is done in the loop with k_rows per image.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(net: nn.Module, loader: DataLoader, A: torch.Tensor, B: torch.Tensor, normalize_b: bool, device: torch.device, k_rows: int):
    net.eval()
    sse = 0.0
    norm_y0 = 0.0
    peak = 0.0
    n_elem = 0
    rel_sum = 0.0
    count = 0
    for imgs in loader:
        imgs = imgs.to(device)
        bsz = imgs.shape[0]
        y = Ax(imgs.view(bsz, -1), A)
        # sample random rows
        # sample k_rows per image (uniform over all rows)
        idx = torch.randint(0, B.shape[0], (bsz, k_rows), device=device)
        b_rows = B[idx]  # [B, k, n]
        if normalize_b:
            b_rows = torch.nn.functional.normalize(b_rows, dim=2)
        img_flat = imgs.view(bsz, 1, -1)
        target = torch.sum(img_flat * b_rows, dim=2)  # [B,k]
        y_rep = y.unsqueeze(1).expand(-1, k_rows, -1).reshape(-1, y.shape[1])
        b_rep = b_rows.reshape(-1, B.shape[1])
        idx_flat = idx.reshape(-1)
        pred = net(y_rep, b_rep, idx_flat).view(bsz, k_rows)
        err = pred - target
        sse += torch.sum(err ** 2).item()
        norm_y0 += torch.sum(target ** 2).item()
        peak = max(peak, torch.max(torch.abs(target)).item())
        n_elem += err.numel()
        rel_sum += (torch.norm(err) / (torch.norm(target) + 1e-8)).item()
        count += 1
    mse = sse / max(1, n_elem)
    snr_db = 10 * math.log10(norm_y0 / (sse + 1e-12)) if sse > 0 else float("inf")
    psnr_db = 10 * math.log10((peak ** 2) / (mse + 1e-12)) if peak > 0 else float("inf")
    rel = rel_sum / max(1, count)
    return mse, snr_db, psnr_db, rel


def train(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    A = load_fixed_A().to(device)
    B_np = np.load(args.B_path)
    B = torch.from_numpy(B_np).float().to(device)
    if args.normalize_b:
        B = torch.nn.functional.normalize(B, dim=1)
    m_y = A.shape[0]
    n = B.shape[1]
    pos_size = B.shape[0]

    net = NetBonline(m_y=m_y, n=n, pos_size=pos_size, y_dim=args.y_dim, b_dim=args.b_dim, pos_dim=args.pos_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    else:
        scheduler = None
    loss_fn = nn.MSELoss()

    train_loader, val_loader = make_dataloaders(args, device, A, B)

    hist_epochs = []
    hist_train = []
    hist_val = []

    for epoch in range(args.epochs):
        net.train()
        running = 0.0
        count = 0
        # determine block for this epoch if block scheduling is enabled
        if args.block_size is not None and args.block_size > 0:
            num_blocks = math.ceil(pos_size / args.block_size)
            block_id = epoch % num_blocks
            start = block_id * args.block_size
            end = min(pos_size, start + args.block_size)
            epoch_rows = torch.arange(start, end, device=device)
        else:
            epoch_rows = None  # use all rows

        for step, imgs in enumerate(train_loader):
            imgs = imgs.to(device)
            bsz = imgs.shape[0]
            y = Ax(imgs.view(bsz, -1), A)
            # sample k_rows per image
            if epoch_rows is not None:
                idx_local = torch.randint(0, epoch_rows.numel(), (bsz, args.k_rows), device=device)
                idx = epoch_rows[idx_local]
            else:
                idx = torch.randint(0, pos_size, (bsz, args.k_rows), device=device)
            b_rows = B[idx]  # [B, k, n]
            img_flat = imgs.view(bsz, 1, -1)
            target = torch.sum(img_flat * b_rows, dim=2)  # [B,k]
            y_rep = y.unsqueeze(1).expand(-1, args.k_rows, -1).reshape(-1, y.shape[1])
            b_rep = b_rows.reshape(-1, B.shape[1])
            idx_flat = idx.reshape(-1)
            pred = net(y_rep, b_rep, idx_flat).view(bsz, args.k_rows)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * bsz
            count += bsz
            if step % args.log_every == 0:
                print(f"[epoch {epoch} step {step}] loss={loss.item():.4e}")
        if scheduler is not None:
            scheduler.step()
        if count > 0:
            print(f"[epoch {epoch}] avg_loss={running / count:.4e}")
        hist_epochs.append(epoch)
        hist_train.append(running / max(1, count))
        if (epoch + 1) % args.val_every == 0:
            mse, snr_db, psnr_db, rel = evaluate(net, val_loader, A, B, args.normalize_b, device, args.k_rows)
            print(f"[val epoch {epoch}] mse={mse:.4e} snr={snr_db:.2f}dB psnr={psnr_db:.2f}dB rel={rel:.4e}")
            hist_val.append((mse, snr_db, psnr_db, rel))
        else:
            hist_val.append((None, None, None, None))

    ckpt = {
        "model": net.state_dict(),
        "B_path": args.B_path,
        "normalize_b": args.normalize_b,
        "arch": "netBonline",
        "y_dim": args.y_dim,
        "b_dim": args.b_dim,
        "pos_dim": args.pos_dim,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"Saved netBonline to {out_path}")

    if args.log_csv is not None:
        log_path = Path(args.log_csv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_mse", "val_snr_db", "val_psnr_db", "val_rel"])
            for e, tr, (vm, vs, vp, vr) in zip(hist_epochs, hist_train, hist_val):
                writer.writerow([e, tr, vm, vs, vp, vr])
        print(f"Saved logs to {log_path}")
    if args.log_plot is not None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            logp = Path(args.log_plot)
            logp.parent.mkdir(parents=True, exist_ok=True)
            train_losses = hist_train
            val_psnr = [v[2] for v in hist_val]
            fig, ax1 = plt.subplots()
            ax1.plot(hist_epochs, train_losses, label="train_loss")
            ax1.set_xlabel("epoch")
            ax1.set_ylabel("train_loss")
            ax2 = ax1.twinx()
            ax2.plot(hist_epochs, val_psnr, color="orange", label="val_psnr")
            ax2.set_ylabel("val_psnr (dB)")
            fig.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(logp)
            plt.close(fig)
            print(f"Saved plot to {logp}")
        except Exception as e:
            print(f"Could not save plot: {e}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/train")
    p.add_argument("--val-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/test")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max-images", type=int, default=10000)
    p.add_argument("--val-max-images", type=int, default=1000)
    p.add_argument("--k-rows", type=int, default=4, help="Number of Bi rows sampled per image per batch/val step.")
    p.add_argument("--block-size", type=int, default=None, help="Optional block size for Bi row scheduling; if set, each epoch samples rows from a contiguous block, cycling through blocks.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--y-dim", type=int, default=192)
    p.add_argument("--b-dim", type=int, default=96)
    p.add_argument("--pos-dim", type=int, default=64)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--B-path", type=str, default="/home/hdsp/RESULTS/B_teacher02.npy")
    p.add_argument("--normalize-b", action="store_true")
    p.add_argument("--out", type=str, default="RESULTS/netBonline.pt")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    p.add_argument("--log-csv", type=str, default="RESULTS/netBonline_loss.csv")
    p.add_argument("--log-plot", type=str, default="RESULTS/netBonline_loss.png")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
