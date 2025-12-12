"""Continue training Teacher 0.2 predictor (net02p)."""

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
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import wandb
except Exception:
    wandb = None

sys.path.append(str(Path(__file__).resolve().parent.parent))
from linear_ops import load_fixed_A, Ax, estimate_spectral_norm
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


class TransformerTeacher(nn.Module):
    def __init__(self, m_in: int, m_out: int, d_model: int = 128, nhead: int = 4, depth: int = 2):
        super().__init__()
        self.proj = nn.Linear(m_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(d_model, m_out)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # Treat whole y as a single token after projection
        z = self.proj(y).unsqueeze(1)  # [B,1,d_model]
        h = self.encoder(z)  # [B,1,d_model]
        return self.head(h.squeeze(1))


class YToImageUNet(nn.Module):
    """Project measurement y to 3x64x64 then apply UNetLeon."""

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
    n = A.shape[1]
    m = A.shape[0]
    m0 = int(max(1, args.ratio * n))
    B_path = Path(args.B_path)
    if B_path.exists():
        B = torch.from_numpy(np.load(B_path)).to(device)
        print(f"Loaded B from {B_path} shape {B.shape}")
        m0 = B.shape[0]
    else:
        raise RuntimeError(f"B matrix not found at {B_path}")

    train_ds = PlacesDataset(args.data_dir, size=64, max_images=args.max_images)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = None
    if args.val_dir is not None:
        val_ds = PlacesDataset(args.val_dir, size=64, max_images=args.val_max_images)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.arch == "mlp":
        net = YOnlyTeacher(m_in=A.shape[0], m_out=m0, hidden=args.hidden).to(device)
    elif args.arch == "transformer":
        net = TransformerTeacher(m_in=A.shape[0], m_out=m0, d_model=args.d_model, nhead=args.nhead, depth=args.depth).to(device)
    elif args.arch == "unet_proj":
        net = YToImageUNet(m_in=A.shape[0], m_out=m0, base_channel=args.base_channel).to(device)
    else:  # unet padding baseline
        net = UNetLeon(n_channels=3, base_channel=args.base_channel).to(device)

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["model"])
        print(f"Resumed weights from {args.resume}")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * len(train_loader)))
    else:
        scheduler = None
    loss_fn = nn.MSELoss()

    use_unet_pad = args.arch == "unet"
    use_unet_proj = args.arch == "unet_proj"

    def evaluate(loader: DataLoader) -> tuple[float, float, float, float]:
        """Return (MSE per element, SNR dB, relative L2 error, PSNR dB w.r.t. max |y0|)."""
        net.eval()
        sse = 0.0
        norm_y0 = 0.0
        rel_sum = 0.0
        count = 0
        n_elems = 0
        max_abs_y0 = 0.0
        with torch.no_grad():
            for imgs in loader:
                imgs = imgs.to(device)
                Bsz = imgs.shape[0]
                y = Ax(imgs.view(Bsz, -1), A)
                y0 = torch.matmul(imgs.view(Bsz, -1), B.t())
                max_abs_y0 = max(max_abs_y0, torch.max(torch.abs(y0)).item())
                if use_unet_proj:
                    pred = net(y)
                elif use_unet_pad:
                    z_img = torch.zeros((Bsz, 3, 64, 64), device=device, dtype=y.dtype)
                    pad_len = 64 * 64
                    if y.shape[1] < pad_len:
                        padded = torch.zeros((Bsz, pad_len), device=device, dtype=y.dtype)
                        padded[:, : y.shape[1]] = y
                    else:
                        padded = y[:, :pad_len]
                    z_img[:, 0] = padded.reshape(Bsz, 64, 64)
                    pred = net(z_img).reshape(Bsz, -1)
                    pred = pred[:, : B.shape[0]]
                else:
                    pred = net(y)
                err = pred - y0
                sse += (err ** 2).sum().item()
                norm_y0 += (y0 ** 2).sum().item()
                rel_sum += (torch.norm(err, dim=1) / (torch.norm(y0, dim=1) + 1e-8)).sum().item()
                n_elems += err.numel()
                count += Bsz
        mse = sse / max(1, n_elems)
        snr_db = 10 * math.log10(norm_y0 / (sse + 1e-12)) if sse > 0 else float("inf")
        rel_l2 = rel_sum / max(1, count)
        peak = max_abs_y0 if max_abs_y0 > 0 else 1.0
        psnr_db = 10 * math.log10((peak ** 2) / (mse + 1e-12))
        net.train()
        return mse, snr_db, rel_l2, psnr_db

    history: list[dict[str, float]] = []

    if args.use_wandb and wandb is not None:
        wandb.init(project=args.wandb_project, config=vars(args), name=args.wandb_run_name or None)

    for epoch in range(args.epochs):
        running_loss = 0.0
        count = 0
        for step, imgs in enumerate(train_loader):
            imgs = imgs.to(device)
            Bsz = imgs.shape[0]
            y = Ax(imgs.view(Bsz, -1), A)          # [B, mA]
            y0 = torch.matmul(imgs.view(Bsz, -1), B.t())  # [B, m0]
            # Use y directly as input
            if use_unet_proj:
                pred = net(y)
            elif use_unet_pad:
                # map y (mA) into 3x64x64 by padding to 4096 and filling channel 0
                z_img = torch.zeros((Bsz, 3, 64, 64), device=device, dtype=y.dtype)
                pad_len = 64 * 64
                if y.shape[1] < pad_len:
                    padded = torch.zeros((Bsz, pad_len), device=device, dtype=y.dtype)
                    padded[:, : y.shape[1]] = y
                else:
                    padded = y[:, :pad_len]
                z_img[:, 0] = padded.reshape(Bsz, 64, 64)
                pred = net(z_img).reshape(Bsz, -1)
                pred = pred[:, : m0]
            else:
                pred = net(y)
            loss = loss_fn(pred, y0)
            running_loss += loss.item() * Bsz
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
        if args.max_steps and step >= args.max_steps:
            break
        train_loss = running_loss / count if count > 0 else float("nan")
        print(f"[epoch {epoch}] avg_loss={train_loss:.4e}")
        record = {"epoch": epoch, "train_loss": train_loss}
        if val_loader is not None and ((epoch + 1) % args.val_every == 0):
            val_mse, val_snr, val_rel, val_psnr = evaluate(val_loader)
            print(f"[val epoch {epoch}] mse={val_mse:.4e} snr={val_snr:.2f}dB psnr={val_psnr:.2f}dB rel_l2={val_rel:.4e}")
            record.update({"val_mse": val_mse, "val_snr_db": val_snr, "val_rel_l2": val_rel, "val_psnr_db": val_psnr})
        if args.use_wandb and wandb is not None:
            wandb.log(record)
        history.append(record)

    ckpt = {"model": net.state_dict(), "B_path": str(B_path), "ratio": args.ratio, "arch": args.arch}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"Saved net02p to {out_path}")

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

        if plt is not None and history:
            epochs = [r["epoch"] for r in history]
            train_loss_series = [r["train_loss"] for r in history]
            val_psnr_series = [r.get("val_psnr_db") for r in history]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.plot(epochs, train_loss_series, label="train_loss")
            ax1.set_xlabel("epoch")
            ax1.set_ylabel("MSE loss")
            ax1.grid(True)
            ax1.legend()
            if any(v is not None for v in val_psnr_series):
                ax2.plot([e for e, v in zip(epochs, val_psnr_series) if v is not None],
                         [v for v in val_psnr_series if v is not None],
                         label="val_psnr")
            ax2.set_xlabel("epoch")
            ax2.set_ylabel("PSNR (dB)")
            ax2.grid(True)
            ax2.legend()
            fig.tight_layout()
            plot_path = log_dir / f"{out_path.stem}_curves.png"
            fig.savefig(plot_path)
            plt.close(fig)
            print(f"Saved training curves to {plot_path}")
        elif plt is None:
            print("matplotlib not available; skipped curve plotting")
    if args.use_wandb and wandb is not None:
        wandb.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/train")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-images", type=int, default=50000)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--arch", type=str, default="mlp", choices=["mlp", "transformer", "unet", "unet_proj"])
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--base-channel", type=int, default=64)
    p.add_argument("--ratio", type=float, default=0.2)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--B-path", type=str, default="/home/hdsp/RESULTS/B_teacher02.npy")
    p.add_argument("--out", type=str, default="RESULTS/teacher02p_continued.pt")
    p.add_argument("--resume", type=str, default="RESULTS/teacher02p.pt")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    p.add_argument("--val-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/test")
    p.add_argument("--val-max-images", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--log-dir", type=str, default="RESULTS/logs_net02p")
    p.add_argument("--use-wandb", action="store_true", help="Log metrics to Weights & Biases.")
    p.add_argument("--wandb-project", type=str, default="net02p-training")
    p.add_argument("--wandb-run-name", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
