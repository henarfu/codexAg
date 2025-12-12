"""Train Teacher 0.2 predictor (net02p) using UNet architecture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

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

    def __len__(
        self,
    ):
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

    if args.arch == "mlp":
        net = YOnlyTeacher(m_in=A.shape[0], m_out=m0, hidden=args.hidden).to(device)
    elif args.arch == "transformer":
        net = TransformerTeacher(m_in=A.shape[0], m_out=m0, d_model=args.d_model, nhead=args.nhead, depth=args.depth).to(device)
    else:  # unet
        # Expect y to be reshaped into image-like input; here we keep using flattened measurements and project to 3-channel map
        # m_in (A.shape[0]) is the input size for y, but UNet expects image channels and spatial dimensions.
        # The original code here assumes a fixed reshaping of y to (Bsz, 3, 64, 64) for UNet.
        # So, n_channels should be 3 and base_channel is set by args.base_channel.
        net = UNetLeon(n_channels=3, base_channel=args.base_channel).to(device)

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["model"])
        print(f"Resumed weights from {args.resume}")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * len(loader)))
    else:
        scheduler = None
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        running_loss = 0.0
        count = 0
        for step, imgs in enumerate(loader):
            imgs = imgs.to(device)
            Bsz = imgs.shape[0]
            y = Ax(imgs.view(Bsz, -1), A)          # [B, mA]
            y0 = torch.matmul(imgs.view(Bsz, -1), B.t())  # [B, m0]
            # Use y directly as input
            if args.arch == "unet":
                # map y (mA) into 3x64x64 by padding to 4096 and filling channel 0
                z_img = torch.zeros((Bsz, 3, 64, 64), device=device, dtype=y.dtype)
                pad_len = 64 * 64 # Size of a single channel 64x64 image
                if y.shape[1] < pad_len: # If mA is less than 4096, pad it
                    padded = torch.zeros((Bsz, pad_len), device=device, dtype=y.dtype)
                    padded[:, : y.shape[1]] = y
                else: # If mA is larger, truncate
                    padded = y[:, :pad_len]
                z_img[:, 0] = padded.reshape(Bsz, 64, 64) # Only use the first channel for now
                # The UNet is trained to output 3 channels
                pred_img = net(z_img)
                # We need to extract the part corresponding to m0 from this image-like output
                # This part is complex and depends on how m0 relates to the image output.
                # Assuming m0 is related to a flattened image, we take the first m0 elements.
                pred = pred_img.view(Bsz, -1)[:, :m0]
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
        if count > 0:
            print(f"[epoch {epoch}] avg_loss={running_loss / count:.4e}")

    ckpt = {"model": net.state_dict(), "B_path": str(B_path), "ratio": args.ratio, "arch": args.arch}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"Saved net02p to {out_path}")


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
    p.add_argument("--arch", type=str, default="unet", choices=["mlp", "transformer", "unet"])
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--base-channel", type=int, default=64)
    p.add_argument("--ratio", type=float, default=0.2)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--B-path", type=str, default="/home/hdsp/RESULTS/B_teacher02.npy")
    p.add_argument("--out", type=str, default="RESULTS/teacher02p_unet.pt")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
