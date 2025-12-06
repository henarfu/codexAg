"""Fine-tune a DnCNN denoiser on Places images and save as generaldenoiser."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# DeepInv lives outside this repo.
sys.path.append("/home/hdsp/Documents/Henry/pnp")
import deepinv as dinv  # type: ignore


EXTS = {".jpg", ".jpeg", ".png"}


class PlacesDataset(Dataset):
    def __init__(self, root: Path, size: int = 64, max_images: int | None = None):
        self.paths: List[Path] = [p for p in sorted(root.glob("*")) if p.suffix.lower() in EXTS]
        if max_images is not None:
            self.paths = self.paths[:max_images]
        self.size = size
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = img.resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W]
        if random.random() < 0.5:
            arr = torch.flip(arr, dims=[2])  # horizontal flip
        return arr


def make_loader(root: Path, size: int, batch: int, workers: int, max_images: int | None):
    ds = PlacesDataset(root, size=size, max_images=max_images)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )


def train(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    loader = make_loader(Path(args.train_dir), args.size, args.batch_size, args.num_workers, args.max_images)
    model = dinv.models.DnCNN(in_channels=3, out_channels=3, pretrained="download_lipschitz").to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    global_step = 0
    sigma_min, sigma_max = args.sigma_min, args.sigma_max

    for epoch in range(args.epochs):
        for clean in loader:
            clean = clean.to(device)
            b = clean.shape[0]
            sigmas = torch.rand(b, 1, device=device) * (sigma_max - sigma_min) + sigma_min
            noise = torch.randn_like(clean) * sigmas.view(b, 1, 1, 1)
            noisy = (clean + noise).clamp(0.0, 1.0)
            pred = model(noisy, sigma=sigmas)
            loss = F.mse_loss(pred, clean)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            if global_step % args.log_every == 0:
                print(f"[{global_step}] loss={loss.item():.4e} sigma~U[{sigma_min},{sigma_max}]")
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    out = {
        "model_state": model.state_dict(),
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "steps": global_step,
        "args": vars(args),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"Saved finetuned denoiser to {out_path} (steps={global_step})")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DnCNN on Places and save as generaldenoiser.")
    parser.add_argument("--train-dir", type=str, default="/home/hdsp/Documents/Henry/pnp/data/places/train")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500, help="Stop after this many steps (None = full epoch budget).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sigma-min", type=float, default=0.0)
    parser.add_argument("--sigma-max", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on number of training images.")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="RESULTS/generaldenoiser.pth")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
