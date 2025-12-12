"""Train a denoiser-only router (no codelets/lambda) with short-horizon BPTT."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from linear_ops import load_fixed_A, Ax, ATz, grad_f, estimate_spectral_norm
from denoisers_bank import make_denoiser_bank, apply_denoisers
from image_encoder import ImageEncoder
from deno_router import DenoiserRouter, extract_scalar_features, st_gumbel_onehot
from wandb_utils import init_wandb


class FlatImageDataset(Dataset):
    def __init__(self, root: str, size: int, max_images: int | None = None):
        paths = sorted([p for p in Path(root).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
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


def make_loader(data_dir: str, size: int, batch_size: int, workers: int, max_images: int | None):
    ds = FlatImageDataset(data_dir, size=size, max_images=max_images)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)


def train(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = make_loader(args.data_dir, args.size, args.batch_size, args.num_workers, args.max_images)

    A = load_fixed_A().to(device)
    norm = estimate_spectral_norm(A)
    eta = 0.9 / (norm * norm + 1e-8)
    denos = make_denoiser_bank(device)

    encoder = ImageEncoder(embed_dim=64).to(device)
    state_dim = 5 + 64
    router = DenoiserRouter(state_dim=state_dim, hidden=args.hidden, num_deno=len(denos)).to(device)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(router.parameters()), lr=args.lr)
    tau_start, tau_end = 1.0, 0.3
    global_step = 0
    run = init_wandb(
        args.wandb,
        args.wandb_project,
        args.wandb_run_name,
        {"eta": eta, **vars(args)},
    )

    for epoch in range(args.epochs):
        for step, imgs in enumerate(loader):
            imgs = imgs.to(device)
            B = imgs.shape[0]
            y = Ax(imgs.reshape(B, -1), A)
            x = ATz(y, A).reshape_as(imgs)
            x_prev = x.clone()
            tau = tau_start + (tau_end - tau_start) * (epoch / max(1, args.epochs - 1))
            for k in range(args.horizon):
                scalars = extract_scalar_features(x, x_prev, A, y, k, args.horizon)
                img_emb = encoder(x)
                state = torch.cat([scalars, img_emb], dim=1)
                out = router(state)
                d_onehot = st_gumbel_onehot(out["logits_deno"], tau)
                d_idx = d_onehot.argmax(dim=-1)
                g_data = grad_f(x, A, y)
                x_prev = x
                x_pred = x - eta * g_data
                x = apply_denoisers(x_pred, d_idx, denos).clamp(0.0, 1.0)
            loss = ((x - imgs) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1
            psnr_val = (10 * torch.log10(torch.tensor(1.0, device=loss.device) / (loss.detach() + 1e-8))).item()
            if step % args.log_every == 0:
                print(f"[epoch {epoch} step {step}] loss={loss.item():.4e} psnr={psnr_val:.2f} tau={tau:.3f}")
                if run is not None:
                    run.log(
                        {
                            "train/loss": loss.item(),
                            "train/psnr": psnr_val,
                            "train/tau": tau,
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

    ckpt = {"encoder": encoder.state_dict(), "router": router.state_dict(), "eta": eta}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"Saved denoiser router to {out_path}")
    if run is not None:
        run.finish()


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
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--max-images", type=int, default=256)
    p.add_argument("--out", type=str, default="RESULTS/deno_router.pt")
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb-project", type=str, default="codexAgemini", help="wandb project name.")
    p.add_argument("--wandb-run-name", type=str, default=None, help="Optional wandb run name.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
