"""
Train a simple selector network (NetS) that picks a codelet per iteration.
Features are extracted from the PRECOND_DATA trajectory (as a proxy state),
labels are the per-iteration winners (argmax PSNR across codelets).
This is a lightweight, bounded-time scaffold; adjust num_images/epochs as needed.
"""
import json
import sys
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

sys.path.append("/home/hdsp/Documents/Henry/pnp")
import deepinv as dinv  # noqa: E402

from codelets_custom import (
    codelet_robust_data,
    codelet_precond_data,
    codelet_ares,
    codelet_sparse_w,
    codelet_nullspace,
)
from tune_and_probabilities import make_fixed_operators


def load_images(folder, size, device, limit=None, offset=0):
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((size, size), antialias=True),
        ]
    )
    paths = sorted(Path(folder).glob("*.jpg"))
    if offset:
        paths = paths[offset:]
    if limit:
        paths = paths[:limit]
    for p in paths:
        img = Image.open(p).convert("RGB")
        yield p, tfm(img).unsqueeze(0).to(device)


def psnr(x, y):
    mse = torch.mean((x - y) ** 2)
    return 10 * torch.log10(torch.tensor(1.0, device=x.device) / (mse + 1e-12))


def apply_codelet_step(x, name, codelets, data_fid, prior, A, y, step_base):
    params = codelets[name]["params"]
    g_dir = codelets[name]["fn"](x, A, y, params)
    grad = data_fid.grad(x, y, A)
    x_pred = x - step_base * params.get("step_mul", 1.0) * (grad + params.get("lambda", 1.0) * g_dir)
    x_new = prior.prox(x_pred, sigma_denoiser=step_base * 5e-4, physics=A)
    return x_new


def run_algorithms(A, y, x0, x_gt, codelets: Dict[str, Dict], iters, alpha, lambd):
    data_fid = dinv.optim.L2()
    prior = dinv.optim.PnP(
        denoiser=dinv.models.DnCNN(
            in_channels=3, out_channels=3, pretrained="download_lipschitz"
        ).to(x0.device)
    )
    norm = A.compute_norm(A.A_adjoint(y), tol=1e-3).item() + 1e-8
    step_base = alpha / norm
    sigma_base = lambd * step_base

    alg_names = list(codelets.keys())
    traj_psnr = torch.zeros((len(alg_names), iters), device=x0.device)
    traj_states = torch.zeros((iters, *x0.shape), device=x0.device)  # for PRECOND proxy
    feat_list = []

    def xk_features(x_tensor):
        """Extract features from x_k."""
        ch_mean = x_tensor.mean(dim=[0, 2, 3])  # (C,)
        ch_std = x_tensor.std(dim=[0, 2, 3])
        x_min = x_tensor.min()
        x_max = x_tensor.max()
        return torch.cat([ch_mean, ch_std, x_min.unsqueeze(0), x_max.unsqueeze(0)])

    with torch.no_grad():
        # Run PRECOND once to get proxy states for features.
        precond_params = codelets["PRECOND_DATA"]["params"]
        x_state = x0.clone()
        prev = x_state.clone()
        for k in range(iters):
            grad = data_fid.grad(x_state, y, A)
            g_dir = codelet_precond_data(x_state, A, y, precond_params)
            x_pred = x_state - step_base * (grad + precond_params.get("lambda", 1.0) * g_dir)
            x_state = prior.prox(x_pred, sigma_denoiser=sigma_base, physics=A)
            traj_states[k] = x_state
            feat = xk_features(x_state)
            feat_list.append(feat.cpu())
            prev = x_state.clone()  # kept for parity with original flow

        for ai, name in enumerate(alg_names):
            g_fn, params = codelets[name]["fn"], codelets[name]["params"]
            step_mul = params.get("step_mul", 1.0)
            lam_scale = params.get("lambda", 1.0)
            x = x0.clone()
            for k in range(iters):
                grad = data_fid.grad(x, y, A)
                g_dir = g_fn(x, A, y, params)
                x_pred = x - step_base * step_mul * (grad + lam_scale * g_dir)
                x = prior.prox(x_pred, sigma_denoiser=sigma_base, physics=A)
                traj_psnr[ai, k] = psnr(x, x_gt)
    features = torch.stack(feat_list)  # (K, feat_dim)
    return alg_names, traj_psnr, traj_states, features, data_fid, prior, step_base, sigma_base


def build_dataset(features, winners):
    """
    features: (K, F) precomputed per-iteration features
    winners: (K,) int labels
    """
    feats = features.cpu()
    labels = torch.tensor([int(w.item()) for w in winners], dtype=torch.long)
    return feats, labels


class NetS(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def compute_horizon_labels(traj_states, codelets, alg_names, data_fid, prior, A, y, step_base, sigma_base, x_gt, horizon):
    K = traj_states.shape[0]
    labels = []
    names = list(alg_names)
    perms = list(itertools.permutations(names, horizon))
    for k in range(K):
        x_start = traj_states[k].clone()  # already has batch dim (1,C,H,W)
        best_psnr = -1e9
        best_first = names[0]
        for perm in perms:
            x = x_start.clone()
            with torch.no_grad():
                for name in perm:
                    params = codelets[name]["params"]
                    grad = data_fid.grad(x, y, A)
                    g_dir = codelets[name]["fn"](x, A, y, params)
                    x_pred = x - step_base * params.get("step_mul", 1.0) * (grad + params.get("lambda", 1.0) * g_dir)
                    x = prior.prox(x_pred, sigma_denoiser=sigma_base, physics=A)
            score = psnr(x, x_gt).item()
            if score > best_psnr:
                best_psnr = score
                best_first = perm[0]
        labels.append(names.index(best_first))
    return torch.tensor(labels, dtype=torch.long)


def train_nets(device, size=64, iters=150, alpha=1.5, lambd=5e-4, num_images=5, epochs=5, hidden=128, horizon=3):
    ops = make_fixed_operators(size, device)
    codelets = {
        "ROBUST_DATA": {"fn": codelet_robust_data, "params": {"tau_rob": 0.1, "p_rob": 1.2, "lambda": 1.5, "step_mul": 0.5}},
        "PRECOND_DATA": {"fn": codelet_precond_data, "params": {"kernels": ops["kernels"], "coeffs": ops["coeffs"], "lambda": 0.8, "step_mul": 1.0}},
        "ARES": {"fn": codelet_ares, "params": {"alpha": 0.05, "beta": 0.0, "lambda": 1.0, "step_mul": 1.25}},
        "SPARSE_W": {"fn": lambda x, A, y, p: codelet_sparse_w(x, p), "params": {"W": ops["W"], "tau_sp": 0.003, "lambda": 0.5, "step_mul": 0.5}},
        "NULLSPACE": {"fn": lambda x, A, y, p: codelet_nullspace(x, p), "params": {"U": ops["U"], "tau_n": 0.003, "lambda": 0.5, "step_mul": 1.0}},
    }
    alg_names = list(codelets.keys())
    num_classes = len(alg_names)

    feats_list = []
    labels_list = []
    for idx, (_, x_gt) in enumerate(load_images("/home/hdsp/Documents/Henry/pnp/data/places/test", size, device, limit=num_images, offset=0)):
        A = dinv.physics.SinglePixelCamera(
            m=int(0.1 * x_gt.numel() / x_gt.shape[0]),
            img_size=tuple(x_gt.shape[1:]),
            device=device,
        )
        y = A(x_gt)
        x0 = A.A_adjoint(y)
        algs, traj_psnr, traj_states, features, data_fid, prior, step_base, sigma_base = run_algorithms(A, y, x0, x_gt, codelets, iters, alpha, lambd)
        winners = compute_horizon_labels(traj_states, codelets, algs, data_fid, prior, A, y, step_base, sigma_base, x_gt, horizon)
        feats, labels = build_dataset(features, winners)
        feats_list.append(feats)
        labels_list.append(labels)
        print(f"[data] image {idx} done")

    X = torch.cat(feats_list, dim=0)
    Y = torch.cat(labels_list, dim=0)

    model = NetS(input_dim=X.shape[1], num_classes=num_classes, hidden=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(X.to(device), Y.to(device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for ep in range(epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f"[train] epoch {ep} loss {total / len(dataset):.4f}")

    out_path = Path("RESULTS/nets_weights.pth")
    torch.save({"model": model.state_dict(), "algs": alg_names}, out_path)
    print(f"Saved NetS weights to {out_path}")
    return model, alg_names


def run_policy(model, alg_names, device, size=64, iters=150, alpha=1.5, lambd=5e-4, num_images=2):
    ops = make_fixed_operators(size, device)
    codelets = {
        "ROBUST_DATA": {"fn": codelet_robust_data, "params": {"tau_rob": 0.1, "p_rob": 1.2, "lambda": 1.5, "step_mul": 0.5}},
        "PRECOND_DATA": {"fn": codelet_precond_data, "params": {"kernels": ops["kernels"], "coeffs": ops["coeffs"], "lambda": 0.8, "step_mul": 1.0}},
        "ARES": {"fn": codelet_ares, "params": {"alpha": 0.05, "beta": 0.0, "lambda": 1.0, "step_mul": 1.25}},
        "SPARSE_W": {"fn": lambda x, A, y, p: codelet_sparse_w(x, p), "params": {"W": ops["W"], "tau_sp": 0.003, "lambda": 0.5, "step_mul": 0.5}},
        "NULLSPACE": {"fn": lambda x, A, y, p: codelet_nullspace(x, p), "params": {"U": ops["U"], "tau_n": 0.003, "lambda": 0.5, "step_mul": 1.0}},
    }
    data_fid = dinv.optim.L2()
    prior = dinv.optim.PnP(
        denoiser=dinv.models.DnCNN(
            in_channels=3, out_channels=3, pretrained="download_lipschitz"
        ).to(device)
    )

    psnr_nets = []
    psnr_base = []

    model.eval()
    with torch.no_grad():
        for idx, (_, x_gt) in enumerate(load_images("/home/hdsp/Documents/Henry/pnp/data/places/test", size, device, limit=num_images, offset=10)):
            A = dinv.physics.SinglePixelCamera(
                m=int(0.1 * x_gt.numel() / x_gt.shape[0]),
                img_size=tuple(x_gt.shape[1:]),
                device=device,
            )
            y = A(x_gt)
            x0 = A.A_adjoint(y)
            norm = A.compute_norm(A.A_adjoint(y), tol=1e-3).item() + 1e-8
            step_base = alpha / norm
            sigma_base = lambd * step_base

            # NetS-driven run
            x = x0.clone()
            prev = x.clone()
            for k in range(iters):
                k_norm = torch.tensor([k / iters], device=device)
                resid = A(x) - y
                res_norm = resid.norm() / (y.norm() + 1e-8)
                bp = A.A_adjoint(resid)
                grad_norm = bp.norm()
                delta_norm = (x - prev).norm() / (prev.norm() + 1e-8)
                stats = torch.stack([x.mean(), x.std(), x.max(), x.min()])
                feat = torch.cat([k_norm, res_norm.unsqueeze(0), grad_norm.unsqueeze(0), delta_norm.unsqueeze(0), stats]).unsqueeze(0)
                logits = model(feat)
                choice = torch.argmax(logits, dim=1).item()
                name = alg_names[choice]
                params = codelets[name]["params"]
                g_dir = codelets[name]["fn"](x, A, y, params)
                grad = data_fid.grad(x, y, A)
                x_pred = x - step_base * params.get("step_mul", 1.0) * (grad + params.get("lambda", 1.0) * g_dir)
                x = prior.prox(x_pred, sigma_denoiser=sigma_base, physics=A)
                prev = x.clone()
            psnr_nets.append(psnr(x, x_gt).item())

            # Baseline: PRECOND only
            x = x0.clone()
            params = codelets["PRECOND_DATA"]["params"]
            for _ in range(iters):
                grad = data_fid.grad(x, y, A)
                g_dir = codelet_precond_data(x, A, y, params)
                x_pred = x - step_base * (grad + params.get("lambda", 1.0) * g_dir)
                x = prior.prox(x_pred, sigma_denoiser=sigma_base, physics=A)
            psnr_base.append(psnr(x, x_gt).item())

            print(f"[eval] image {idx}: NetS PSNR {psnr_nets[-1]:.2f}, baseline {psnr_base[-1]:.2f}")
    print(f"NetS avg PSNR: {sum(psnr_nets)/len(psnr_nets):.2f}, baseline avg: {sum(psnr_base)/len(psnr_base):.2f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train NetS selector and evaluate vs baseline.")
    parser.add_argument("--train-images", type=int, default=10, help="Number of training images.")
    parser.add_argument("--eval-images", type=int, default=3, help="Number of eval images.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--hidden", type=int, default=128, help="NetS hidden width.")
    parser.add_argument("--size", type=int, default=64, help="Image size (must respect the law).")
    parser.add_argument("--iters", type=int, default=150, help="PGD iterations per image (set 1 to remove horizon).")
    parser.add_argument("--horizon", type=int, default=3, help="Horizon length for label generation (3 or 4).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, algs = train_nets(
        device,
        size=args.size,
        iters=args.iters,
        num_images=args.train_images,
        epochs=args.epochs,
        hidden=args.hidden,
        horizon=args.horizon,
    )
    run_policy(
        model,
        algs,
        device,
        size=args.size,
        iters=args.iters,
        num_images=args.eval_images,
    )


if __name__ == "__main__":
    main()
