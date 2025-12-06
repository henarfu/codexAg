import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torchvision import transforms
from PIL import Image

sys.path.append("/home/hdsp/Documents/Henry/pnp")
import deepinv as dinv

from codelets_custom import (
    codelet_data,
    codelet_robust_data,
    codelet_precond_data,
    codelet_ares,
    codelet_l2,
    codelet_l1,
    codelet_tv,
    codelet_graph,
    codelet_nonlocal,
    codelet_sparse_w,
    codelet_nullspace,
    codelet_bquad,
)


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


def make_fixed_operators(size, device):
    laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    kernels = [laplacian / 4, sobel_x / 8, sobel_y / 8]
    coeffs = [0.5, 0.25, 0.25]

    n = size * size
    L = torch.zeros((n, n), dtype=torch.float32)
    for i in range(size):
        for j in range(size):
            idx = i * size + j
            L[idx, idx] = 4.0
            if i > 0:
                L[idx, idx - size] = -1.0
            if i < size - 1:
                L[idx, idx + size] = -1.0
            if j > 0:
                L[idx, idx - 1] = -1.0
            if j < size - 1:
                L[idx, idx + 1] = -1.0

    L_nl = L.clone()
    W = torch.eye(n, dtype=torch.float32)
    q = 8
    U = torch.linalg.qr(torch.randn(n, q, dtype=torch.float32), mode="reduced")[0]
    # Use graph Laplacian as B to make BQUAD a meaningful smoother.
    B = L.clone()

    return {
        "kernels": [k.to(device) for k in kernels],
        "coeffs": coeffs,
        "L": L.to(device),
        "L_nl": L_nl.to(device),
        "W": W.to(device),
        "U": U.to(device),
        "B": B.to(device),
    }


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

    with torch.no_grad():
        for ai, name in enumerate(alg_names):
            g_fn, params = codelets[name]["fn"], codelets[name]["params"]
            step_mul = params.get("step_mul", 1.0)
            lam_scale = params.get("lambda", 1.0)
            x = x0.clone()
            for k in range(iters):
                grad = data_fid.grad(x, y, A)
                g_dir = g_fn(x, A, y, params) if "A" in g_fn.__code__.co_varnames else g_fn(x, params)
                x_pred = x - step_base * step_mul * (grad + lam_scale * g_dir)
                x = prior.prox(x_pred, sigma_denoiser=sigma_base, physics=A)
                traj_psnr[ai, k] = psnr(x, x_gt)
    return alg_names, traj_psnr


def compute_probabilities(traj_psnr, temperature=5.0, gumbel_noise=True, hard_winner=True):
    # Hard winner: add Gumbel noise, take argmax -> one-hot per iteration.
    A, K = traj_psnr.shape
    probs = torch.zeros((K, A), device=traj_psnr.device)
    for k in range(K):
        vals = traj_psnr[:, k]
        vals = torch.where(torch.isfinite(vals), vals, torch.full_like(vals, -1e9))
        if gumbel_noise:
            noise = -torch.log(-torch.log(torch.rand_like(vals) + 1e-8) + 1e-8)
            vals = vals + noise
        if hard_winner:
            idx = torch.argmax(vals)
            onehot = torch.zeros_like(vals)
            onehot[idx] = 1.0
            probs[k] = onehot
        else:
            v_max = torch.max(vals)
            weights = torch.exp((vals - v_max) / max(temperature, 1e-6))
            if torch.all(weights == 0):
                weights = torch.ones_like(weights)
            probs[k] = weights / weights.sum()
    return probs


def run_single(A, y, x0, x_gt, g_fn, params, iters, alpha, lambd):
    data_fid = dinv.optim.L2()
    prior = dinv.optim.PnP(
        denoiser=dinv.models.DnCNN(
            in_channels=3, out_channels=3, pretrained="download_lipschitz"
        ).to(x0.device)
    )
    norm = A.compute_norm(A.A_adjoint(y), tol=1e-3).item() + 1e-8
    step_base = alpha / norm
    sigma_base = lambd * step_base
    x = x0.clone()
    step_mul = params.get("step_mul", 1.0)
    lam_scale = params.get("lambda", 1.0)
    with torch.no_grad():
        for _ in range(iters):
            grad = data_fid.grad(x, y, A)
            g_dir = g_fn(x, A, y, params) if "A" in g_fn.__code__.co_varnames else g_fn(x, params)
            x_pred = x - step_base * step_mul * (grad + lam_scale * g_dir)
            x = prior.prox(x_pred, sigma_denoiser=sigma_base, physics=A)
    return psnr(x, x_gt).item()


def tune_params(device, size, iters, alpha, lambd):
    ops = make_fixed_operators(size, device)
    # Grids per codelet
    grids = {
        "ROBUST_DATA": [
            {"tau_rob": t, "p_rob": p, "lambda": lam, "step_mul": sm}
            for t in [0.02, 0.05, 0.1]
            for p in [0.8, 1.0, 1.2]
            for lam in [0.5, 1.0, 1.5]
            for sm in [0.5, 1.0, 1.5]
        ],
        "PRECOND_DATA": [
            {"coeffs": [s * c for c in ops["coeffs"]], "kernels": ops["kernels"], "lambda": lam, "step_mul": sm}
            for s in [0.5, 1.0, 1.5]
            for lam in [0.8, 1.0]
            for sm in [0.5, 1.0]
        ],
        "ARES": [
            {"alpha": a, "beta": b, "lambda": lam, "step_mul": sm}
            for a in [0.05, 0.1, 0.2, 0.3]
            for b in [0.0, 0.01, 0.05]
            for lam in [1.0, 1.5, 2.0]
            for sm in [0.75, 1.0, 1.25]
        ],
        "SPARSE_W": [
            {"W": ops["W"], "tau_sp": t, "lambda": lam, "step_mul": sm}
            for t in [0.003, 0.005, 0.01]
            for lam in [0.5, 1.0, 1.5]
            for sm in [0.5, 1.0, 1.5]
        ],
        "NULLSPACE": [
            {"U": ops["U"], "tau_n": t, "lambda": lam, "step_mul": sm}
            for t in [0.003, 0.005, 0.01]
            for lam in [0.5, 1.0, 1.5]
            for sm in [0.5, 1.0, 1.5]
        ],
    }
    fixed_params = {}

    # Use small subset for tuning
    imgs = list(load_images("/home/hdsp/Documents/Henry/pnp/data/places/test", size, device, limit=2))

    best_params = {}
    # First set all to defaults
    best_params.update({
        "ROBUST_DATA": {"tau_rob": 0.05, "p_rob": 1.0, "lambda": 1.0, "step_mul": 1.0},
        "PRECOND_DATA": {"kernels": ops["kernels"], "coeffs": ops["coeffs"], "lambda": 1.0, "step_mul": 1.0},
        "ARES": {"alpha": 0.1, "beta": 0.01, "lambda": 1.0, "step_mul": 1.0},
        "SPARSE_W": {"W": ops["W"], "tau_sp": 0.01, "lambda": 0.5, "step_mul": 0.5},
        "NULLSPACE": {"U": ops["U"], "tau_n": 0.01, "lambda": 0.5, "step_mul": 0.5},
    })

    def avg_psnr(fn, params):
        vals = []
        for _, x_gt in imgs:
            A = dinv.physics.SinglePixelCamera(
                m=int(0.1 * x_gt.numel() / x_gt.shape[0]),
                img_size=tuple(x_gt.shape[1:]),
                device=device,
            )
            y = A(x_gt)
            x0 = A.A_adjoint(y)
            vals.append(run_single(A, y, x0, x_gt, fn, params, iters, alpha, lambd))
        return sum(vals) / len(vals)

    # Tune grids
    for name, grid in grids.items():
        fn = {
            "ROBUST_DATA": codelet_robust_data,
            "PRECOND_DATA": codelet_precond_data,
            "ARES": codelet_ares,
            "TV": codelet_tv,
            "SPARSE_W": codelet_sparse_w,
            "NULLSPACE": codelet_nullspace,
            "L2": codelet_l2,
            "L1": codelet_l1,
            "GRAPH": codelet_graph,
            "NONLOCAL": codelet_nonlocal,
            "BQUAD": codelet_bquad,
        }[name]
        best = None
        best_val = -1e9
        for p in grid:
            v = avg_psnr(fn, p)
            if v > best_val:
                best_val = v
                best = p
        if best is None:
            best = grid[0]
            best_val = float("-inf")
        best_params[name] = best
        print(f"[tune] {name} best {best} psnr {best_val:.3f}")
    return best_params, ops


def run_probabilities(device, size, iters, alpha, lambd, params, ops):
    # DATA/grad + denoiser are always present; only these codelets compete.
    codelets = {
        "ROBUST_DATA": {"fn": codelet_robust_data, "params": params["ROBUST_DATA"]},
        "PRECOND_DATA": {"fn": codelet_precond_data, "params": params["PRECOND_DATA"]},
        "ARES": {"fn": codelet_ares, "params": params["ARES"]},
        "SPARSE_W": {"fn": lambda x, p: codelet_sparse_w(x, p), "params": params["SPARSE_W"]},
        "NULLSPACE": {"fn": lambda x, p: codelet_nullspace(x, p), "params": params["NULLSPACE"]},
    }

    results = []
    for idx, (path, x_gt) in enumerate(load_images("/home/hdsp/Documents/Henry/pnp/data/places/test", size, device, limit=10, offset=0)):
        A = dinv.physics.SinglePixelCamera(
            m=int(0.1 * x_gt.numel() / x_gt.shape[0]),
            img_size=tuple(x_gt.shape[1:]),
            device=device,
        )
        y = A(x_gt)
        x0 = A.A_adjoint(y)
        alg_names, traj_psnr = run_algorithms(A, y, x0, x_gt, codelets, iters, alpha, lambd)
        probs = compute_probabilities(traj_psnr)
        results.append({"path": str(path), "probs": probs.cpu().tolist()})
        print(f"[prob] image {idx} done")

    out = {"algorithms": alg_names, "results": results}
    out_path = Path("RESULTS/probabilities_tuned.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved tuned probabilities to {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = 64
    iters = 150
    alpha = 1.5
    lambd = 5e-4

    params, ops = tune_params(device, size, iters, alpha, lambd)
    run_probabilities(device, size, iters, alpha, lambd, params, ops)


if __name__ == "__main__":
    main()
