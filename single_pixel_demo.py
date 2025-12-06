import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

try:
    import cupy as cp  # type: ignore

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

from tval3 import tval3_admm


def synthetic_target(size: int) -> np.ndarray:
    """Simple piecewise-constant phantom."""
    img = np.zeros((size, size), dtype=np.float64)
    s = size
    img[s // 8 : s // 2, s // 8 : s // 2] = 0.95
    img[5 * s // 8 : 7 * s // 8, 5 * s // 8 : 7 * s // 8] = 0.6
    img[s // 3 : s // 2, s // 5 : 2 * s // 5] = 0.35
    img[2 * s // 3 : 3 * s // 4, s // 3 : s // 2] = 0.75
    return img


def load_image(path: str | None, size: int) -> np.ndarray:
    if path is None:
        return synthetic_target(size)
    img = Image.open(path).convert("L").resize((size, size))
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr


def build_sensing_matrix(m: int, n: int, rng: np.random.Generator, xp):
    # Rademacher entries (±1) scaled so columns have unit variance.
    A = rng.choice([-1.0, 1.0], size=(m, n)) / math.sqrt(m)
    return xp.asarray(A)


def create_measurements(
    x_true,
    sampling: float,
    noise_std: float,
    rng: np.random.Generator,
    xp,
):
    h, w = x_true.shape
    n = h * w
    m = max(1, int(sampling * n))
    A = build_sensing_matrix(m, n, rng, xp)
    y_clean = A @ x_true.ravel()
    y = y_clean + noise_std * xp.asarray(rng.standard_normal(m))
    return A, y, y_clean


def psnr(gt: np.ndarray, rec: np.ndarray):
    mse = np.mean((gt - rec) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def to_numpy(arr):
    if HAS_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def select_xp(device: str):
    if device == "cpu":
        return np, "cpu"
    if device == "gpu":
        if HAS_CUPY:
            try:
                n_dev = cp.cuda.runtime.getDeviceCount()
            except Exception:
                n_dev = 0
            if n_dev > 0:
                return cp, "gpu"
        raise RuntimeError("GPU requested but CuPy/GPU device is not available.")
    # auto: prefer GPU if present
    if HAS_CUPY:
        try:
            n_dev = cp.cuda.runtime.getDeviceCount()
        except Exception:
            n_dev = 0
        if n_dev > 0:
            return cp, "gpu"
    return np, "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-pixel reconstruction with TVAL3-style ADMM."
    )
    parser.add_argument("--image", type=str, default=None, help="Path to input image.")
    parser.add_argument("--size", type=int, default=64, help="Reconstruction size.")
    parser.add_argument(
        "--sampling",
        type=float,
        default=0.25,
        help="Sampling ratio m/n (0-1].",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.01,
        help="Std of AWGN added to measurements.",
    )
    parser.add_argument("--lam", type=float, default=0.1, help="TV weight λ.")
    parser.add_argument("--rho", type=float, default=2.0, help="Penalty ρ.")
    parser.add_argument(
        "--iters", type=int, default=200, help="Max outer ADMM iterations."
    )
    parser.add_argument(
        "--cg-iters", type=int, default=60, help="CG iterations for x-update."
    )
    parser.add_argument(
        "--cg-tol", type=float, default=1e-5, help="CG tolerance for x-update."
    )
    parser.add_argument("--tol", type=float, default=1e-4, help="ADMM stop tolerance.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Computation device (GPU requires CuPy).",
    )
    parser.add_argument(
        "--save-fig",
        type=str,
        default="recon.png",
        help="Where to save the comparison figure.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-iteration logging."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    sampling = float(np.clip(args.sampling, 1e-3, 1.0))

    xp, device = select_xp(args.device)
    x_true_np = load_image(args.image, args.size)
    x_true = xp.asarray(x_true_np)
    A_mat, y, y_clean = create_measurements(x_true, sampling, args.noise, rng, xp)
    A = lambda v: A_mat @ v
    AT = lambda v: A_mat.T @ v

    print(
        f"Target: {x_true.shape}, measurements: {len(y)} "
        f"(sampling={sampling:.3f}), noise std={args.noise}, device={device}"
    )
    x_rec, hist = tval3_admm(
        A=A,
        AT=AT,
        y=y,
        shape=x_true.shape,
        lam=args.lam,
        rho=args.rho,
        max_iters=args.iters,
        cg_iters=args.cg_iters,
        cg_tol=args.cg_tol,
        tol=args.tol,
        verbose=not args.quiet,
        xp=xp,
    )

    x_true_cpu = to_numpy(x_true)
    x_rec_cpu = to_numpy(x_rec)
    y_cpu = to_numpy(y)
    y_clean_cpu = to_numpy(y_clean)

    rec_psnr = psnr(x_true_cpu, x_rec_cpu)
    data_mse = np.mean((y_cpu - y_clean_cpu) ** 2)
    print(f"Reconstruction PSNR: {rec_psnr:.2f} dB")
    print(
        f"Final objective={hist['objective'][-1]:.4e}, "
        f"primal={hist['primal_res'][-1]:.3e}, dual={hist['dual_res'][-1]:.3e}"
    )
    print(f"Measurement MSE (noise level): {data_mse:.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(x_true_cpu, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Ground truth")
    axes[0].axis("off")
    axes[1].imshow(np.clip(x_rec_cpu, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("TVAL3 reconstruction")
    axes[1].axis("off")
    fig.tight_layout()

    out_path = args.save_fig
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved comparison figure to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
