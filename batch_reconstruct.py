import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from single_pixel_demo import (
    create_measurements,
    load_image,
    psnr,
    select_xp,
    to_numpy,
)
from tval3 import tval3_admm


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch single-pixel reconstructions with TVAL3-style ADMM."
    )
    p.add_argument("--input-dir", required=True, help="Directory with input images.")
    p.add_argument(
        "--output-dir",
        default="recon_out",
        help="Directory to store recon images and metrics.",
    )
    p.add_argument("--size", type=int, default=96, help="Resize images to size x size.")
    p.add_argument("--sampling", type=float, default=0.25, help="Sampling ratio m/n.")
    p.add_argument("--noise", type=float, default=0.01, help="Measurement noise std.")
    p.add_argument("--lam", type=float, default=0.1, help="TV weight.")
    p.add_argument("--rho", type=float, default=2.0, help="ADMM penalty.")
    p.add_argument("--iters", type=int, default=150, help="Outer ADMM iterations.")
    p.add_argument("--cg-iters", type=int, default=50, help="CG iterations per solve.")
    p.add_argument("--cg-tol", type=float, default=1e-5, help="CG tolerance.")
    p.add_argument("--tol", type=float, default=1e-4, help="ADMM stop tolerance.")
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Prefer GPU if available (requires CuPy).",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on number of images to process.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose reconstruction already exists in output-dir.",
    )
    p.add_argument(
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg"],
        help="Image file extensions to include.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    xp, device = select_xp(args.device)
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {e.lower() for e in args.extensions}
    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])
    if args.max_images:
        files = files[: args.max_images]
    if not files:
        raise SystemExit(f"No images found in {in_dir} with extensions {exts}.")

    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file",
                "psnr_db",
                "objective",
                "primal_res",
                "dual_res",
                "time_sec",
                "device",
            ]
        )
        for idx, img_path in enumerate(files, 1):
            out_img = out_dir / f"{img_path.stem}_recon.png"
            if args.skip_existing and out_img.exists():
                print(f"[{idx}/{len(files)}] {img_path.name}: skip (exists)")
                continue

            t0 = perf_counter()
            x_true_np = load_image(str(img_path), args.size)
            x_true = xp.asarray(x_true_np)
            A_mat, y, y_clean = create_measurements(
                x_true, args.sampling, args.noise, np.random.default_rng(idx), xp
            )
            A = lambda v: A_mat @ v
            AT = lambda v: A_mat.T @ v
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
                verbose=False,
                xp=xp,
            )

            x_true_cpu = to_numpy(x_true)
            x_rec_cpu = to_numpy(x_rec)
            psnr_db = psnr(x_true_cpu, x_rec_cpu)
            elapsed = perf_counter() - t0

            from PIL import Image

            Image.fromarray(np.uint8(np.clip(x_rec_cpu, 0, 1) * 255)).save(out_img)

            writer.writerow(
                [
                    img_path.name,
                    f"{psnr_db:.4f}",
                    f"{hist['objective'][-1]:.6e}",
                    f"{hist['primal_res'][-1]:.6e}",
                    f"{hist['dual_res'][-1]:.6e}",
                    f"{elapsed:.3f}",
                    device,
                ]
            )
            print(
                f"[{idx}/{len(files)}] {img_path.name}: PSNR {psnr_db:.2f} dB "
                f"obj={hist['objective'][-1]:.3e} time={elapsed:.2f}s device={device}"
            )


if __name__ == "__main__":
    main()
