Single-pixel reconstruction with TVAL3-style ADMM
================================================

This project shows how to solve the single-pixel measurement problem `y = A x + w` with a total variation (TV) prior, using an augmented Lagrangian / ADMM scheme inspired by TVAL3.

What's inside
-------------
- `single_pixel_demo.py`: end-to-end example that generates a measurement matrix, simulates noisy measurements, and reconstructs with TV minimization.
- `tval3.py`: small TV-ADMM solver (isotropic TV) used by the demo.
- `batch_reconstruct.py`: batch runner for a directory of images; saves reconstructions and metrics.
- `requirements.txt`: runtime dependencies (NumPy, SciPy, Matplotlib, Pillow).
- `.venv/`: created locally for convenience; you can recreate it or use another environment.

Quickstart
----------
```bash
cd /home/hdsp/Desktop/codexPre
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python single_pixel_demo.py --sampling 0.25 --noise 0.01 --save-fig recon.png
# Prefer GPU if you have CuPy and a CUDA device:
# .venv/bin/pip install cupy-cuda12x  # pick the wheel matching your CUDA/toolkit
# .venv/bin/python single_pixel_demo.py --device gpu ...
# Batch run over a folder:
# .venv/bin/python batch_reconstruct.py --input-dir trainingimagens --output-dir recon_out --device auto --skip-existing
```

You can pass `--image path/to/img.png` to reconstruct a custom grayscale image (it will be resized to the requested `--size`). If omitted, a simple synthetic target is used. Use `--device gpu` to force GPU (requires CuPy and an available CUDA device); otherwise `--device auto` defaults to GPU if present or CPU.

Solver details
--------------
We minimize

```
min_x 0.5 ||A x - y||_2^2 + λ ||∇x||_{2,1}
```

with ADMM:
- `x`-update solves `(AᵀA + ρ ∇ᵀ∇) x = Aᵀ y + ρ ∇ᵀ(d - u)` via conjugate gradients (no explicit matrices).
- `d`-update applies isotropic soft-thresholding of the image gradients.
- `u` is the scaled dual variable.

Tune `--lam` (TV weight) and `--rho` (augmented Lagrangian penalty) to balance data fidelity vs. smoothness; `--sampling` controls how many measurements are taken (`m = sampling * n`).

Outputs
-------
- Console metrics: relative residuals, objective value, PSNR against the ground truth when available.
- A figure with the ground-truth and reconstructed images (saved to `recon.png` by default; override with `--save-fig`).

Notes
-----
- Measurements use a Rademacher (±1) sensing matrix normalized by `1/sqrt(m)`.
- Noise is additive white Gaussian with standard deviation `--noise`.
- The implementation is intentionally compact for clarity, not optimized for very large images. Increase `--cg-iters` for tighter inner solves.
