# Baseline (fixed A, general denoiser)

- Forward model: `y = A x`, compression 0.1 (m = 1228, n = 12288). Matrix saved locally at `RESULTS/AA.npy` (frozen). Not tracked in git.
- Denoiser: `RESULTS/generaldenoiser.pth`, sigma=0.02 (fixed). For baseline, only this denoiser is used.
- PnP-PSG settings: 150 iterations; step size `eta = 0.9 / ||A||_2^2` (power iteration); update `x_{k+1} = D(x_k - eta * A^T (A x_k - y))`.
- Image size: 64×64 RGB. Dataset: first 50 images in `/home/hdsp/Documents/Henry/pnp/data/places/test` (Places365_val_00000001.jpg … 00000050.jpg).
- Result: average PSNR = **24.80 dB** over 50 images. Full per-image log saved locally at `RESULTS/baseline_eval_50.txt`.

Reproduce:
```
python pnp_psg_baseline.py \
  --img-dir /home/hdsp/Documents/Henry/pnp/data/places/test \
  --max-images 50 \
  --iters 150 \
  --sigma 0.02 \
  --denoiser-path RESULTS/generaldenoiser.pth \
  --device cpu
```

Notes:
- `RESULTS/` is ignored from git; keep `AA.npy`, `generaldenoiser.pth`, and the eval log there.
- The same fixed A should be used in all simulations for comparability.
