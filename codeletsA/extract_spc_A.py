"""Extract dense A matrix from DeepInv SinglePixelCamera by probing basis vectors in batches."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch

# Add DeepInv path
sys.path.append('/home/hdsp/Documents/Henry/pnp')
import deepinv as dinv  # type: ignore


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = 1228
    img_size = (3, 64, 64)
    n = img_size[0] * img_size[1] * img_size[2]
    physics = dinv.physics.SinglePixelCamera(m=m, img_size=img_size, device=device)
    batch = 64
    A = np.zeros((m, n), dtype=np.float32)
    cols = np.arange(n)
    # Process in batches of basis vectors
    for start in range(0, n, batch):
        end = min(start + batch, n)
        bsz = end - start
        x = torch.zeros((bsz, *img_size), device=device)
        for i, col in enumerate(cols[start:end]):
            c = col // (64 * 64)
            rem = col % (64 * 64)
            h = rem // 64
            w = rem % 64
            x[i, c, h, w] = 1.0
        with torch.no_grad():
            y = physics(x)  # expect [B, m]
        y_flat = y.view(bsz, -1).cpu().numpy()  # [B, m]
        for i in range(bsz):
            A[:, start + i] = y_flat[i]
        if (start // batch) % 10 == 0:
            print(f"Processed columns {start} to {end} / {n}")
    out = Path('RESULTS/AA.npy')
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, A)
    print('Saved extracted A to', out)


if __name__ == '__main__':
    main()
