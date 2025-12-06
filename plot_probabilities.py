import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot averaged win probabilities.")
    parser.add_argument("--file", default="RESULTS/probabilities.json", help="Path to probabilities JSON.")
    parser.add_argument("--out", default=None, help="Output image path (png).")
    args = parser.parse_args()

    data_path = Path(args.file)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    algs = data["algorithms"]
    # Aggregate probabilities across images by averaging
    prob_acc = None
    count = 0
    for item in data["results"]:
        probs = np.array(item["probs"], dtype=np.float32)  # shape (K, A)
        if prob_acc is None:
            prob_acc = probs
        else:
            prob_acc += probs
        count += 1
    prob_mean = prob_acc / max(count, 1)

    plt.figure(figsize=(10, 6))
    plt.imshow(prob_mean, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Win probability")
    plt.yticks(np.linspace(0, prob_mean.shape[0] - 1, 6))
    plt.xticks(np.arange(len(algs)), algs, rotation=45, ha="right")
    plt.xlabel("Algorithm")
    plt.ylabel("Iteration")
    plt.title("Mean win probability per iteration (averaged over images)")
    out_path = Path(args.out) if args.out else data_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
