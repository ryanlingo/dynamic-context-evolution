"""Step 9, Item 9: VTS probability distribution analysis.

Plots probability histograms for generated ideas and reports acceptance rate.
"""

from __future__ import annotations

from pathlib import Path

import jsonlines
import numpy as np
import matplotlib.pyplot as plt

from analysis.plot_utils import COLORS, setup_style, save_figure

METHODS = ["naive", "vts_only", "vts_dedup", "dce"]


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def load_probabilities(method: str, domain: str = "sustainable packaging concepts") -> list[float]:
    """Load all probability scores from a method's results."""
    suffix = _domain_suffix(domain)
    path = f"data/raw/exp2_comparison_{suffix}/{method}/results.jsonl"
    probs = []
    with jsonlines.open(path) as reader:
        for batch in reader:
            for idea in batch["ideas"]:
                probs.append(idea["probability"])
    return probs


def run(domain: str = "sustainable packaging concepts"):
    suffix = _domain_suffix(domain)
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    print(f"\nDomain: {domain}")
    print("=" * 60)
    print(f"{'Method':<15} {'N ideas':<10} {'Mean P':<10} {'Median P':<10} {'% below 0.10':<15}")
    print("-" * 60)

    for i, method in enumerate(METHODS):
        probs = load_probabilities(method, domain=domain)
        probs_arr = np.array(probs)

        below_thresh = np.mean(probs_arr < 0.10)
        print(f"{method:<15} {len(probs):<10} {probs_arr.mean():<10.3f} {np.median(probs_arr):<10.3f} {below_thresh:<15.1%}")

        ax = axes[i]
        ax.hist(probs_arr, bins=20, range=(0, 1), color=COLORS.get(method, "#999"),
                edgecolor="white", alpha=0.8)
        ax.axvline(x=0.10, color="red", linestyle="--", alpha=0.7, label="Threshold (0.10)")
        ax.set_title(method)
        ax.set_xlabel("Self-assessed probability")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    print("=" * 60)

    fig.suptitle("VTS Probability Distribution by Method", fontsize=14)
    fig.tight_layout()
    save_figure(fig, f"vts_distribution_{suffix}")

    output_dir = Path(f"data/processed/vts_analysis_{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        probs = load_probabilities(method, domain=domain)
        np.save(output_dir / f"{method}_probs.npy", np.array(probs))
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="sustainable packaging concepts")
    args = parser.parse_args()
    run(domain=args.domain)
