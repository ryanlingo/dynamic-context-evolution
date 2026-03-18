"""Generate two-panel EDV figure: methods without dedup (left) vs with dedup (right).

Left panel: naive, vts_only, prompt_evo_only + DCE as gray dashed reference.
Right panel: vts_dedup, dce, dedup_only, prompt_evo_dedup (DCE thick solid).

v14: Apply 10-batch rolling average smoothing; show raw as faint background.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis.plot_utils import COLORS, LABELS, setup_style


DATA_DIR = Path("data/processed/exp2_comparison_sustainable_packaging_concepts")

LEFT_METHODS = ["naive", "vts_only", "prompt_evo_only"]
RIGHT_METHODS = ["vts_dedup", "dedup_only", "prompt_evo_dedup", "dce"]

WINDOW = 10  # rolling average window


def rolling_avg(arr: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Compute centered rolling average with edge padding."""
    kernel = np.ones(window) / window
    # Pad edges to keep same length
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_metrics(method: str) -> dict:
    path = DATA_DIR / f"{method}_metrics.npz"
    d = np.load(str(path), allow_pickle=True)
    return {
        "batch_numbers": d["batch_numbers"],
        "edvs": d["edvs"],
    }


def run():
    setup_style()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # Load all data
    data = {}
    all_methods = set(LEFT_METHODS + RIGHT_METHODS + ["dce"])
    for method in all_methods:
        data[method] = load_metrics(method)

    # ---- Left panel: methods without dedup ----
    for method in LEFT_METHODS:
        d = data[method]
        batches = d["batch_numbers"]
        raw = d["edvs"]
        smooth = rolling_avg(raw)

        # Faint raw values
        ax_left.plot(batches, raw, color=COLORS[method], alpha=0.1, linewidth=0.8)
        # Smoothed line
        ax_left.plot(batches, smooth, color=COLORS[method], label=LABELS[method],
                     linewidth=1.5)

    # DCE as gray dashed reference in left panel
    dce = data["dce"]
    dce_smooth = rolling_avg(dce["edvs"])
    ax_left.plot(dce["batch_numbers"], dce["edvs"], color="gray", alpha=0.1, linewidth=0.8)
    ax_left.plot(
        dce["batch_numbers"], dce_smooth,
        color="gray",
        linestyle="--",
        linewidth=2.5,
        label=f"{LABELS['dce']} (ref.)",
        zorder=10,
    )

    ax_left.set_xlabel("Batch Number")
    ax_left.set_ylabel("EDV (Effective Diversity Volume)")
    ax_left.set_title("Without Deduplication")
    ax_left.legend(fontsize=9)

    # ---- Right panel: methods with dedup ----
    for method in RIGHT_METHODS:
        d = data[method]
        batches = d["batch_numbers"]
        raw = d["edvs"]
        smooth = rolling_avg(raw)

        # Faint raw values
        ax_right.plot(batches, raw, color=COLORS[method], alpha=0.1, linewidth=0.8)

        if method == "dce":
            ax_right.plot(batches, smooth, color=COLORS[method],
                          label=LABELS[method], linewidth=2.5)
        else:
            ax_right.plot(batches, smooth, color=COLORS[method],
                          label=LABELS[method], linewidth=1.5)

    ax_right.set_xlabel("Batch Number")
    ax_right.set_title("With Deduplication")
    ax_right.legend(fontsize=9)

    fig.tight_layout()

    # Save
    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp2_edv_two_panel.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    run()
