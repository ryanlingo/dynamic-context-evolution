"""Generate delta sensitivity tradeoff figure.

Dual-axis plot showing accepted ideas vs EDV retention as delta varies.
Data from Table 6 in the paper (delta sensitivity analysis).

v14: Discrete markers instead of connected line for EDV; horizontal
reference at 1000 ideas; value labels on markers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis.plot_utils import setup_style, COLORS


# Data from Table 6 (delta sensitivity)
DELTAS = [0.80, 0.85, 0.90, 0.95]
ACCEPTED = [869, 950, 994, 999]
EDV_RETENTION = [27.5, 28.5, 21.4, 25.5]
COLLAPSE = [0.0, 0.0, 0.0, 2.5]


def run():
    setup_style()
    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    # Left y-axis: accepted ideas (bars)
    bar_color = "#56B4E9"  # sky blue
    bar_width = 0.025
    bars = ax1.bar(DELTAS, ACCEPTED, width=bar_width, color=bar_color,
                   alpha=0.6, label="Accepted Ideas", zorder=2)
    ax1.set_xlabel(r"Deduplication Threshold ($\delta$)")
    ax1.set_ylabel("Accepted Ideas (out of 1,000)", color=bar_color)
    ax1.tick_params(axis="y", labelcolor=bar_color)
    ax1.set_ylim(800, 1050)
    ax1.set_xticks(DELTAS)
    ax1.set_xticklabels([f"{d:.2f}" for d in DELTAS])

    # Horizontal reference line at 1000 ideas
    ax1.axhline(y=1000, color=bar_color, linestyle=":", linewidth=1.0, alpha=0.5, zorder=1)
    ax1.text(0.805, 1003, "1,000 baseline", fontsize=8, color=bar_color, alpha=0.7)

    # Right y-axis: EDV retention (discrete markers with value labels, no connecting line)
    ax2 = ax1.twinx()
    edv_color = COLORS["dce"]  # vermillion
    ax2.scatter(DELTAS, EDV_RETENTION, color=edv_color, marker="o",
                s=80, label="EDV Retention (%)", zorder=5)
    ax2.set_ylabel("EDV Retention (%)", color=edv_color)
    ax2.tick_params(axis="y", labelcolor=edv_color)
    ax2.set_ylim(15, 35)

    # Value labels on each marker
    for d, edv in zip(DELTAS, EDV_RETENTION):
        ax2.annotate(
            f"{edv:.1f}%",
            xy=(d, edv),
            xytext=(0, 8),
            textcoords="offset points",
            fontsize=9,
            color=edv_color,
            ha="center",
            fontweight="bold",
        )

    # Re-enable right spine for dual axis
    ax2.spines["right"].set_visible(True)

    # Annotate collapse at delta=0.95
    collapse_idx = 3  # delta=0.95
    ax2.annotate(
        f"2.5% collapse",
        xy=(DELTAS[collapse_idx], EDV_RETENTION[collapse_idx]),
        xytext=(DELTAS[collapse_idx] - 0.06, EDV_RETENTION[collapse_idx] + 5),
        fontsize=10,
        color="#CC79A7",  # reddish-purple (Okabe-Ito), distinct from vermillion EDV line
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#CC79A7", lw=1.5),
        zorder=5,
    )

    # Safe zone shading (delta <= 0.90) — full height, visible
    ax1.axvspan(0.78, 0.905, alpha=0.15, color="#009E73", zorder=0)
    ax1.text(0.84, 815, "safe zone ($\\delta \\leq 0.90$)", fontsize=9,
             ha="center", color="#009E73", fontweight="bold", fontstyle="italic")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title(r"Deduplication Threshold ($\delta$) Tradeoff")
    fig.tight_layout()

    # Save
    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "delta_tradeoff.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    run()
