"""Step 8: UMAP visualization of naive vs DCE embedding spaces.

Side-by-side scatter plots showing how ideas cluster in 2D space,
colored by batch number to reveal temporal patterns.

v14: Diverging colormap (blue=early, red=late), density contours for
early/late subsets to show overlap vs separation.
"""

from __future__ import annotations

from pathlib import Path

import jsonlines
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import umap
from scipy.ndimage import gaussian_filter

from analysis.plot_utils import setup_style


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def load_ideas_with_batch(method: str, domain: str) -> tuple[list[dict], list[int]]:
    """Load all ideas and their batch numbers."""
    suffix = _domain_suffix(domain)
    path = Path(f"data/raw/exp2_comparison_{suffix}/{method}/results.jsonl")
    ideas = []
    batch_numbers = []
    with jsonlines.open(str(path)) as reader:
        for batch in reader:
            bn = batch["batch_number"]
            for idea in batch["ideas"]:
                ideas.append(idea)
                batch_numbers.append(bn)
    return ideas, batch_numbers


def add_density_contours(ax, points, color, alpha=0.4, levels=4):
    """Add KDE-based density contours for a subset of points."""
    from scipy.stats import gaussian_kde
    if len(points) < 10:
        return
    try:
        kde = gaussian_kde(points.T, bw_method=0.3)
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        z = kde(positions).reshape(xx.shape)
        ax.contour(xx, yy, z, levels=levels, colors=color, alpha=alpha,
                   linewidths=1.2)
    except np.linalg.LinAlgError:
        pass


def run(domain: str = "sustainable packaging concepts"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Requires: pip install sentence-transformers")
        return

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    methods_data = {}
    for method in ["naive", "dce"]:
        print(f"Loading {method} ideas...")
        ideas, batch_numbers = load_ideas_with_batch(method, domain=domain)
        texts = [f"{idea['name']}: {idea['description']}" for idea in ideas]
        print(f"  Embedding {len(texts)} ideas...")
        embeddings = model.encode(texts, normalize_embeddings=True)
        methods_data[method] = {
            "embeddings": np.array(embeddings),
            "batch_numbers": np.array(batch_numbers),
        }

    # UMAP projection — fit on combined data for comparable axes
    print("Running UMAP projection...")
    all_embs = np.concatenate([
        methods_data["naive"]["embeddings"],
        methods_data["dce"]["embeddings"],
    ])
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    all_projected = reducer.fit_transform(all_embs)

    n_naive = len(methods_data["naive"]["embeddings"])
    methods_data["naive"]["projected"] = all_projected[:n_naive]
    methods_data["dce"]["projected"] = all_projected[n_naive:]

    # ---- Diverging colormap: blue (early) -> white (mid) -> red (late) ----
    cmap = plt.cm.RdYlBu_r  # blue=low batch, red=high batch
    max_batch = max(
        methods_data["naive"]["batch_numbers"].max(),
        methods_data["dce"]["batch_numbers"].max(),
    )
    norm = mcolors.Normalize(vmin=0, vmax=max_batch)

    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    from analysis.plot_utils import LABELS

    for ax, method in [(ax1, "naive"), (ax2, "dce")]:
        d = methods_data[method]
        proj = d["projected"]
        bn = d["batch_numbers"]

        sc = ax.scatter(
            proj[:, 0],
            proj[:, 1],
            c=bn,
            cmap=cmap,
            norm=norm,
            s=8,
            alpha=0.7,
            edgecolors="none",
        )

        # Compute centroids for early (1-50) and late (151-200) batches
        early_mask = bn <= 50
        late_mask = bn >= 151

        early_centroid = proj[early_mask].mean(axis=0)
        late_centroid = proj[late_mask].mean(axis=0)

        # Plot centroid stars
        ax.scatter(*early_centroid, marker="*", s=300, color="#0072B2",
                   edgecolors="white", linewidths=0.8, zorder=10,
                   label="Early centroid (1\u201350)")
        ax.scatter(*late_centroid, marker="*", s=300, color="#D55E00",
                   edgecolors="white", linewidths=0.8, zorder=10,
                   label="Late centroid (151\u2013200)")

        # Compute and annotate centroid distance
        dist = np.linalg.norm(late_centroid - early_centroid)
        mid = (early_centroid + late_centroid) / 2
        ax.annotate(
            f"d = {dist:.2f}",
            xy=mid,
            fontsize=9,
            fontweight="bold",
            color="black",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
            zorder=11,
        )

        ax.set_title(LABELS.get(method, method))
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        # Add density contours for early and late batches
        add_density_contours(ax, proj[early_mask], color="#0072B2", alpha=0.5, levels=3)
        add_density_contours(ax, proj[late_mask], color="#D55E00", alpha=0.5, levels=3)

        ax.legend(fontsize=7, loc="lower right")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label("Batch number")

    fig.tight_layout()

    # Save as PDF only (scatter plots don't convert well to TikZ)
    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "embedding_space.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="sustainable packaging concepts")
    args = parser.parse_args()
    run(domain=args.domain)
