"""Step 4: Compare naive vs DCE on batch novelty and cluster count.

Two-panel figure showing how naive generation collapses while DCE
maintains diversity over 200 batches.
"""

from __future__ import annotations

from pathlib import Path

import hdbscan
import jsonlines
import numpy as np
import matplotlib.pyplot as plt

from analysis.plot_utils import COLORS, LABELS, setup_style, save_figure


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def load_batches(method: str, domain: str) -> list[dict]:
    suffix = _domain_suffix(domain)
    path = Path(f"data/raw/exp2_comparison_{suffix}/{method}/results.jsonl")
    batches = []
    with jsonlines.open(str(path)) as reader:
        for batch in reader:
            batches.append(batch)
    return batches


def embed_batches(batches: list[dict], model) -> list[np.ndarray]:
    all_batch_embs = []
    for batch in batches:
        texts = [f"{idea['name']}: {idea['description']}" for idea in batch["ideas"]]
        if not texts:
            all_batch_embs.append(np.array([]).reshape(0, 384))
            continue
        embs = model.encode(texts, normalize_embeddings=True)
        all_batch_embs.append(np.array(embs))
    return all_batch_embs


def compute_batch_novelty_series(batch_embs: list[np.ndarray]) -> list[float]:
    """Compute batch novelty for each batch (avg min cosine distance to prior)."""
    novelties = []
    all_prior: list[np.ndarray] = []

    for embs in batch_embs:
        if len(embs) == 0:
            novelties.append(0.0)
            continue

        if not all_prior:
            novelties.append(1.0)
        else:
            prior = np.concatenate(all_prior)
            sims = embs @ prior.T
            min_distances = 1.0 - np.max(sims, axis=1)
            novelties.append(float(np.mean(min_distances)))

        all_prior.append(embs)

    return novelties


def compute_cumulative_cluster_counts(batch_embs: list[np.ndarray]) -> list[int]:
    """Compute HDBSCAN cluster count after each batch (cumulative)."""
    all_embs: list[np.ndarray] = []
    counts = []

    for embs in batch_embs:
        if len(embs) > 0:
            all_embs.append(embs)

        if all_embs:
            stacked = np.concatenate(all_embs)
            if len(stacked) >= 5:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=5, metric="euclidean"
                )
                labels = clusterer.fit_predict(stacked)
                n_clusters = len(set(labels) - {-1})
            else:
                n_clusters = 0
        else:
            n_clusters = 0
        counts.append(n_clusters)

    return counts


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
        print(f"Processing {method}...")
        batches = load_batches(method, domain=domain)
        batch_embs = embed_batches(batches, model)
        batch_numbers = [b["batch_number"] for b in batches]

        novelties = compute_batch_novelty_series(batch_embs)
        cluster_counts = compute_cumulative_cluster_counts(batch_embs)

        methods_data[method] = {
            "batch_numbers": batch_numbers,
            "novelties": novelties,
            "cluster_counts": cluster_counts,
        }

    # ---- Two-panel figure ----
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: batch novelty
    for method in ["naive", "dce"]:
        d = methods_data[method]
        ax1.plot(
            d["batch_numbers"],
            d["novelties"],
            color=COLORS[method],
            label=LABELS[method],
        )
    ax1.set_xlabel("Batch Number")
    ax1.set_ylabel("Batch Novelty")
    ax1.set_title("Batch Novelty Over Time")
    ax1.legend()

    # Right panel: cumulative cluster count
    for method in ["naive", "dce"]:
        d = methods_data[method]
        ax2.plot(
            d["batch_numbers"],
            d["cluster_counts"],
            color=COLORS[method],
            label=LABELS[method],
        )
    ax2.set_xlabel("Batch Number")
    ax2.set_ylabel("Cumulative HDBSCAN Cluster Count")
    ax2.set_title("Cluster Count Over Time")
    ax2.legend()

    fig.tight_layout()
    save_figure(fig, "collapse_comparison")
    print("Saved collapse_comparison figure.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="sustainable packaging concepts")
    args = parser.parse_args()
    run(domain=args.domain)
