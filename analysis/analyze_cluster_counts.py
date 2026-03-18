"""Step 3: HDBSCAN cluster counts for all exp2 methods.

Embeds ideas with MiniLM, computes cumulative cluster count at batch
boundaries (every 10 batches), prints a summary table, and generates
a line plot of cluster count over batches.
"""

from __future__ import annotations

from pathlib import Path

import hdbscan
import jsonlines
import numpy as np
import matplotlib.pyplot as plt

from analysis.plot_utils import COLORS, LABELS, setup_style, save_figure

METHODS = [
    "naive", "vts_only", "vts_dedup", "dce",
    "dedup_only", "prompt_evo_only", "prompt_evo_dedup",
]

CHECKPOINTS = [50, 100, 200]


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def load_batches(method: str, domain: str) -> list[dict]:
    suffix = _domain_suffix(domain)
    path = Path(f"data/raw/exp2_comparison_{suffix}/{method}/results.jsonl")
    if not path.exists():
        return []
    batches = []
    with jsonlines.open(str(path)) as reader:
        for batch in reader:
            batches.append(batch)
    return batches


def embed_batches(batches: list[dict], model) -> list[np.ndarray]:
    """Return a list of embedding arrays, one per batch."""
    all_batch_embs = []
    for batch in batches:
        texts = [f"{idea['name']}: {idea['description']}" for idea in batch["ideas"]]
        if not texts:
            all_batch_embs.append(np.array([]).reshape(0, 384))
            continue
        embs = model.encode(texts, normalize_embeddings=True)
        all_batch_embs.append(np.array(embs))
    return all_batch_embs


def cumulative_cluster_counts(
    batch_embs: list[np.ndarray],
    every: int = 10,
) -> tuple[list[int], list[int]]:
    """Compute HDBSCAN cluster count at every N-th batch boundary.

    Returns (batch_numbers, cluster_counts).
    """
    all_embs: list[np.ndarray] = []
    batch_numbers = []
    counts = []

    for i, embs in enumerate(batch_embs):
        batch_num = i + 1
        if len(embs) > 0:
            all_embs.append(embs)

        if batch_num % every == 0 or batch_num == len(batch_embs):
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
            batch_numbers.append(batch_num)
            counts.append(n_clusters)

    return batch_numbers, counts


def run(domain: str = "sustainable packaging concepts"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Requires: pip install sentence-transformers")
        return

    suffix = _domain_suffix(domain)
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    results: dict[str, dict] = {}

    for method in METHODS:
        batches = load_batches(method, domain=domain)
        if not batches:
            print(f"Skipping {method} (no data)")
            continue

        print(f"Processing {method} ({len(batches)} batches)...")
        batch_embs = embed_batches(batches, model)
        batch_numbers, counts = cumulative_cluster_counts(batch_embs, every=10)
        results[method] = {
            "batch_numbers": batch_numbers,
            "counts": counts,
        }

    # ---- Print table ----
    print("\n" + "=" * 65)
    header = f"{'Method':<25}"
    for cp in CHECKPOINTS:
        header += f" {'Clusters@' + str(cp):<15}"
    print(header)
    print("-" * 65)

    for method, r in results.items():
        label = LABELS.get(method, method)
        row = f"{label:<25}"
        bn_to_count = dict(zip(r["batch_numbers"], r["counts"]))
        for cp in CHECKPOINTS:
            val = bn_to_count.get(cp, "—")
            row += f" {str(val):<15}"
        print(row)
    print("=" * 65)

    # ---- Plot ----
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, r in results.items():
        ax.plot(
            r["batch_numbers"],
            r["counts"],
            color=COLORS.get(method, "#000000"),
            label=LABELS.get(method, method),
        )

    ax.set_xlabel("Batch Number")
    ax.set_ylabel("Cumulative HDBSCAN Cluster Count")
    ax.set_title("Cluster Count Over Time (All Methods)")
    ax.legend()
    save_figure(fig, "cluster_counts_all_methods")

    # ---- Save processed results ----
    output_dir = Path(f"data/processed/cluster_counts_{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for method, r in results.items():
        np.savez(
            output_dir / f"{method}_cluster_counts.npz",
            batch_numbers=r["batch_numbers"],
            counts=r["counts"],
        )
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="sustainable packaging concepts")
    args = parser.parse_args()
    run(domain=args.domain)
