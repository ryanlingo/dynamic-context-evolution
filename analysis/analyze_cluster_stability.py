"""Analyze HDBSCAN clustering stability across seeds and min_cluster_size settings.

For each seed (42, 123, 456) and method (naive, dce), embeds all ideas with
MiniLM, runs HDBSCAN at multiple min_cluster_size values, and reports
cross-seed cluster counts + inter-cluster centroid distances.
"""

from __future__ import annotations

import json
from pathlib import Path

import hdbscan
import jsonlines
import numpy as np
from scipy.spatial.distance import cosine

SEEDS = [42, 123, 456]
METHODS = ["naive", "dce"]
MIN_CLUSTER_SIZES = [3, 5, 7, 10]
DEFAULT_MCS = 5


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def _data_dir(seed: int, domain: str) -> Path:
    """Return data directory for a given seed, handling seed 42 special case."""
    suffix = _domain_suffix(domain)
    # seed 42 may be at the unsuffixed path
    seed_path = Path(f"data/raw/exp2_comparison_{suffix}_seed{seed}")
    if seed_path.exists():
        return seed_path
    if seed == 42:
        unsuffixed = Path(f"data/raw/exp2_comparison_{suffix}")
        if unsuffixed.exists():
            return unsuffixed
    return seed_path  # return expected path even if missing


def load_ideas(method: str, seed: int, domain: str) -> list[dict]:
    """Load all batches for a method/seed from exp2 results."""
    data_dir = _data_dir(seed, domain)
    path = data_dir / method / "results.jsonl"
    if not path.exists():
        return []
    batches = []
    with jsonlines.open(str(path)) as reader:
        for batch in reader:
            batches.append(batch)
    return batches


def embed_all_ideas(batches: list[dict], model) -> np.ndarray:
    """Embed all ideas across all batches, returning a single array."""
    all_texts = []
    for batch in batches:
        for idea in batch["ideas"]:
            all_texts.append(f"{idea['name']}: {idea['description']}")
    if not all_texts:
        return np.array([])
    embeddings = model.encode(all_texts, normalize_embeddings=True)
    return np.array(embeddings)


def cluster_count(embeddings: np.ndarray, mcs: int) -> int:
    """Run HDBSCAN and return number of clusters (excluding noise)."""
    if len(embeddings) < mcs:
        return 0
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters


def mean_inter_cluster_centroid_distance(embeddings: np.ndarray, mcs: int) -> float:
    """Compute mean pairwise cosine distance between cluster centroids."""
    if len(embeddings) < mcs:
        return 0.0
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)
    unique_labels = sorted(set(labels) - {-1})
    if len(unique_labels) < 2:
        return 0.0

    centroids = []
    for label in unique_labels:
        mask = labels == label
        centroid = embeddings[mask].mean(axis=0)
        centroids.append(centroid)

    distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            distances.append(cosine(centroids[i], centroids[j]))

    return float(np.mean(distances)) if distances else 0.0


def run(domain: str = "sustainable packaging concepts"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Requires: pip install sentence-transformers")
        return

    suffix = _domain_suffix(domain)
    print(f"Loading MiniLM embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # --- Embed all ideas per seed × method ---
    embeddings_cache: dict[tuple[int, str], np.ndarray] = {}
    for seed in SEEDS:
        for method in METHODS:
            batches = load_ideas(method, seed, domain)
            if not batches:
                print(f"  WARNING: No data for {method} seed {seed}")
                continue
            embs = embed_all_ideas(batches, model)
            embeddings_cache[(seed, method)] = embs
            print(f"  Embedded {method}/seed{seed}: {len(embs)} ideas")

    # --- Table 1: Cross-seed cluster counts at default min_cluster_size ---
    print(f"\n{'='*60}")
    print(f"Cross-seed cluster counts (min_cluster_size={DEFAULT_MCS})")
    print(f"{'='*60}")
    header = f"{'Method':<10}"
    for seed in SEEDS:
        header += f"  {'Seed ' + str(seed):<12}"
    print(header)
    print("-" * 60)

    cluster_counts_table: dict[str, dict[int, int]] = {}
    for method in METHODS:
        cluster_counts_table[method] = {}
        row = f"{method:<10}"
        for seed in SEEDS:
            key = (seed, method)
            if key in embeddings_cache:
                nc = cluster_count(embeddings_cache[key], DEFAULT_MCS)
                cluster_counts_table[method][seed] = nc
                row += f"  {nc:<12}"
            else:
                row += f"  {'N/A':<12}"
        print(row)
    print(f"{'='*60}")

    # --- Table 2: Stability across min_cluster_size (averaged across seeds) ---
    print(f"\n{'='*60}")
    print("Stability across min_cluster_size (mean across seeds)")
    print(f"{'='*60}")
    header = f"{'Method':<10}"
    for mcs in MIN_CLUSTER_SIZES:
        header += f"  {'mcs=' + str(mcs):<10}"
    print(header)
    print("-" * 60)

    stability_table: dict[str, dict[int, float]] = {}
    for method in METHODS:
        stability_table[method] = {}
        row = f"{method:<10}"
        for mcs in MIN_CLUSTER_SIZES:
            counts = []
            for seed in SEEDS:
                key = (seed, method)
                if key in embeddings_cache:
                    counts.append(cluster_count(embeddings_cache[key], mcs))
            if counts:
                mean_count = np.mean(counts)
                stability_table[method][mcs] = float(mean_count)
                row += f"  {mean_count:<10.1f}"
            else:
                row += f"  {'N/A':<10}"
        print(row)
    print(f"{'='*60}")

    # --- Table 3: Mean inter-cluster centroid distance ---
    print(f"\n{'='*60}")
    print(f"Mean inter-cluster centroid distance (min_cluster_size={DEFAULT_MCS})")
    print(f"{'='*60}")
    print(f"{'Method':<10}  {'Mean Dist':<12}  {'Std':<12}")
    print("-" * 40)

    centroid_table: dict[str, dict[str, float]] = {}
    for method in METHODS:
        dists = []
        for seed in SEEDS:
            key = (seed, method)
            if key in embeddings_cache:
                d = mean_inter_cluster_centroid_distance(embeddings_cache[key], DEFAULT_MCS)
                dists.append(d)
        if dists:
            mean_d = float(np.mean(dists))
            std_d = float(np.std(dists))
            centroid_table[method] = {"mean": mean_d, "std": std_d}
            print(f"{method:<10}  {mean_d:<12.4f}  {std_d:<12.4f}")
        else:
            print(f"{method:<10}  {'N/A':<12}  {'N/A':<12}")
    print(f"{'='*60}")

    # --- Save results ---
    output_dir = Path(f"data/processed/cluster_stability_{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": domain,
        "seeds": SEEDS,
        "methods": METHODS,
        "min_cluster_sizes": MIN_CLUSTER_SIZES,
        "default_min_cluster_size": DEFAULT_MCS,
        "cross_seed_cluster_counts": {
            method: {str(seed): count for seed, count in counts.items()}
            for method, counts in cluster_counts_table.items()
        },
        "stability_across_mcs": {
            method: {str(mcs): val for mcs, val in vals.items()}
            for method, vals in stability_table.items()
        },
        "inter_cluster_centroid_distance": centroid_table,
    }

    output_path = output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="sustainable packaging concepts")
    args = parser.parse_args()
    run(domain=args.domain)
