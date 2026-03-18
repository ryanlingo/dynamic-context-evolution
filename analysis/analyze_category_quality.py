"""Category quality analysis: evaluate consistency and distinctness of model-assigned categories.

Loads DCE results, extracts category fields, and computes:
- Category distribution and entropy (normalized)
- Intra-category cosine similarity (consistency)
- Inter-category centroid cosine similarity (distinctness)

Uses MiniLM-L6-v2 for embedding (independent of generation-time embeddings).
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import jsonlines
import numpy as np


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def load_dce_ideas(domain: str) -> list[dict]:
    """Load all accepted ideas from DCE method."""
    suffix = _domain_suffix(domain)
    # Try default path first, then seed42 variant
    path = Path(f"data/raw/exp2_comparison_{suffix}/dce/results.jsonl")
    if not path.exists():
        path = Path(f"data/raw/exp2_comparison_{suffix}_seed42/dce/results.jsonl")
    ideas = []
    with jsonlines.open(str(path)) as reader:
        for batch in reader:
            for idea in batch["ideas"]:
                ideas.append(idea)
    return ideas


def compute_entropy(counts: list[int]) -> tuple[float, float, float]:
    """Compute distribution entropy, max entropy, and normalized entropy.

    Returns (H, H_max, H_norm) where H_norm = H / H_max.
    """
    total = sum(counts)
    probs = np.array([c / total for c in counts])
    # Filter out zero-probability entries to avoid log(0)
    probs = probs[probs > 0]
    H = -np.sum(probs * np.log2(probs))
    H_max = np.log2(len(counts))
    H_norm = H / H_max if H_max > 0 else 0.0
    return float(H), float(H_max), float(H_norm)


def run(domain: str = "sustainable packaging concepts"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Requires: pip install sentence-transformers")
        return

    suffix = _domain_suffix(domain)
    print(f"Loading DCE ideas for domain: {domain}")
    ideas = load_dce_ideas(domain=domain)
    print(f"Loaded {len(ideas)} DCE ideas")

    # --- Category distribution ---
    categories = [idea.get("category", "Unknown") for idea in ideas]
    cat_counts = Counter(categories)
    n_categories = len(cat_counts)

    print(f"\n{'='*65}")
    print(f"Category Distribution — {domain}")
    print(f"{'='*65}")
    print(f"Unique categories: {n_categories}")
    print(f"Total ideas: {len(ideas)}")

    print(f"\n{'Category':<40} {'Count':>6} {'Pct':>7}")
    print("-" * 55)
    for cat, count in cat_counts.most_common():
        pct = count / len(ideas) * 100
        print(f"{cat[:39]:<40} {count:>6} {pct:>6.1f}%")

    # Entropy
    counts_list = list(cat_counts.values())
    H, H_max, H_norm = compute_entropy(counts_list)
    print(f"\nDistribution entropy: {H:.3f} bits")
    print(f"Maximum entropy (log2({n_categories})): {H_max:.3f} bits")
    print(f"Normalized entropy: {H_norm:.3f} (1.0 = perfectly uniform)")

    # --- Embedding-based consistency analysis ---
    print("\nLoading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [f"{idea['name']}: {idea['description']}" for idea in ideas]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings)

    # Group embeddings by category
    cat_to_indices: dict[str, list[int]] = {}
    for i, cat in enumerate(categories):
        cat_to_indices.setdefault(cat, []).append(i)

    # Intra-category similarity (categories with >= 5 ideas)
    MIN_SIZE = 5
    intra_sims = {}
    cat_centroids = {}

    print(f"\n{'='*65}")
    print(f"Intra-Category Cosine Similarity (categories with >= {MIN_SIZE} ideas)")
    print(f"{'='*65}")
    print(f"\n{'Category':<40} {'N':>4} {'Mean Sim':>9}")
    print("-" * 55)

    for cat, indices in sorted(cat_to_indices.items()):
        cat_embs = embeddings[indices]
        # Compute centroid for inter-category analysis
        centroid = cat_embs.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        cat_centroids[cat] = centroid

        if len(indices) < MIN_SIZE:
            continue

        # Pairwise cosine similarity within category
        sim_matrix = cat_embs @ cat_embs.T
        # Extract upper triangle (excluding diagonal)
        n = len(indices)
        upper_mask = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[upper_mask]
        mean_sim = float(pairwise_sims.mean())
        intra_sims[cat] = mean_sim
        print(f"{cat[:39]:<40} {n:>4} {mean_sim:>9.4f}")

    overall_intra = np.mean(list(intra_sims.values())) if intra_sims else 0.0
    print(f"\nMean intra-category similarity: {overall_intra:.4f} (higher = more consistent)")

    # Inter-category centroid similarity
    cats_with_centroids = sorted(cat_centroids.keys())
    inter_sims = []

    if len(cats_with_centroids) >= 2:
        print(f"\n{'='*65}")
        print(f"Inter-Category Centroid Cosine Similarity")
        print(f"{'='*65}")

        centroid_matrix = np.array([cat_centroids[c] for c in cats_with_centroids])
        sim_matrix = centroid_matrix @ centroid_matrix.T
        n_cats = len(cats_with_centroids)
        upper_mask = np.triu_indices(n_cats, k=1)
        inter_sims_arr = sim_matrix[upper_mask]

        # Top-5 most similar pairs
        pair_indices = list(zip(upper_mask[0], upper_mask[1]))
        sorted_pairs = sorted(
            zip(inter_sims_arr, pair_indices), key=lambda x: x[0], reverse=True
        )

        print(f"\nTop-5 most similar category pairs:")
        for sim_val, (i, j) in sorted_pairs[:5]:
            print(f"  {sim_val:.4f}  {cats_with_centroids[i]} <-> {cats_with_centroids[j]}")

        mean_inter = float(inter_sims_arr.mean())
        print(f"\nMean inter-category centroid similarity: {mean_inter:.4f} (lower = more distinct)")
    else:
        mean_inter = 0.0
        print("\nInsufficient categories for inter-category analysis.")

    # --- Summary ---
    print(f"\n{'='*65}")
    print(f"Summary")
    print(f"{'='*65}")
    print(f"Unique categories:              {n_categories}")
    print(f"Distribution entropy:           {H:.3f} / {H_max:.3f} bits ({H_norm:.3f} normalized)")
    print(f"Mean intra-category similarity: {overall_intra:.4f}")
    print(f"Mean inter-category similarity: {mean_inter:.4f}")

    # --- Save results ---
    output_dir = Path(f"data/processed/category_quality_{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": domain,
        "n_ideas": len(ideas),
        "n_categories": n_categories,
        "category_counts": dict(cat_counts.most_common()),
        "entropy": H,
        "max_entropy": H_max,
        "normalized_entropy": H_norm,
        "intra_category_similarity": {k: float(v) for k, v in intra_sims.items()},
        "mean_intra_category_similarity": float(overall_intra),
        "mean_inter_category_similarity": float(mean_inter),
        "min_category_size_for_intra": MIN_SIZE,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", type=str, default="sustainable packaging concepts"
    )
    args = parser.parse_args()
    run(domain=args.domain)
