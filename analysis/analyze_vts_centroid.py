"""VTS centroid validation: test whether VTS probability rankings correlate with typicality.

Loads naive results (no VTS filtering applied), embeds all ideas with MiniLM-L6-v2,
computes the centroid of the full distribution, then compares mean distance-to-centroid
for VTS-would-accept (P < 0.10) vs VTS-would-reject (P >= 0.10) groups.

Expected: rejected (high-P) ideas are closer to centroid (more typical);
accepted (low-P) ideas are farther (more unusual).
"""

from __future__ import annotations

from pathlib import Path

import jsonlines
import numpy as np


VTS_PROBABILITY_THRESHOLD = 0.10


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def load_naive_ideas(domain: str) -> list[dict]:
    """Load all ideas from naive method (unfiltered generation)."""
    suffix = _domain_suffix(domain)
    path = Path(f"data/raw/exp2_comparison_{suffix}/naive/results.jsonl")
    ideas = []
    with jsonlines.open(str(path)) as reader:
        for batch in reader:
            for idea in batch["ideas"]:
                ideas.append(idea)
    return ideas


def run(domain: str = "sustainable packaging concepts"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Requires: pip install sentence-transformers")
        return

    from scipy import stats

    suffix = _domain_suffix(domain)
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    ideas = load_naive_ideas(domain=domain)
    print(f"Loaded {len(ideas)} naive ideas")

    # Extract probability scores
    probs = np.array([idea.get("probability", 0.0) for idea in ideas])

    # Embed all ideas
    texts = [f"{idea['name']}: {idea['description']}" for idea in ideas]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings)

    # Compute centroid
    centroid = embeddings.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)  # normalize

    # Distance to centroid for each idea (1 - cosine similarity)
    distances = 1.0 - embeddings @ centroid

    # Split by VTS decision
    accept_mask = probs < VTS_PROBABILITY_THRESHOLD
    reject_mask = ~accept_mask

    accept_distances = distances[accept_mask]
    reject_distances = distances[reject_mask]

    print(f"\nDomain: {domain}")
    print(f"VTS threshold: P < {VTS_PROBABILITY_THRESHOLD}")
    print(f"Total ideas: {len(ideas)}")
    print(f"VTS-would-accept (P < {VTS_PROBABILITY_THRESHOLD}): {accept_mask.sum()}")
    print(f"VTS-would-reject (P >= {VTS_PROBABILITY_THRESHOLD}): {reject_mask.sum()}")

    print(f"\nMean distance to centroid:")
    print(f"  VTS-accept (low P, should be unusual): {accept_distances.mean():.4f} +/- {accept_distances.std():.4f}")
    print(f"  VTS-reject (high P, should be typical): {reject_distances.mean():.4f} +/- {reject_distances.std():.4f}")

    # Statistical test
    t_stat, p_value = stats.mannwhitneyu(
        accept_distances, reject_distances, alternative="greater"
    )
    print(f"\nMann-Whitney U test (accept > reject):")
    print(f"  U statistic: {t_stat:.1f}")
    print(f"  p-value: {p_value:.6f}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        (accept_distances.std() ** 2 + reject_distances.std() ** 2) / 2
    )
    cohens_d = (accept_distances.mean() - reject_distances.mean()) / pooled_std
    print(f"  Cohen's d: {cohens_d:.3f}")

    # Probability score statistics
    print(f"\nProbability score distribution:")
    print(f"  Accept group mean P: {probs[accept_mask].mean():.3f}")
    print(f"  Reject group mean P: {probs[reject_mask].mean():.3f}")
    print(f"  Overall mean P: {probs.mean():.3f}")

    # Correlation between probability and distance
    corr, corr_p = stats.spearmanr(probs, distances)
    print(f"\nSpearman correlation (probability vs distance to centroid):")
    print(f"  rho = {corr:.3f}, p = {corr_p:.6f}")

    # Save results
    output_dir = Path(f"data/processed/vts_centroid_{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "centroid_analysis.npz",
        distances=distances,
        probs=probs,
        accept_mask=accept_mask,
        accept_mean_dist=accept_distances.mean(),
        reject_mean_dist=reject_distances.mean(),
        cohens_d=cohens_d,
        mann_whitney_p=p_value,
        spearman_rho=corr,
        spearman_p=corr_p,
    )
    print(f"\nResults saved to {output_dir}")

    return {
        "accept_mean_dist": float(accept_distances.mean()),
        "reject_mean_dist": float(reject_distances.mean()),
        "accept_std": float(accept_distances.std()),
        "reject_std": float(reject_distances.std()),
        "cohens_d": float(cohens_d),
        "p_value": float(p_value),
        "spearman_rho": float(corr),
        "spearman_p": float(corr_p),
        "n_accept": int(accept_mask.sum()),
        "n_reject": int(reject_mask.sum()),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", type=str, default="sustainable packaging concepts"
    )
    args = parser.parse_args()
    run(domain=args.domain)
