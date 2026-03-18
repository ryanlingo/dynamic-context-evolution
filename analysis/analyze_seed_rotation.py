"""Seed rotation + post-hoc dedup baseline.

Pools naive ideas from 3 seeds (42, 123, 456), interleaves by batch number
(round-robin seed rotation), then applies greedy post-hoc dedup at delta=0.85.
Computes EDV retention, collapse rate, and HDBSCAN cluster count on the
accepted set.
"""

from __future__ import annotations

import json
from pathlib import Path

import hdbscan
import jsonlines
import numpy as np


SEEDS = [42, 123, 456]
DEDUP_THRESHOLD = 0.85


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def load_seed_batches(seed: int, domain: str) -> list[dict]:
    """Load all batches for a given seed's naive results."""
    suffix = _domain_suffix(domain)
    path = f"data/raw/exp2_comparison_{suffix}_seed{seed}/naive/results.jsonl"
    batches = []
    with jsonlines.open(path) as reader:
        for batch in reader:
            batches.append(batch)
    return batches


def interleave_batches(seed_batches: dict[int, list[dict]]) -> list[dict]:
    """Interleave batches round-robin by batch number across seeds.

    For each batch index, take batch from seed 42, then 123, then 456.
    Returns a flat list of idea dicts with 'batch_number' set to the
    interleaved position.
    """
    max_batches = max(len(b) for b in seed_batches.values())
    interleaved: list[dict] = []
    interleaved_batch_num = 0
    for batch_idx in range(max_batches):
        for seed in SEEDS:
            batches = seed_batches[seed]
            if batch_idx >= len(batches):
                continue
            batch = batches[batch_idx]
            for idea in batch["ideas"]:
                interleaved.append({
                    **idea,
                    "interleaved_batch": interleaved_batch_num,
                    "original_seed": seed,
                    "original_batch": batch.get("batch_number", batch_idx),
                })
            interleaved_batch_num += 1
    return interleaved


def greedy_dedup(
    ideas: list[dict],
    embeddings: np.ndarray,
    threshold: float = DEDUP_THRESHOLD,
) -> tuple[list[dict], np.ndarray, list[int]]:
    """Greedy post-hoc dedup: accept idea only if max cosine sim to all
    previously accepted ideas is <= threshold."""
    accepted_ideas: list[dict] = []
    accepted_embs: list[np.ndarray] = []
    accepted_indices: list[int] = []

    for i, (idea, emb) in enumerate(zip(ideas, embeddings)):
        if len(accepted_embs) == 0:
            accepted_ideas.append(idea)
            accepted_embs.append(emb)
            accepted_indices.append(i)
            continue
        # Cosine similarity (embeddings are already normalized)
        sims = emb @ np.array(accepted_embs).T
        if np.max(sims) <= threshold:
            accepted_ideas.append(idea)
            accepted_embs.append(emb)
            accepted_indices.append(i)

    return accepted_ideas, np.array(accepted_embs), accepted_indices


def compute_edv_retention(
    ideas: list[dict],
    embeddings: np.ndarray,
    ideas_per_batch: int = 5,
) -> float:
    """Compute EDV retention (last batch / first batch) on accepted ideas.

    Groups ideas into virtual batches of `ideas_per_batch` in order.
    EDV = mean(depth * breadth) per batch.
    """
    n = len(ideas)
    if n < ideas_per_batch * 2:
        return 0.0

    n_batches = n // ideas_per_batch
    all_prior: list[np.ndarray] = []
    edvs: list[float] = []

    for b in range(n_batches):
        start = b * ideas_per_batch
        end = start + ideas_per_batch
        batch_embs = embeddings[start:end]
        batch_ideas = ideas[start:end]

        depths = [1.0 - idea.get("probability", 0.5) for idea in batch_ideas]

        if len(all_prior) == 0:
            breadths = [1.0] * len(batch_ideas)
        else:
            prior = np.array(all_prior)
            sims = batch_embs @ prior.T
            breadths = [1.0 - float(np.max(sims[i])) for i in range(len(batch_ideas))]

        edv = float(np.mean([d * b for d, b in zip(depths, breadths)]))
        edvs.append(edv)
        all_prior.extend(batch_embs.tolist())

    if not edvs or edvs[0] == 0:
        return 0.0
    return (edvs[-1] / edvs[0]) * 100


def compute_collapse_rate(
    embeddings: np.ndarray,
    ideas_per_batch: int = 5,
    threshold: float = DEDUP_THRESHOLD,
) -> float:
    """Fraction of ideas in final 50 'batches' with sim > threshold to
    any idea in first 50 'batches'."""
    n = len(embeddings)
    early_end = min(50 * ideas_per_batch, n)
    late_start = max(0, n - 50 * ideas_per_batch)

    early = embeddings[:early_end]
    late = embeddings[late_start:]

    if len(early) == 0 or len(late) == 0:
        return 0.0

    sims = late @ early.T
    max_sims = np.max(sims, axis=1)
    return float(np.mean(max_sims > threshold))


def compute_cluster_count(embeddings: np.ndarray) -> int:
    """HDBSCAN cluster count with min_cluster_size=5."""
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters


def run(domain: str = "sustainable packaging concepts") -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Requires: pip install sentence-transformers")
        return

    suffix = _domain_suffix(domain)

    # ------------------------------------------------------------------
    # 1. Load naive ideas from all seeds
    # ------------------------------------------------------------------
    print(f"Domain: {domain}")
    print(f"Loading naive ideas from seeds {SEEDS}...")

    seed_batches: dict[int, list[dict]] = {}
    for seed in SEEDS:
        batches = load_seed_batches(seed, domain)
        seed_batches[seed] = batches
        n_ideas = sum(len(b["ideas"]) for b in batches)
        print(f"  Seed {seed}: {len(batches)} batches, {n_ideas} ideas")

    # ------------------------------------------------------------------
    # 2. Interleave by batch number (round-robin seed rotation)
    # ------------------------------------------------------------------
    print("Interleaving batches (round-robin by batch number)...")
    interleaved = interleave_batches(seed_batches)
    print(f"  Total interleaved ideas: {len(interleaved)}")

    # ------------------------------------------------------------------
    # 3. Embed all with MiniLM
    # ------------------------------------------------------------------
    print("Loading MiniLM embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [f"{idea['name']}: {idea['description']}" for idea in interleaved]
    print(f"Embedding {len(texts)} ideas...")
    all_embeddings = model.encode(texts, normalize_embeddings=True)
    all_embeddings = np.array(all_embeddings)

    # ------------------------------------------------------------------
    # 4. Greedy post-hoc dedup at delta=0.85
    # ------------------------------------------------------------------
    print(f"Applying greedy post-hoc dedup (delta={DEDUP_THRESHOLD})...")
    accepted_ideas, accepted_embs, accepted_indices = greedy_dedup(
        interleaved, all_embeddings, threshold=DEDUP_THRESHOLD
    )
    print(f"  Accepted: {len(accepted_ideas)} / {len(interleaved)} "
          f"({len(accepted_ideas) / len(interleaved) * 100:.1f}%)")

    # ------------------------------------------------------------------
    # 5. Compute metrics
    # ------------------------------------------------------------------
    print("\nComputing metrics...")

    total_accepted = len(accepted_ideas)
    edv_retention = compute_edv_retention(accepted_ideas, accepted_embs)
    collapse_rate = compute_collapse_rate(accepted_embs)
    n_clusters = compute_cluster_count(accepted_embs)

    # ------------------------------------------------------------------
    # 6. Print results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{'Seed Rotation + Post-Hoc Dedup Baseline':^60}")
    print(f"{'Domain: ' + domain:^60}")
    print("=" * 60)
    print(f"  {'Seeds:':<30} {SEEDS}")
    print(f"  {'Dedup threshold (delta):':<30} {DEDUP_THRESHOLD}")
    print(f"  {'Total pooled ideas:':<30} {len(interleaved)}")
    print(f"  {'Accepted after dedup:':<30} {total_accepted}")
    print(f"  {'Acceptance rate:':<30} {total_accepted / len(interleaved) * 100:.1f}%")
    print(f"  {'EDV retention:':<30} {edv_retention:.1f}%")
    print(f"  {'Collapse rate:':<30} {collapse_rate:.1%}")
    print(f"  {'HDBSCAN clusters:':<30} {n_clusters}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    output_dir = Path(f"data/processed/seed_rotation_{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "domain": domain,
        "seeds": SEEDS,
        "dedup_threshold": DEDUP_THRESHOLD,
        "total_pooled": len(interleaved),
        "total_accepted": total_accepted,
        "acceptance_rate": total_accepted / len(interleaved),
        "edv_retention_pct": round(edv_retention, 2),
        "collapse_rate": round(collapse_rate, 4),
        "hdbscan_clusters": n_clusters,
    }

    out_path = output_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Seed rotation + post-hoc dedup baseline"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="sustainable packaging concepts",
    )
    args = parser.parse_args()
    run(domain=args.domain)
