"""Per-strategy analysis: evaluate individual strategy contributions in DCE.

Loads DCE results (200 batches with strategy field) and computes per-strategy:
- Acceptance rate (accepted_count / generated_count)
- Mean EDV
- Mean batch novelty
Broken down by phase (exploration vs exploitation).
"""

from __future__ import annotations

from pathlib import Path

import jsonlines
import numpy as np


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def load_dce_batches(domain: str, seed: str | None = None) -> list[dict]:
    """Load all DCE batches with strategy and phase metadata."""
    suffix = _domain_suffix(domain)
    if seed:
        path = Path(f"data/raw/exp2_comparison_{suffix}_seed{seed}/dce/results.jsonl")
    else:
        path = Path(f"data/raw/exp2_comparison_{suffix}/dce/results.jsonl")
    batches = []
    with jsonlines.open(str(path)) as reader:
        for batch in reader:
            batches.append(batch)
    return batches


def compute_batch_novelty(batch_ideas: list[dict], all_prior_embeddings: np.ndarray, model) -> float:
    """Compute mean minimum cosine distance of batch ideas to all prior ideas."""
    if len(all_prior_embeddings) == 0:
        return 1.0
    texts = [f"{idea['name']}: {idea['description']}" for idea in batch_ideas]
    batch_embs = model.encode(texts, normalize_embeddings=True)
    # Cosine similarity to all prior
    sims = batch_embs @ all_prior_embeddings.T
    min_dists = 1.0 - sims.max(axis=1)
    return float(min_dists.mean())


def compute_batch_edv(batch_ideas: list[dict], all_prior_embeddings: np.ndarray, model) -> float:
    """Compute EDV for a batch: mean of depth * breadth."""
    if len(all_prior_embeddings) == 0:
        # First batch: breadth is max
        depths = [1.0 - idea.get("probability", 0.0) for idea in batch_ideas]
        return float(np.mean(depths))

    texts = [f"{idea['name']}: {idea['description']}" for idea in batch_ideas]
    batch_embs = model.encode(texts, normalize_embeddings=True)
    sims = batch_embs @ all_prior_embeddings.T
    min_dists = 1.0 - sims.max(axis=1)  # breadth

    edvs = []
    for i, idea in enumerate(batch_ideas):
        depth = 1.0 - idea.get("probability", 0.0)
        breadth = min_dists[i]
        edvs.append(depth * breadth)
    return float(np.mean(edvs))


def run(domain: str = "sustainable packaging concepts"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Requires: pip install sentence-transformers")
        return

    suffix = _domain_suffix(domain)
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    batches = load_dce_batches(domain=domain)
    print(f"Loaded {len(batches)} DCE batches")

    # Per-strategy accumulators
    strategies = ["gap", "inversion", "cross_industry", "constraint"]
    phases = ["exploration", "exploitation"]

    # Compute metrics per batch, accumulating embeddings
    all_embeddings = np.zeros((0, 384))  # MiniLM dimension
    batch_metrics = []

    for batch in batches:
        strategy = batch.get("strategy", "unknown")
        phase = batch.get("phase", "unknown")
        accepted = batch.get("accepted_count", len(batch["ideas"]))
        generated = batch.get("generated_count", 5)
        ideas = batch["ideas"]

        # Only compute on accepted ideas
        novelty = compute_batch_novelty(ideas, all_embeddings, model)
        edv = compute_batch_edv(ideas, all_embeddings, model)

        # Add batch embeddings to memory
        texts = [f"{idea['name']}: {idea['description']}" for idea in ideas]
        embs = model.encode(texts, normalize_embeddings=True)
        all_embeddings = np.vstack([all_embeddings, embs])

        batch_metrics.append({
            "batch": batch["batch_number"],
            "strategy": strategy,
            "phase": phase,
            "accepted": accepted,
            "generated": generated,
            "novelty": novelty,
            "edv": edv,
        })

    # Aggregate by strategy
    print(f"\n{'='*75}")
    print(f"Per-Strategy Analysis — {domain}")
    print(f"{'='*75}")

    print(f"\n{'Strategy':<18} {'Batches':>7} {'Accept%':>8} {'Mean EDV':>9} {'Novelty':>9}")
    print("-" * 55)

    strategy_results = {}
    for s in strategies:
        s_batches = [m for m in batch_metrics if m["strategy"] == s]
        if not s_batches:
            continue
        n = len(s_batches)
        accept_rate = np.mean([m["accepted"] / m["generated"] * 100 for m in s_batches])
        mean_edv = np.mean([m["edv"] for m in s_batches])
        mean_novelty = np.mean([m["novelty"] for m in s_batches])
        strategy_results[s] = {
            "n": n,
            "accept_rate": accept_rate,
            "mean_edv": mean_edv,
            "mean_novelty": mean_novelty,
        }
        label = s.replace("_", " ").title()
        print(f"{label:<18} {n:>7} {accept_rate:>7.1f}% {mean_edv:>9.4f} {mean_novelty:>9.4f}")

    # By phase
    print(f"\n{'Phase':<18} {'Batches':>7} {'Accept%':>8} {'Mean EDV':>9} {'Novelty':>9}")
    print("-" * 55)

    phase_results = {}
    for p in phases:
        p_batches = [m for m in batch_metrics if m["phase"] == p]
        if not p_batches:
            continue
        n = len(p_batches)
        accept_rate = np.mean([m["accepted"] / m["generated"] * 100 for m in p_batches])
        mean_edv = np.mean([m["edv"] for m in p_batches])
        mean_novelty = np.mean([m["novelty"] for m in p_batches])
        phase_results[p] = {
            "n": n,
            "accept_rate": accept_rate,
            "mean_edv": mean_edv,
            "mean_novelty": mean_novelty,
        }
        print(f"{p.title():<18} {n:>7} {accept_rate:>7.1f}% {mean_edv:>9.4f} {mean_novelty:>9.4f}")

    # Strategy × Phase breakdown
    print(f"\n{'Strategy × Phase':<30} {'N':>4} {'Accept%':>8} {'Mean EDV':>9} {'Novelty':>9}")
    print("-" * 65)

    cross_results = {}
    for s in strategies:
        for p in phases:
            sp_batches = [m for m in batch_metrics if m["strategy"] == s and m["phase"] == p]
            if not sp_batches:
                continue
            n = len(sp_batches)
            accept_rate = np.mean([m["accepted"] / m["generated"] * 100 for m in sp_batches])
            mean_edv = np.mean([m["edv"] for m in sp_batches])
            mean_novelty = np.mean([m["novelty"] for m in sp_batches])
            label = f"{s.replace('_', ' ').title()} / {p.title()}"
            cross_results[f"{s}_{p}"] = {
                "n": n,
                "accept_rate": accept_rate,
                "mean_edv": mean_edv,
                "mean_novelty": mean_novelty,
            }
            print(f"{label:<30} {n:>4} {accept_rate:>7.1f}% {mean_edv:>9.4f} {mean_novelty:>9.4f}")

    # Save results
    output_dir = Path(f"data/processed/per_strategy_{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    import json
    results = {
        "domain": domain,
        "strategy_results": strategy_results,
        "phase_results": phase_results,
        "cross_results": cross_results,
        "batch_metrics": batch_metrics,
    }
    with open(output_dir / "per_strategy_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {output_dir}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain", type=str, default="sustainable packaging concepts"
    )
    args = parser.parse_args()
    run(domain=args.domain)
