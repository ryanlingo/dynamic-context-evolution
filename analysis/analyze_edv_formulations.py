"""Compare EDV under three formulations: multiplicative, additive, geometric.

Re-embeds all exp2 ideas (seed 42) with MiniLM and computes EDV retention
under each formulation. Tests whether method rankings are preserved across
formulations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jsonlines
import numpy as np

METHODS = [
    "naive",
    "vts_only",
    "vts_dedup",
    "dce",
    "dedup_only",
    "prompt_evo_only",
    "prompt_evo_dedup",
]


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def load_ideas(method: str, domain: str = "sustainable packaging concepts") -> list[dict]:
    """Load all batches for a method from exp2 results (seed 42 first, then unseeded)."""
    suffix = _domain_suffix(domain)
    seed_path = Path(f"data/raw/exp2_comparison_{suffix}_seed42/{method}/results.jsonl")
    plain_path = Path(f"data/raw/exp2_comparison_{suffix}/{method}/results.jsonl")
    path = seed_path if seed_path.exists() else plain_path
    batches = []
    with jsonlines.open(path) as reader:
        for batch in reader:
            batches.append(batch)
    return batches


def embed_all(batches: list[dict], model) -> list[np.ndarray]:
    """Embed all ideas batch-by-batch, returning list of arrays per batch."""
    all_batch_embs = []
    for batch in batches:
        texts = [f"{idea['name']}: {idea['description']}" for idea in batch["ideas"]]
        if not texts:
            all_batch_embs.append(np.array([]))
            continue
        embs = model.encode(texts, normalize_embeddings=True)
        all_batch_embs.append(np.array(embs))
    return all_batch_embs


def compute_edv_series_all(
    batch_embeddings: list[np.ndarray],
    batch_probs: list[list[float]],
) -> dict[str, list[float]]:
    """Compute EDV per batch under three formulations."""
    series = {"multiplicative": [], "additive": [], "geometric": []}
    all_prior = []

    for embs, probs in zip(batch_embeddings, batch_probs):
        if len(embs) == 0:
            for key in series:
                series[key].append(0.0)
            continue

        prior = np.array(all_prior) if all_prior else np.array([])
        depths = [1.0 - p for p in probs]

        if prior.size == 0:
            breadths = [1.0] * len(probs)
        else:
            sims = embs @ prior.T
            breadths = [1.0 - float(np.max(sims[i])) for i in range(len(probs))]

        mult_vals = [d * b for d, b in zip(depths, breadths)]
        add_vals = [(d + b) / 2.0 for d, b in zip(depths, breadths)]
        geom_vals = [math.sqrt(d * b) if d * b >= 0 else 0.0 for d, b in zip(depths, breadths)]

        series["multiplicative"].append(float(np.mean(mult_vals)))
        series["additive"].append(float(np.mean(add_vals)))
        series["geometric"].append(float(np.mean(geom_vals)))

        all_prior.extend(embs.tolist())

    return series


def edv_retention(edvs: list[float]) -> float:
    """EDV retention: batch 200 / batch 1 * 100."""
    if not edvs or edvs[0] <= 0:
        return 0.0
    return edvs[-1] / edvs[0] * 100


def run(domain: str = "sustainable packaging concepts"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Requires: pip install sentence-transformers")
        return

    suffix = _domain_suffix(domain)
    print("Loading independent embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    output_dir = Path(f"data/processed/edv_formulations_{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect retentions per formulation
    retentions: dict[str, dict[str, float]] = {m: {} for m in METHODS}
    all_series: dict[str, dict[str, list[float]]] = {}

    for method in METHODS:
        print(f"  Processing {method}...")
        batches = load_ideas(method, domain=domain)
        probs = [[idea["probability"] for idea in b["ideas"]] for b in batches]
        batch_embs = embed_all(batches, model)
        series = compute_edv_series_all(batch_embs, probs)
        all_series[method] = series

        for form_name, edvs in series.items():
            retentions[method][form_name] = edv_retention(edvs)

    # Determine multiplicative ranking
    mult_ranking = sorted(METHODS, key=lambda m: retentions[m]["multiplicative"], reverse=True)

    # Print table
    print(f"\nDomain: {domain}")
    print("\n" + "=" * 90)
    print(
        f"{'Method':<18} {'Mult %':<12} {'Additive %':<12} {'Geometric %':<14} {'Rank preserved?'}"
    )
    print("-" * 90)

    results_json = {}
    for method in METHODS:
        r = retentions[method]
        # Check if ranking matches multiplicative for additive and geometric
        add_ranking = sorted(METHODS, key=lambda m: retentions[m]["additive"], reverse=True)
        geom_ranking = sorted(METHODS, key=lambda m: retentions[m]["geometric"], reverse=True)
        rank_preserved = (add_ranking == mult_ranking) and (geom_ranking == mult_ranking)

        print(
            f"{method:<18} {r['multiplicative']:<12.1f} {r['additive']:<12.1f} "
            f"{r['geometric']:<14.1f} {'yes' if rank_preserved else 'no'}"
        )

        results_json[method] = {
            "multiplicative_retention": round(r["multiplicative"], 2),
            "additive_retention": round(r["additive"], 2),
            "geometric_retention": round(r["geometric"], 2),
            "edv_series": {
                k: [round(v, 6) for v in vals] for k, vals in all_series[method].items()
            },
        }

    print("=" * 90)

    # Overall rank preservation check
    add_ranking = sorted(METHODS, key=lambda m: retentions[m]["additive"], reverse=True)
    geom_ranking = sorted(METHODS, key=lambda m: retentions[m]["geometric"], reverse=True)
    print(f"\nMultiplicative ranking: {mult_ranking}")
    print(f"Additive ranking:      {add_ranking}")
    print(f"Geometric ranking:     {geom_ranking}")
    print(f"Rankings fully preserved: {mult_ranking == add_ranking == geom_ranking}")

    results_json["_meta"] = {
        "domain": domain,
        "seed": 42,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "formulations": {
            "multiplicative": "depth * breadth",
            "additive": "(depth + breadth) / 2",
            "geometric": "sqrt(depth * breadth)",
        },
        "multiplicative_ranking": mult_ranking,
        "additive_ranking": add_ranking,
        "geometric_ranking": geom_ranking,
        "rankings_preserved": mult_ranking == add_ranking == geom_ranking,
    }

    out_path = output_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="sustainable packaging concepts")
    args = parser.parse_args()
    run(domain=args.domain)
