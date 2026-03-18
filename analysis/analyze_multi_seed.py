"""Step 6: Analyze multi-seed experiment results.

Computes mean +/- std for EDV retention and collapse rate across seeds,
runs paired significance tests (Wilcoxon signed-rank).
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonlines
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from src.embeddings import EmbeddingClient
from src.metrics import edv_batch, collapse_rate


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def load_results(method: str, seed: int, domain: str) -> list[dict]:
    suffix = _domain_suffix(domain)
    path = Path(f"data/raw/exp2_comparison_{suffix}_seed{seed}/{method}/results.jsonl")
    if not path.exists():
        return []
    batches = []
    with jsonlines.open(str(path)) as reader:
        for batch in reader:
            batches.append(batch)
    return batches


def compute_metrics_for_run(batches: list[dict]) -> dict:
    """Compute EDV retention and collapse rate for one run."""
    embedding_client = EmbeddingClient()
    all_embeddings = []
    edvs = []

    for batch in batches:
        texts = [f"{idea['name']}: {idea['description']}" for idea in batch["ideas"]]
        probs = [idea["probability"] for idea in batch["ideas"]]
        if not texts:
            edvs.append(0.0)
            continue
        new_embs = np.array(embedding_client.embed(texts))
        prior_embs = np.array(all_embeddings) if all_embeddings else np.array([])
        edv = edv_batch(np.array(probs), new_embs, prior_embs)
        edvs.append(edv)
        all_embeddings.extend(new_embs.tolist())

    all_emb_arr = np.array(all_embeddings)
    ideas_per_batch = len(all_embeddings) // len(batches) if batches else 5
    early_end = min(50 * ideas_per_batch, len(all_emb_arr))
    late_start = max(0, len(all_emb_arr) - 50 * ideas_per_batch)
    cr = collapse_rate(all_emb_arr[:early_end], all_emb_arr[late_start:])

    edv_ret = (edvs[-1] / edvs[0] * 100) if edvs and edvs[0] > 0 else 0

    return {"edv_retention": edv_ret, "collapse_rate": cr, "edvs": edvs}


def run(domain: str = "sustainable packaging concepts"):
    suffix = _domain_suffix(domain)

    # Load manifest
    manifest_path = Path(f"data/raw/multi_seed_manifest_{suffix}.json")
    if not manifest_path.exists():
        print(f"No manifest found at {manifest_path}. Run experiments/run_multi_seed.py first.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    seeds = manifest["seeds"]
    methods = manifest["methods"]

    print(f"\nDomain: {domain}")
    print(f"Seeds: {seeds}")
    print(f"Methods: {methods}")

    # Collect per-seed metrics
    results = {}
    for method in methods:
        edv_rets = []
        collapse_rates = []
        for seed in seeds:
            batches = load_results(method, seed, domain)
            if not batches:
                print(f"  WARNING: No data for {method} seed {seed}")
                continue
            m = compute_metrics_for_run(batches)
            edv_rets.append(m["edv_retention"])
            collapse_rates.append(m["collapse_rate"])
        results[method] = {
            "edv_retention_mean": np.mean(edv_rets),
            "edv_retention_std": np.std(edv_rets),
            "collapse_rate_mean": np.mean(collapse_rates),
            "collapse_rate_std": np.std(collapse_rates),
            "edv_rets": edv_rets,
            "collapse_rates": collapse_rates,
        }

    # Print table
    print("\n" + "=" * 70)
    print(f"{'Method':<20} {'EDV Retention':<25} {'Collapse Rate':<25}")
    print("-" * 70)
    for method in methods:
        r = results[method]
        edv_str = f"{r['edv_retention_mean']:.1f} +/- {r['edv_retention_std']:.1f}%"
        cr_str = f"{r['collapse_rate_mean']:.1%} +/- {r['collapse_rate_std']:.1%}"
        print(f"{method:<20} {edv_str:<25} {cr_str:<25}")
    print("=" * 70)

    # Significance tests (DCE vs each other method)
    if "dce" in results and len(results["dce"]["edv_rets"]) >= 3:
        try:
            from scipy.stats import wilcoxon
            print("\nPaired Wilcoxon tests (DCE vs others):")
            dce_edvs = results["dce"]["edv_rets"]
            for method in methods:
                if method == "dce":
                    continue
                other_edvs = results[method]["edv_rets"]
                if len(other_edvs) == len(dce_edvs) and len(dce_edvs) >= 3:
                    try:
                        stat, p = wilcoxon(dce_edvs, other_edvs)
                        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                        print(f"  DCE vs {method}: p={p:.4f} ({sig})")
                    except ValueError:
                        print(f"  DCE vs {method}: insufficient variation for test")
        except ImportError:
            print("\nInstall scipy for significance tests: pip install scipy")

    # Save
    output_dir = Path(f"data/processed/multi_seed_{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump({k: {kk: v for kk, v in vv.items() if kk not in ("edv_rets", "collapse_rates")}
                    for k, vv in results.items()}, f, indent=2)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="sustainable packaging concepts")
    args = parser.parse_args()
    run(domain=args.domain)
