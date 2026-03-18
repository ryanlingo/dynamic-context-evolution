"""Analyze Experiment 2: EDV curves + comparison table."""

from __future__ import annotations

from pathlib import Path

import jsonlines
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

from src.embeddings import EmbeddingClient
from src.metrics import batch_novelty, edv_batch, collapse_rate, cluster_count
from analysis.plot_utils import COLORS, LABELS, setup_style, save_figure

METHODS = ["naive", "vts_only", "vts_dedup", "dce"]


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def load_results(method: str, domain: str = "sustainable packaging concepts") -> list[dict]:
    suffix = _domain_suffix(domain)
    path = f"data/raw/exp2_comparison_{suffix}/{method}/results.jsonl"
    batches = []
    with jsonlines.open(path) as reader:
        for batch in reader:
            batches.append(batch)
    return batches


def compute_metrics_for_method(batches: list[dict]) -> dict:
    embedding_client = EmbeddingClient()
    all_embeddings = []
    edvs = []
    novelties = []
    batch_numbers = []

    for batch in batches:
        texts = [f"{idea['name']}: {idea['description']}" for idea in batch["ideas"]]
        probs = [idea["probability"] for idea in batch["ideas"]]
        if not texts:
            edvs.append(0.0)
            novelties.append(0.0)
            batch_numbers.append(batch["batch_number"])
            continue

        new_embs = np.array(embedding_client.embed(texts))
        prior_embs = np.array(all_embeddings) if all_embeddings else np.array([])

        edv = edv_batch(np.array(probs), new_embs, prior_embs)
        nov = batch_novelty(new_embs, prior_embs)

        edvs.append(edv)
        novelties.append(nov)
        all_embeddings.extend(new_embs.tolist())
        batch_numbers.append(batch["batch_number"])

    # Collapse rate: first 50 vs last 50 batches
    all_emb_arr = np.array(all_embeddings)
    # Estimate idea count per batch (roughly 5)
    ideas_per_batch = len(all_embeddings) // len(batches) if batches else 5
    early_end = 50 * ideas_per_batch
    late_start = max(0, len(all_embeddings) - 50 * ideas_per_batch)
    early_embs = all_emb_arr[:early_end] if early_end <= len(all_emb_arr) else all_emb_arr
    late_embs = all_emb_arr[late_start:] if late_start < len(all_emb_arr) else all_emb_arr

    cr = collapse_rate(early_embs, late_embs)

    return {
        "batch_numbers": batch_numbers,
        "edvs": edvs,
        "novelties": novelties,
        "collapse_rate": cr,
        "edv_at_200_pct": (edvs[-1] / edvs[0] * 100) if edvs and edvs[0] > 0 else 0,
    }


def plot_edv_curves(all_metrics: dict[str, dict], domain: str = "sustainable packaging concepts"):
    setup_style()
    suffix = _domain_suffix(domain)
    fig, ax = plt.subplots(figsize=(8, 5))

    for method in all_metrics:
        m = all_metrics[method]
        ax.plot(
            m["batch_numbers"], m["edvs"],
            color=COLORS.get(method, "#000000"),
            label=LABELS.get(method, method),
        )

    ax.set_xlabel("Batch Number")
    ax.set_ylabel("EDV (Effective Diversity Volume)")
    ax.set_title("EDV Over Time: DCE vs. Baselines")
    ax.legend()
    save_figure(fig, f"exp2_edv_curves_{suffix}")


ALL_METHODS = METHODS + [
    "dedup_only", "prompt_evo_only", "prompt_evo_dedup",
    "temp_1.2_dedup", "nucleus_0.9_dedup",
]


def print_comparison_table(all_metrics: dict[str, dict]):
    print("\n" + "=" * 60)
    print(f"{'Method':<25} {'EDV@200 (% of B1)':<22} {'Collapse Rate':<15}")
    print("-" * 60)
    for method in all_metrics:
        m = all_metrics[method]
        label = LABELS.get(method, method)
        print(f"{label:<25} {m['edv_at_200_pct']:.1f}%{'':<16} {m['collapse_rate']:.1%}")
    print("=" * 60)


def run(domain: str = "sustainable packaging concepts", methods: list[str] | None = None):
    suffix = _domain_suffix(domain)
    methods = methods or METHODS
    all_metrics = {}
    for method in methods:
        results_path = f"data/raw/exp2_comparison_{suffix}/{method}/results.jsonl"
        from pathlib import Path as _P
        if not _P(results_path).exists():
            print(f"Skipping {method} (no data at {results_path})")
            continue
        print(f"Computing metrics for {method}...")
        batches = load_results(method, domain=domain)
        all_metrics[method] = compute_metrics_for_method(batches)

    plot_edv_curves(all_metrics, domain=domain)
    print_comparison_table(all_metrics)

    # Save
    output_dir = Path(f"data/processed/exp2_comparison_{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for method, m in all_metrics.items():
        np.savez(
            output_dir / f"{method}_metrics.npz",
            batch_numbers=m["batch_numbers"],
            edvs=m["edvs"],
            novelties=m["novelties"],
            collapse_rate=m["collapse_rate"],
            edv_at_200_pct=m["edv_at_200_pct"],
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="sustainable packaging concepts")
    parser.add_argument("--methods", nargs="*", default=None, help="Methods to analyze (default: core 4)")
    parser.add_argument("--all-methods", action="store_true", help="Analyze all available methods")
    args = parser.parse_args()
    methods = ALL_METHODS if args.all_methods else args.methods
    run(domain=args.domain, methods=methods)
