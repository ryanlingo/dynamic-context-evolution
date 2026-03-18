"""Analyze Experiment 1: batch novelty + cluster count plots."""

from __future__ import annotations

from pathlib import Path

import jsonlines
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

from src.embeddings import EmbeddingClient
from src.metrics import batch_novelty, cluster_count
from analysis.plot_utils import COLORS, setup_style, save_figure


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def load_results(results_path: str | None = None, domain: str = "sustainable packaging concepts"):
    """Load all batch results."""
    if results_path is None:
        results_path = f"data/raw/exp1_collapse_{_domain_suffix(domain)}/results.jsonl"
    batches = []
    with jsonlines.open(results_path) as reader:
        for batch in reader:
            batches.append(batch)
    return batches


def compute_metrics(batches: list[dict]) -> dict:
    """Compute batch novelty and cluster count over time."""
    embedding_client = EmbeddingClient()

    all_embeddings = []
    novelties = []
    cluster_counts = []
    batch_numbers = []

    for batch in batches:
        texts = [f"{idea['name']}: {idea['description']}" for idea in batch["ideas"]]
        if not texts:
            continue
        new_embs = np.array(embedding_client.embed(texts))
        prior_embs = np.array(all_embeddings) if all_embeddings else np.array([])

        nov = batch_novelty(new_embs, prior_embs)
        novelties.append(nov)

        all_embeddings.extend(new_embs.tolist())
        cc = cluster_count(np.array(all_embeddings))
        cluster_counts.append(cc)
        batch_numbers.append(batch["batch_number"])

    return {
        "batch_numbers": batch_numbers,
        "novelties": novelties,
        "cluster_counts": cluster_counts,
    }


def plot(metrics: dict, domain: str = "sustainable packaging concepts"):
    setup_style()
    suffix = _domain_suffix(domain)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Batch novelty
    ax1.plot(metrics["batch_numbers"], metrics["novelties"], color=COLORS["naive"])
    ax1.set_xlabel("Batch Number")
    ax1.set_ylabel("Batch Novelty (avg min cosine distance)")
    ax1.set_title("Cross-Batch Mode Collapse: Novelty Decay")
    ax1.axhline(y=0.15, color="gray", linestyle="--", alpha=0.5, label="Collapse threshold (0.15)")
    ax1.legend()

    # Cluster count
    ax2.plot(metrics["batch_numbers"], metrics["cluster_counts"], color=COLORS["naive"])
    ax2.set_xlabel("Batch Number")
    ax2.set_ylabel("Unique Cluster Count (HDBSCAN)")
    ax2.set_title("Cross-Batch Mode Collapse: Cluster Saturation")

    fig.tight_layout()
    save_figure(fig, f"exp1_collapse_{suffix}")

    # Save computed metrics
    output_dir = Path(f"data/processed/exp1_collapse_{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "metrics.npz",
        batch_numbers=metrics["batch_numbers"],
        novelties=metrics["novelties"],
        cluster_counts=metrics["cluster_counts"],
    )
    print(f"Metrics saved to {output_dir}/metrics.npz")


def run(domain: str = "sustainable packaging concepts"):
    batches = load_results(domain=domain)
    metrics = compute_metrics(batches)
    plot(metrics, domain=domain)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="sustainable packaging concepts")
    args = parser.parse_args()
    run(domain=args.domain)
