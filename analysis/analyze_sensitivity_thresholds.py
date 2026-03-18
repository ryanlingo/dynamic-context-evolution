"""Analyze threshold sensitivity: tau (VTS) and delta (dedup) sweeps."""

from __future__ import annotations

from pathlib import Path

import jsonlines
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

from src.embeddings import EmbeddingClient
from src.metrics import edv_batch, collapse_rate
from analysis.plot_utils import setup_style, save_figure


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def load_results(param: str, value: float, domain: str) -> list[dict]:
    """Load results for a specific param/value combination."""
    suffix = _domain_suffix(domain)
    value_label = f"{param}_{value:.2f}"
    path = Path(f"data/raw/sensitivity_thresholds_{suffix}/{value_label}/results.jsonl")
    batches = []
    with jsonlines.open(str(path)) as reader:
        for batch in reader:
            batches.append(batch)
    return batches


def discover_values(param: str, domain: str) -> list[float]:
    """Auto-discover available values from data directory."""
    suffix = _domain_suffix(domain)
    base_dir = Path(f"data/raw/sensitivity_thresholds_{suffix}")
    if not base_dir.exists():
        return []
    prefix = f"{param}_"
    values = sorted(
        float(d.name[len(prefix):])
        for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    )
    return values


def compute_metrics(batches: list[dict]) -> dict:
    """Compute EDV retention and collapse rate for a set of batches."""
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

    total_accepted = sum(b.get("accepted_count", len(b["ideas"])) for b in batches)

    all_emb_arr = np.array(all_embeddings)
    ideas_per_batch = len(all_embeddings) // len(batches) if batches else 5
    early_end = 50 * ideas_per_batch
    late_start = max(0, len(all_embeddings) - 50 * ideas_per_batch)
    cr = collapse_rate(
        all_emb_arr[:early_end],
        all_emb_arr[late_start:],
    )

    return {
        "edvs": edvs,
        "collapse_rate": cr,
        "edv_retention_pct": (edvs[-1] / edvs[0] * 100) if edvs and edvs[0] > 0 else 0,
        "total_accepted": total_accepted,
    }


def run(param: str, domain: str = "sustainable packaging concepts"):
    values = discover_values(param, domain)
    if not values:
        print(f"No data found for param={param}, domain={domain}")
        return

    suffix = _domain_suffix(domain)
    output_dir = Path(f"data/processed/sensitivity_thresholds_{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    param_label = "tau" if param == "tau" else "delta"
    param_symbol = "\u03C4" if param == "tau" else "\u03B4"

    # Collect metrics for each value
    all_metrics = {}
    print(f"\nThreshold sensitivity: {param_symbol} ({param})")
    print(f"Domain: {domain}")
    print("\n" + "=" * 65)
    print(f"{'Value':<10} {'Ideas accepted':<18} {'EDV retention':<18} {'Collapse rate':<15}")
    print("-" * 65)

    for value in values:
        batches = load_results(param, value, domain)
        m = compute_metrics(batches)
        all_metrics[value] = m
        print(
            f"{value:<10.2f} {m['total_accepted']:<18} "
            f"{m['edv_retention_pct']:<18.1f}% {m['collapse_rate']:<15.1%}"
        )
        np.savez(
            output_dir / f"{param}_{value:.2f}_metrics.npz",
            **{k: v for k, v in m.items() if k != "edvs"},
            edvs=m["edvs"],
        )

    print("=" * 65)

    # Generate line plot: EDV retention and collapse rate vs parameter value
    setup_style()
    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    sorted_values = sorted(all_metrics.keys())
    edv_retentions = [all_metrics[v]["edv_retention_pct"] for v in sorted_values]
    collapse_rates_pct = [all_metrics[v]["collapse_rate"] * 100 for v in sorted_values]

    color_edv = "#D55E00"   # vermillion
    color_cr = "#0072B2"    # blue

    ax1.plot(sorted_values, edv_retentions, "o-", color=color_edv, label="EDV retention (%)")
    ax1.set_xlabel(f"{param_symbol} value")
    ax1.set_ylabel("EDV retention (%)", color=color_edv)
    ax1.tick_params(axis="y", labelcolor=color_edv)

    ax2 = ax1.twinx()
    ax2.plot(sorted_values, collapse_rates_pct, "s--", color=color_cr, label="Collapse rate (%)")
    ax2.set_ylabel("Collapse rate (%)", color=color_cr)
    ax2.tick_params(axis="y", labelcolor=color_cr)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title(f"Threshold sensitivity: {param_symbol}")
    fig.tight_layout()

    save_figure(fig, f"sensitivity_{param}")
    print(f"\nFigure saved as sensitivity_{param}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze threshold sensitivity (tau/delta)"
    )
    parser.add_argument(
        "--param",
        type=str,
        required=True,
        choices=["tau", "delta"],
        help="Parameter to analyze",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="sustainable packaging concepts",
        help="Domain to analyze",
    )
    args = parser.parse_args()
    run(param=args.param, domain=args.domain)
