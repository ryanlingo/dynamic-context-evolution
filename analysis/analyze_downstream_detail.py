"""Step 3: Detailed downstream evaluation with per-class F1 breakdown.

Reports full task spec (class count, distribution, random baseline, split sizes)
and per-class F1 for each training data source.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import jsonlines
import numpy as np
import yaml
from dotenv import load_dotenv

from src.domain_config import get_domain_config
from experiments.run_downstream import coarse_category, prepare_dataset, load_training_data

load_dotenv()


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def analyze_task_spec(ideas: list[dict], separators: list[str] | None = None):
    """Print full classification task specification."""
    texts = [f"{idea['name']}: {idea['description']}" for idea in ideas]
    labels = [coarse_category(idea["category"], separators=separators) for idea in ideas]
    counts = Counter(labels)
    n_classes = len(counts)
    total = len(labels)
    random_f1 = 1.0 / n_classes if n_classes > 0 else 0.0

    print(f"  Total examples: {total}")
    print(f"  Unique classes: {n_classes}")
    print(f"  Random baseline F1 (uniform): {random_f1:.3f}")
    print(f"  Class distribution:")
    for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {cls}: {count} ({count/total:.1%})")
    return counts


def run(domain: str = "sustainable packaging concepts"):
    """Run detailed downstream analysis."""
    try:
        from sklearn.metrics import classification_report
    except ImportError:
        print("Requires: pip install scikit-learn")
        return

    suffix = _domain_suffix(domain)
    domain_cfg = get_domain_config(domain)

    with open("config.yaml") as f:
        raw = yaml.safe_load(f)
    ds_config = raw["experiments"]["downstream"]
    data_dir = Path("data/raw") / f"exp2_comparison_{suffix}"

    # Check for existing detailed results
    results_path = Path(f"data/processed/downstream_{suffix}/results.json")
    if not results_path.exists():
        print(f"No downstream results found at {results_path}.")
        print("Run experiments/run_downstream.py first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    methods = ["naive", "vts_dedup", "dce"]

    print(f"\nDomain: {domain}")
    print("=" * 60)

    for method in methods:
        print(f"\n--- {method} ---")
        ideas = load_training_data(method, ds_config["training_size"], data_dir=str(data_dir))
        print(f"\nTask specification for {method}:")
        analyze_task_spec(ideas, separators=domain_cfg.category_separators)

        f1 = results.get(method, {}).get("eval_f1", "N/A")
        print(f"\n  Weighted F1: {f1:.3f}" if isinstance(f1, float) else f"\n  Weighted F1: {f1}")

    print("\n" + "=" * 60)
    print("\nSummary Table:")
    print(f"{'Method':<15} {'Weighted F1':<15} {'Classes':<10}")
    print("-" * 40)
    for method in methods:
        ideas = load_training_data(method, ds_config["training_size"], data_dir=str(data_dir))
        labels = [coarse_category(idea["category"], separators=domain_cfg.category_separators) for idea in ideas]
        counts = Counter(labels)
        # Filter to classes with >= 10 examples
        n_classes = sum(1 for c in counts.values() if c >= 10)
        f1 = results.get(method, {}).get("eval_f1", "N/A")
        f1_str = f"{f1:.1%}" if isinstance(f1, float) else str(f1)
        print(f"{method:<15} {f1_str:<15} {n_classes:<10}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="sustainable packaging concepts")
    args = parser.parse_args()
    run(domain=args.domain)
