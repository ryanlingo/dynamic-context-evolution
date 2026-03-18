"""Analyze downstream validation: F1 results table."""

from __future__ import annotations

import json
from pathlib import Path


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def run(domain: str = "sustainable packaging concepts"):
    suffix = _domain_suffix(domain)
    results_path = Path(f"data/processed/downstream_{suffix}/results.json")
    if not results_path.exists():
        print(f"No downstream results found at {results_path}. Run experiments/run_downstream.py first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    print(f"\nDomain: {domain}")
    print("=" * 50)
    print(f"{'Training Data':<20} {'F1 (classification)':<30}")
    print("-" * 50)
    for method, metrics in results.items():
        f1 = metrics.get("eval_f1", "N/A")
        if isinstance(f1, float):
            print(f"{method:<20} {f1:.1%}")
        else:
            print(f"{method:<20} {f1}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="sustainable packaging concepts")
    args = parser.parse_args()
    run(domain=args.domain)
