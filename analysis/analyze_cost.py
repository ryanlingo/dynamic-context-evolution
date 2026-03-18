"""Step 9, Item 11: Cost breakdown analysis.

Reads token usage logs from experiment runs and reports cost per method.
Uses OpenAI pricing for gpt-5-mini and text-embedding-3-small.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonlines


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


# Approximate pricing (per 1M tokens) — update as needed
PRICING = {
    "gpt-5-mini-2025-08-07": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02},
}


def load_token_usage(method: str, domain: str = "sustainable packaging concepts") -> dict | None:
    """Load token usage log for a method if it exists."""
    suffix = _domain_suffix(domain)
    path = Path(f"data/raw/exp2_comparison_{suffix}/{method}/token_usage.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def estimate_cost(usage: dict, model: str = "gpt-5-mini-2025-08-07") -> float:
    """Estimate cost from token usage."""
    pricing = PRICING.get(model, {"input": 0.15, "output": 0.60})
    input_cost = usage.get("prompt_tokens", 0) / 1_000_000 * pricing["input"]
    output_cost = usage.get("completion_tokens", 0) / 1_000_000 * pricing.get("output", 0)
    return input_cost + output_cost


def run(domain: str = "sustainable packaging concepts"):
    suffix = _domain_suffix(domain)
    methods = ["naive", "vts_only", "vts_dedup", "dce",
               "dedup_only", "prompt_evo_only", "prompt_evo_dedup",
               "temp_1.2_dedup", "nucleus_0.9_dedup"]

    print(f"\nDomain: {domain}")
    print("=" * 70)
    print(f"{'Method':<25} {'Prompt Tok':<15} {'Completion Tok':<15} {'Est. Cost':<12}")
    print("-" * 70)

    total_cost = 0.0
    for method in methods:
        usage = load_token_usage(method, domain=domain)
        if usage is None:
            continue
        cost = estimate_cost(usage)
        total_cost += cost
        print(f"{method:<25} {usage.get('prompt_tokens', 0):<15,} "
              f"{usage.get('completion_tokens', 0):<15,} ${cost:<11.4f}")

    print("-" * 70)
    print(f"{'TOTAL':<25} {'':<15} {'':<15} ${total_cost:<11.4f}")
    print("=" * 70)

    # Estimate embedding cost from idea counts
    for method in methods:
        results_path = Path(f"data/raw/exp2_comparison_{suffix}/{method}/results.jsonl")
        if results_path.exists():
            n_ideas = 0
            with jsonlines.open(str(results_path)) as reader:
                for batch in reader:
                    n_ideas += len(batch["ideas"])
            # Rough estimate: ~100 tokens per idea embedding
            emb_cost = n_ideas * 100 / 1_000_000 * 0.02
            if n_ideas > 0:
                print(f"  {method} embedding cost (est): ${emb_cost:.4f} ({n_ideas} ideas)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="sustainable packaging concepts")
    args = parser.parse_args()
    run(domain=args.domain)
