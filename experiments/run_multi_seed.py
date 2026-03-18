"""Step 6: Multi-seed experiment runner for statistical rigor.

Wraps run_exp2_comparison to run multiple seeds, then
analysis/analyze_multi_seed.py computes mean +/- std with significance tests.
"""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path

import yaml
from dotenv import load_dotenv

from experiments.run_exp2_comparison import load_config, run_method, _domain_suffix

load_dotenv()

DEFAULT_SEEDS = [42, 123, 456]


def run(
    seeds: list[int] | None = None,
    methods: list[str] | None = None,
    domain_override: str | None = None,
):
    seeds = seeds or DEFAULT_SEEDS
    config = load_config(domain_override)
    suffix = _domain_suffix(config.domain)
    methods = methods or ["naive", "vts_only", "vts_dedup", "dce"]

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed} ({seed_idx + 1}/{len(seeds)})")
        print(f"{'='*60}")

        seed_suffix = f"{suffix}_seed{seed}"
        for method in methods:
            print(f"\n  Method: {method}, Seed: {seed}")
            # Each seed gets its own output directory
            run_method(method, config, domain_suffix=seed_suffix)

    # Write seed manifest
    manifest = {
        "seeds": seeds,
        "methods": methods,
        "domain": config.domain,
        "total_batches": config.total_batches,
    }
    manifest_dir = Path(config.output_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_dir / f"multi_seed_manifest_{suffix}.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nMulti-seed runs complete. {len(seeds)} seeds x {len(methods)} methods.")
    print(f"Run: python -m analysis.analyze_multi_seed --domain '{config.domain}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed experiment runner")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()
    run(seeds=args.seeds, methods=args.methods, domain_override=args.domain)
