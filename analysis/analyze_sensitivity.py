"""Analyze sensitivity: exploration/exploitation split table."""

from __future__ import annotations

from pathlib import Path

import jsonlines
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from src.embeddings import EmbeddingClient
from src.metrics import edv_batch, collapse_rate


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def load_results(split_label: str, domain: str = "sustainable packaging concepts") -> list[dict]:
    suffix = _domain_suffix(domain)
    path = f"data/raw/sensitivity_{suffix}/{split_label}/results.jsonl"
    batches = []
    with jsonlines.open(path) as reader:
        for batch in reader:
            batches.append(batch)
    return batches


def compute_metrics(batches: list[dict]) -> dict:
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
    early_end = 50 * ideas_per_batch
    late_start = max(0, len(all_embeddings) - 50 * ideas_per_batch)
    cr = collapse_rate(
        all_emb_arr[:early_end],
        all_emb_arr[late_start:],
    )

    return {
        "edvs": edvs,
        "collapse_rate": cr,
        "edv_at_200_pct": (edvs[-1] / edvs[0] * 100) if edvs and edvs[0] > 0 else 0,
    }


def run(domain: str = "sustainable packaging concepts"):
    suffix = _domain_suffix(domain)
    # Auto-discover splits from data directory
    sens_dir = Path(f"data/raw/sensitivity_{suffix}")
    splits = sorted(
        int(d.name.split("_")[1]) / 100
        for d in sens_dir.iterdir()
        if d.is_dir() and d.name.startswith("split_")
    )
    print("\n" + "=" * 50)
    print(f"{'Split':<12} {'EDV@200':<15} {'Collapse Rate':<15}")
    print("-" * 50)

    output_dir = Path(f"data/processed/sensitivity_{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        label = f"split_{int(split*100)}_{int((1-split)*100)}"
        batches = load_results(label, domain=domain)
        m = compute_metrics(batches)
        print(f"{int(split*100)}/{int((1-split)*100):<8} {m['edv_at_200_pct']:.1f}%{'':<10} {m['collapse_rate']:.1%}")
        np.savez(output_dir / f"{label}_metrics.npz", **m)

    print("=" * 50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="sustainable packaging concepts")
    args = parser.parse_args()
    run(domain=args.domain)
