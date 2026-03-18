"""Step 4: Validate EDV with an independent embedding model.

Re-embeds all exp2 ideas with sentence-transformers/all-MiniLM-L6-v2
and recomputes EDV + collapse rate. Also tests threshold robustness
at delta in {0.80, 0.85, 0.90}.
"""

from __future__ import annotations

from pathlib import Path

import jsonlines
import numpy as np
import matplotlib.pyplot as plt

from analysis.plot_utils import setup_style, save_figure

METHODS = ["naive", "vts_only", "vts_dedup", "dce"]
THRESHOLDS = [0.80, 0.85, 0.90]


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def load_ideas(method: str, domain: str = "sustainable packaging concepts") -> list[dict]:
    """Load all ideas for a method from exp2 results."""
    suffix = _domain_suffix(domain)
    path = f"data/raw/exp2_comparison_{suffix}/{method}/results.jsonl"
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


def compute_edv_series(batch_embeddings: list[np.ndarray], batch_probs: list[list[float]]) -> list[float]:
    """Compute EDV per batch using provided embeddings."""
    edvs = []
    all_prior = []
    for embs, probs in zip(batch_embeddings, batch_probs):
        if len(embs) == 0:
            edvs.append(0.0)
            continue
        prior = np.array(all_prior) if all_prior else np.array([])
        depths = [1.0 - p for p in probs]
        if prior.size == 0:
            breadths = [1.0] * len(probs)
        else:
            sims = embs @ prior.T
            breadths = [1.0 - float(np.max(sims[i])) for i in range(len(probs))]
        edv = np.mean([d * b for d, b in zip(depths, breadths)])
        edvs.append(float(edv))
        all_prior.extend(embs.tolist())
    return edvs


def compute_collapse_rate(all_embs: np.ndarray, threshold: float = 0.85) -> float:
    """Collapse rate: fraction of late ideas similar to early ideas."""
    n = len(all_embs)
    ideas_per_batch = 5
    early_end = min(50 * ideas_per_batch, n)
    late_start = max(0, n - 50 * ideas_per_batch)
    early = all_embs[:early_end]
    late = all_embs[late_start:]
    if len(early) == 0 or len(late) == 0:
        return 0.0
    sims = late @ early.T
    max_sims = np.max(sims, axis=1)
    return float(np.mean(max_sims > threshold))


def run(domain: str = "sustainable packaging concepts"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Requires: pip install sentence-transformers")
        return

    suffix = _domain_suffix(domain)
    print(f"Loading independent embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    output_dir = Path(f"data/processed/edv_validation_{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- EDV comparison table ---
    print(f"\nDomain: {domain}")
    print("\n" + "=" * 70)
    print(f"{'Method':<15} {'EDV ret (OpenAI)':<20} {'EDV ret (MiniLM)':<20}")
    print("-" * 70)

    all_results = {}
    for method in METHODS:
        batches = load_ideas(method, domain=domain)
        probs = [[idea["probability"] for idea in b["ideas"]] for b in batches]

        # Re-embed with MiniLM
        batch_embs = embed_all(batches, model)
        edvs = compute_edv_series(batch_embs, probs)

        # Flatten embeddings
        all_embs = np.concatenate([e for e in batch_embs if len(e) > 0])

        # Load original OpenAI EDV for comparison
        orig_path = Path(f"data/processed/exp2_comparison_{suffix}/{method}_metrics.npz")
        if orig_path.exists():
            orig = np.load(orig_path, allow_pickle=True)
            orig_edvs = orig["edvs"]
            orig_ret = (orig_edvs[-1] / orig_edvs[0] * 100) if orig_edvs[0] > 0 else 0
        else:
            orig_ret = float("nan")

        new_ret = (edvs[-1] / edvs[0] * 100) if edvs and edvs[0] > 0 else 0

        print(f"{method:<15} {orig_ret:<20.1f}% {new_ret:<20.1f}%")

        all_results[method] = {
            "edvs": edvs,
            "all_embs": all_embs,
            "edv_retention_minilm": new_ret,
            "edv_retention_openai": orig_ret,
        }

    print("=" * 70)

    # --- Threshold robustness ---
    print(f"\n{'Method':<15}", end="")
    for t in THRESHOLDS:
        print(f"  CR@{t:<10}", end="")
    print()
    print("-" * 55)

    for method in METHODS:
        r = all_results[method]
        print(f"{method:<15}", end="")
        for t in THRESHOLDS:
            cr = compute_collapse_rate(r["all_embs"], threshold=t)
            print(f"  {cr:<10.1%}", end="")
        print()

    # Save results
    for method, r in all_results.items():
        np.savez(
            output_dir / f"{method}_validation.npz",
            edvs=r["edvs"],
            edv_retention_minilm=r["edv_retention_minilm"],
            edv_retention_openai=r["edv_retention_openai"],
        )
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="sustainable packaging concepts")
    args = parser.parse_args()
    run(domain=args.domain)
