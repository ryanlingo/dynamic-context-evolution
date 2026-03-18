"""Step 6: VTS rejection analysis — confusion matrix of VTS vs dedup decisions.

Uses naive generation data (no filtering applied) to evaluate what VTS
and dedup would independently accept or reject, building a 2x2 confusion
matrix. Key question: what fraction of VTS-rejected ideas are actually
semantically novel?
"""

from __future__ import annotations

from pathlib import Path

import jsonlines
import numpy as np

SIMILARITY_THRESHOLD = 0.85
VTS_PROBABILITY_THRESHOLD = 0.10


def _domain_suffix(domain: str) -> str:
    return domain.replace(" ", "_")[:40]


def load_naive_ideas(domain: str) -> list[dict]:
    """Load all ideas from naive method (unfiltered generation)."""
    suffix = _domain_suffix(domain)
    path = Path(f"data/raw/exp2_comparison_{suffix}/naive/results.jsonl")
    ideas = []
    with jsonlines.open(str(path)) as reader:
        for batch in reader:
            for idea in batch["ideas"]:
                ideas.append(idea)
    return ideas


def run(domain: str = "sustainable packaging concepts"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Requires: pip install sentence-transformers")
        return

    suffix = _domain_suffix(domain)
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    ideas = load_naive_ideas(domain=domain)
    print(f"Loaded {len(ideas)} naive ideas")

    # Embed all ideas
    texts = [f"{idea['name']}: {idea['description']}" for idea in ideas]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings)

    # For each idea, determine VTS and dedup decisions incrementally
    vts_accept = []  # bool per idea
    dedup_accept = []  # bool per idea
    memory: list[np.ndarray] = []  # incrementally built memory bank

    for i, idea in enumerate(ideas):
        # VTS decision: accept if probability < threshold
        prob = idea.get("probability", 0.0)
        vts_accepts = prob < VTS_PROBABILITY_THRESHOLD
        vts_accept.append(vts_accepts)

        # Dedup decision: accept if max similarity to memory < threshold
        emb = embeddings[i]
        if len(memory) == 0:
            dedup_accepts = True
        else:
            memory_arr = np.array(memory)
            sims = emb @ memory_arr.T
            max_sim = float(np.max(sims))
            dedup_accepts = max_sim <= SIMILARITY_THRESHOLD

        dedup_accept.append(dedup_accepts)

        # Add to memory (simulating incremental memory bank)
        memory.append(emb.tolist())

    vts_accept = np.array(vts_accept)
    dedup_accept = np.array(dedup_accept)
    vts_reject = ~vts_accept
    dedup_reject = ~dedup_accept

    # Build 2x2 confusion matrix
    # Rows: VTS decision, Columns: Dedup decision
    both_accept = int(np.sum(vts_accept & dedup_accept))
    vts_acc_dedup_rej = int(np.sum(vts_accept & dedup_reject))
    vts_rej_dedup_acc = int(np.sum(vts_reject & dedup_accept))
    both_reject = int(np.sum(vts_reject & dedup_reject))

    total = len(ideas)

    print(f"\nDomain: {domain}")
    print(f"VTS threshold: probability >= {VTS_PROBABILITY_THRESHOLD}")
    print(f"Dedup threshold: cosine similarity > {SIMILARITY_THRESHOLD}")
    print(f"Total ideas: {total}")

    print("\n" + "=" * 55)
    print(f"{'':20} {'Dedup Accept':<18} {'Dedup Reject':<18}")
    print("-" * 55)
    print(
        f"{'VTS Accept':<20} "
        f"{both_accept:>5} ({both_accept/total*100:5.1f}%)    "
        f"{vts_acc_dedup_rej:>5} ({vts_acc_dedup_rej/total*100:5.1f}%)"
    )
    print(
        f"{'VTS Reject':<20} "
        f"{vts_rej_dedup_acc:>5} ({vts_rej_dedup_acc/total*100:5.1f}%)    "
        f"{both_reject:>5} ({both_reject/total*100:5.1f}%)"
    )
    print("=" * 55)

    # Key statistic
    total_vts_rejected = int(np.sum(vts_reject))
    if total_vts_rejected > 0:
        novel_among_rejected = vts_rej_dedup_acc / total_vts_rejected * 100
        print(
            f"\nOf {total_vts_rejected} VTS-rejected ideas, "
            f"{vts_rej_dedup_acc} ({novel_among_rejected:.1f}%) are semantically "
            f"novel (would pass dedup)."
        )
    else:
        print("\nNo ideas were rejected by VTS.")
        novel_among_rejected = 0.0

    total_dedup_rejected = int(np.sum(dedup_reject))
    total_vts_accepted = int(np.sum(vts_accept))
    print(f"\nSummary:")
    print(f"  VTS acceptance rate:  {total_vts_accepted}/{total} ({total_vts_accepted/total*100:.1f}%)")
    print(f"  Dedup rejection rate: {total_dedup_rejected}/{total} ({total_dedup_rejected/total*100:.1f}%)")
    print(f"  Agreement rate:       {(both_accept + both_reject)/total*100:.1f}%")

    # Save results
    output_dir = Path(f"data/processed/vts_confusion_{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "confusion_matrix.npz",
        confusion_matrix=np.array([
            [both_accept, vts_acc_dedup_rej],
            [vts_rej_dedup_acc, both_reject],
        ]),
        labels=np.array(["vts_accept_dedup_accept", "vts_accept_dedup_reject",
                          "vts_reject_dedup_accept", "vts_reject_dedup_reject"]),
        total=total,
        vts_threshold=VTS_PROBABILITY_THRESHOLD,
        dedup_threshold=SIMILARITY_THRESHOLD,
        novel_among_vts_rejected_pct=novel_among_rejected,
    )
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="sustainable packaging concepts")
    args = parser.parse_args()
    run(domain=args.domain)
