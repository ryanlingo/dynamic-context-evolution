"""Metrics: EDV, batch novelty, collapse rate, HDBSCAN cluster count."""

from __future__ import annotations

import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def batch_novelty(
    new_embeddings: np.ndarray, all_prior_embeddings: np.ndarray
) -> float:
    """Average minimum cosine distance of new ideas to all prior ideas.

    Returns 1.0 if there are no prior embeddings (first batch).
    """
    if len(all_prior_embeddings) == 0 or len(new_embeddings) == 0:
        return 1.0
    sim = cosine_similarity(new_embeddings, all_prior_embeddings)
    # Minimum distance = 1 - max similarity for each new idea
    min_distances = 1 - sim.max(axis=1)
    return float(min_distances.mean())


def edv_batch(
    probabilities: np.ndarray,
    new_embeddings: np.ndarray,
    all_prior_embeddings: np.ndarray,
) -> float:
    """Effective Diversity Volume for a batch.

    EDV = mean(depth_i * breadth_i)
    depth_i = 1 - probability_i
    breadth_i = min cosine distance to memory bank
    """
    if len(new_embeddings) == 0:
        return 0.0

    depths = 1 - probabilities

    if len(all_prior_embeddings) == 0:
        # First batch: breadth is 1.0 (nothing to compare against)
        breadths = np.ones(len(new_embeddings))
    else:
        sim = cosine_similarity(new_embeddings, all_prior_embeddings)
        breadths = 1 - sim.max(axis=1)

    return float((depths * breadths).mean())


def collapse_rate(
    early_embeddings: np.ndarray,
    late_embeddings: np.ndarray,
    threshold: float = 0.85,
) -> float:
    """Percentage of late ideas with cosine similarity > threshold to any early idea.

    early_embeddings: ideas from first 50 batches
    late_embeddings: ideas from last 50 batches
    """
    if len(early_embeddings) == 0 or len(late_embeddings) == 0:
        return 0.0
    sim = cosine_similarity(late_embeddings, early_embeddings)
    max_sim = sim.max(axis=1)
    return float((max_sim > threshold).mean())


def cluster_count(embeddings: np.ndarray, min_cluster_size: int = 5) -> int:
    """Number of distinct clusters found by HDBSCAN."""
    if len(embeddings) < min_cluster_size:
        return 0
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)
    # -1 is noise; count only real clusters
    return len(set(labels) - {-1})
