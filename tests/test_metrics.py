"""Tests for metrics computations."""

import numpy as np

from src.metrics import batch_novelty, edv_batch, collapse_rate, cluster_count


def test_batch_novelty_first_batch():
    new = np.random.randn(5, 1536)
    prior = np.array([])
    assert batch_novelty(new, prior) == 1.0


def test_batch_novelty_identical():
    """Identical embeddings should have zero novelty."""
    embs = np.random.randn(3, 1536)
    nov = batch_novelty(embs, embs)
    assert nov < 0.01


def test_batch_novelty_orthogonal():
    """Orthogonal embeddings should have high novelty."""
    e1 = np.eye(1536)[:3]  # 3 orthogonal unit vectors
    e2 = np.eye(1536)[3:6]
    nov = batch_novelty(e2, e1)
    assert nov > 0.9


def test_edv_first_batch():
    probs = np.array([0.05, 0.03, 0.08])
    new_embs = np.random.randn(3, 1536)
    prior = np.array([])
    edv = edv_batch(probs, new_embs, prior)
    # First batch: breadth = 1.0, so EDV = mean(1 - prob)
    expected = np.mean(1 - probs)
    assert abs(edv - expected) < 0.01


def test_edv_empty():
    assert edv_batch(np.array([]), np.array([]), np.array([])) == 0.0


def test_collapse_rate_identical():
    embs = np.random.randn(10, 1536)
    cr = collapse_rate(embs, embs, threshold=0.85)
    assert cr == 1.0  # all late ideas are identical to early ones


def test_collapse_rate_orthogonal():
    early = np.eye(1536)[:10]
    late = np.eye(1536)[10:20]
    cr = collapse_rate(early, late, threshold=0.85)
    assert cr == 0.0


def test_collapse_rate_empty():
    assert collapse_rate(np.array([]), np.random.randn(5, 10)) == 0.0


def test_cluster_count_basic():
    # Generate 3 clusters of points
    rng = np.random.RandomState(42)
    c1 = rng.randn(20, 10) + np.array([10, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    c2 = rng.randn(20, 10) + np.array([0, 10, 0, 0, 0, 0, 0, 0, 0, 0])
    c3 = rng.randn(20, 10) + np.array([0, 0, 10, 0, 0, 0, 0, 0, 0, 0])
    embs = np.vstack([c1, c2, c3])
    cc = cluster_count(embs, min_cluster_size=5)
    assert cc >= 2  # should find at least 2 of the 3 clusters


def test_cluster_count_too_few():
    embs = np.random.randn(3, 10)
    assert cluster_count(embs, min_cluster_size=5) == 0
