"""Tests for semantic memory (uses mock embeddings to avoid API calls)."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

import numpy as np

from src.memory import SemanticMemory
from src.schemas import GeneratedIdea, StoredIdea


def _mock_embedding_client():
    """Create a mock embedding client that returns deterministic embeddings."""
    client = MagicMock()
    call_count = [0]

    def mock_embed(texts):
        """Generate deterministic embeddings based on text hash."""
        embeddings = []
        for text in texts:
            rng = np.random.RandomState(hash(text) % 2**31)
            emb = rng.randn(1536).tolist()
            # Normalize
            norm = np.linalg.norm(emb)
            emb = (np.array(emb) / norm).tolist()
            embeddings.append(emb)
        return embeddings

    client.embed = mock_embed
    return client


def test_add_and_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = SemanticMemory(
            db_path=tmpdir, embedding_client=_mock_embedding_client()
        )
        assert memory.count == 0

        ideas = [
            StoredIdea(
                id="1", name="Idea A", description="Description A",
                category="Marine", probability=0.05, batch_number=0, session_id="test",
            ),
            StoredIdea(
                id="2", name="Idea B", description="Description B",
                category="Aerospace", probability=0.03, batch_number=0, session_id="test",
            ),
        ]
        memory.add(ideas)
        assert memory.count == 2


def test_duplicate_detection():
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_client = MagicMock()

        # First call: store an idea with embedding [1, 0, 0, ...]
        base_emb = np.zeros(1536)
        base_emb[0] = 1.0

        # Near-duplicate: very similar embedding
        near_dup_emb = np.zeros(1536)
        near_dup_emb[0] = 0.99
        near_dup_emb[1] = 0.01
        near_dup_emb = (near_dup_emb / np.linalg.norm(near_dup_emb)).tolist()

        # Different: orthogonal embedding
        diff_emb = np.zeros(1536)
        diff_emb[1] = 1.0

        mock_client.embed = MagicMock(side_effect=[
            [base_emb.tolist()],     # add
            [near_dup_emb, diff_emb.tolist()],  # check_duplicates
        ])

        memory = SemanticMemory(
            db_path=tmpdir, similarity_threshold=0.85, embedding_client=mock_client
        )

        stored = StoredIdea(
            id="1", name="Base", description="Base idea",
            category="Test", probability=0.05, batch_number=0, session_id="test",
        )
        memory.add([stored])

        candidates = [
            GeneratedIdea(name="Near Dup", description="Near duplicate", category="Test", probability=0.05),
            GeneratedIdea(name="Different", description="Very different", category="Other", probability=0.03),
        ]
        accepted = memory.check_duplicates(candidates)
        # Near-dup should be rejected, Different should be accepted
        assert len(accepted) == 1
        assert accepted[0].name == "Different"


def test_category_distribution():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = SemanticMemory(
            db_path=tmpdir, embedding_client=_mock_embedding_client()
        )
        ideas = [
            StoredIdea(id="1", name="A", description="A", category="Marine",
                      probability=0.05, batch_number=0, session_id="t"),
            StoredIdea(id="2", name="B", description="B", category="Marine",
                      probability=0.05, batch_number=0, session_id="t"),
            StoredIdea(id="3", name="C", description="C", category="Aerospace",
                      probability=0.05, batch_number=0, session_id="t"),
        ]
        memory.add(ideas)
        dist = memory.category_distribution()
        assert dist["Marine"] == 2
        assert dist["Aerospace"] == 1


def test_empty_memory_operations():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = SemanticMemory(
            db_path=tmpdir, embedding_client=_mock_embedding_client()
        )
        assert memory.get_recent() == []
        assert memory.get_near_duplicates() == []
        assert memory.category_distribution() == {}
        assert memory.get_underrepresented_categories() == []
        assert len(memory.get_all_embeddings()) == 0
