"""ChromaDB-backed semantic memory for DCE."""

from __future__ import annotations

import uuid
from collections import Counter

import chromadb
import numpy as np

from src.embeddings import EmbeddingClient
from src.schemas import GeneratedIdea, StoredIdea


class SemanticMemory:
    def __init__(
        self,
        db_path: str = "data/chroma_db",
        collection_name: str = "ideas",
        similarity_threshold: float = 0.85,
        embedding_client: EmbeddingClient | None = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.embedding_client = embedding_client or EmbeddingClient()
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self.collection.count()

    def add(self, ideas: list[StoredIdea]) -> None:
        """Add ideas with their embeddings to the memory bank."""
        if not ideas:
            return
        texts = [f"{idea.name}: {idea.description}" for idea in ideas]
        embeddings = self.embedding_client.embed(texts)
        self.collection.add(
            ids=[idea.id for idea in ideas],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "name": idea.name,
                    "category": idea.category,
                    "probability": idea.probability,
                    "batch_number": idea.batch_number,
                    "session_id": idea.session_id,
                }
                for idea in ideas
            ],
        )
        for idea, emb in zip(ideas, embeddings):
            idea.embedding = emb

    def check_duplicates(
        self, ideas: list[GeneratedIdea]
    ) -> list[GeneratedIdea]:
        """Return only ideas that are NOT duplicates (similarity < threshold)."""
        if self.count == 0:
            return ideas

        texts = [f"{idea.name}: {idea.description}" for idea in ideas]
        embeddings = self.embedding_client.embed(texts)
        accepted = []
        for idea, emb in zip(ideas, embeddings):
            results = self.collection.query(
                query_embeddings=[emb], n_results=1
            )
            if not results["distances"][0]:
                accepted.append(idea)
                continue
            # ChromaDB cosine distance = 1 - cosine_similarity when space="cosine"
            cosine_distance = results["distances"][0][0]
            cosine_similarity = 1 - cosine_distance
            if cosine_similarity < self.similarity_threshold:
                accepted.append(idea)
        return accepted

    def get_recent(self, n: int = 10) -> list[dict]:
        """Get the most recently added ideas."""
        if self.count == 0:
            return []
        results = self.collection.get(
            limit=min(n, self.count),
            include=["documents", "metadatas"],
        )
        # ChromaDB returns in insertion order; take the last n
        docs = results["documents"] or []
        metas = results["metadatas"] or []
        items = list(zip(docs, metas))
        return [
            {"text": doc, "category": meta.get("category", ""), "name": meta.get("name", "")}
            for doc, meta in items[-n:]
        ]

    def get_near_duplicates(self, n: int = 5, novelty_threshold: float = 0.3) -> list[str]:
        """Get ideas that are close to many others (low novelty regions to avoid)."""
        if self.count < 2:
            return []
        all_data = self.collection.get(include=["embeddings", "documents"])
        embeddings = np.array(all_data["embeddings"])
        documents = all_data["documents"]

        # Compute pairwise cosine similarities
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = embeddings / norms
        sim_matrix = normalized @ normalized.T
        np.fill_diagonal(sim_matrix, 0)

        # Average similarity per idea — high avg means it's in a dense region
        avg_sim = sim_matrix.mean(axis=1)
        dense_indices = np.argsort(avg_sim)[-n:]
        return [documents[i] for i in dense_indices]

    def category_distribution(self) -> dict[str, int]:
        """Get category counts across all stored ideas."""
        if self.count == 0:
            return {}
        results = self.collection.get(include=["metadatas"])
        categories = [m.get("category", "Unknown") for m in results["metadatas"]]
        return dict(Counter(categories))

    def get_underrepresented_categories(
        self, saturation_multiplier: float = 1.5, min_categories: int = 3
    ) -> list[str]:
        """Return categories below average count."""
        dist = self.category_distribution()
        if not dist:
            return []
        avg = sum(dist.values()) / len(dist)
        under = [cat for cat, count in dist.items() if count < avg]
        # If too few underrepresented, return the smallest ones
        if len(under) < min_categories:
            sorted_cats = sorted(dist.items(), key=lambda x: x[1])
            under = [cat for cat, _ in sorted_cats[:min_categories]]
        return under

    def get_all_embeddings(self) -> np.ndarray:
        """Get all stored embeddings as a numpy array."""
        if self.count == 0:
            return np.array([])
        results = self.collection.get(include=["embeddings"])
        return np.array(results["embeddings"])

    def get_all_ideas(self) -> list[dict]:
        """Get all stored ideas with metadata."""
        if self.count == 0:
            return []
        results = self.collection.get(include=["documents", "metadatas", "embeddings"])
        return [
            {
                "text": doc,
                "embedding": emb,
                **meta,
            }
            for doc, meta, emb in zip(
                results["documents"], results["metadatas"], results["embeddings"]
            )
        ]
