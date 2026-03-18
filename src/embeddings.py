"""Thin wrapper around OpenAI text-embedding-3-small."""

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class EmbeddingClient:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns list of 1536-dim vectors."""
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed([text])[0]
