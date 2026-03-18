"""Verbalized Tail Sampling filter."""

from src.schemas import GeneratedIdea


def filter_by_probability(
    ideas: list[GeneratedIdea], threshold: float = 0.10
) -> list[GeneratedIdea]:
    """Keep only ideas with self-assessed probability below threshold."""
    return [idea for idea in ideas if idea.probability < threshold]
