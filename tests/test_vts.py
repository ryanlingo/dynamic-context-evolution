"""Tests for verbalized tail sampling filter."""

from src.schemas import GeneratedIdea
from src.vts import filter_by_probability


def _idea(probability: float) -> GeneratedIdea:
    return GeneratedIdea(
        name="Test", description="Test idea", category="Test", probability=probability
    )


def test_filters_high_probability():
    ideas = [_idea(0.05), _idea(0.15), _idea(0.50)]
    result = filter_by_probability(ideas, threshold=0.10)
    assert len(result) == 1
    assert result[0].probability == 0.05


def test_keeps_all_below_threshold():
    ideas = [_idea(0.01), _idea(0.02), _idea(0.09)]
    result = filter_by_probability(ideas, threshold=0.10)
    assert len(result) == 3


def test_empty_input():
    assert filter_by_probability([], threshold=0.10) == []


def test_boundary_excluded():
    """Probability exactly at threshold should be excluded (strict <)."""
    ideas = [_idea(0.10)]
    result = filter_by_probability(ideas, threshold=0.10)
    assert len(result) == 0
