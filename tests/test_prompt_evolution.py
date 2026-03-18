"""Tests for prompt evolution strategies."""

from src.prompt_evolution import get_strategy_name, get_phase, STRATEGY_NAMES


def test_round_robin_strategy():
    assert get_strategy_name(0) == "gap"
    assert get_strategy_name(1) == "inversion"
    assert get_strategy_name(2) == "cross_industry"
    assert get_strategy_name(3) == "constraint"
    assert get_strategy_name(4) == "gap"  # wraps around
    assert get_strategy_name(7) == "constraint"


def test_phase_exploration():
    # First 40% of 200 batches = batches 0-79
    assert get_phase(0, 200, 0.40) == "exploration"
    assert get_phase(79, 200, 0.40) == "exploration"


def test_phase_exploitation():
    assert get_phase(80, 200, 0.40) == "exploitation"
    assert get_phase(199, 200, 0.40) == "exploitation"


def test_phase_custom_threshold():
    # 60/40 split: first 60% = batches 0-119
    assert get_phase(119, 200, 0.60) == "exploration"
    assert get_phase(120, 200, 0.60) == "exploitation"


def test_all_strategies_covered():
    strategies = {get_strategy_name(i) for i in range(4)}
    assert strategies == set(STRATEGY_NAMES)
