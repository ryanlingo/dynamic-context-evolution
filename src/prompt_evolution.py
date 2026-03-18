"""Adaptive prompt evolution with 4 strategies and phase logic."""

from __future__ import annotations

import random
from pathlib import Path

from src.domain_config import get_domain_config
from src.memory import SemanticMemory
from src.schemas import ExperimentConfig

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

STRATEGY_NAMES = ["gap", "inversion", "cross_industry", "constraint"]


def _load_template(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def get_strategy_name(batch_number: int) -> str:
    """Round-robin strategy selection: batch_number % 4."""
    return STRATEGY_NAMES[batch_number % 4]


def get_phase(batch_number: int, total_batches: int, phase_threshold: float = 0.40) -> str:
    """Exploration for first phase_threshold fraction, then exploitation."""
    if batch_number < total_batches * phase_threshold:
        return "exploration"
    return "exploitation"


def build_strategy_instruction(
    strategy: str,
    memory: SemanticMemory,
    config: ExperimentConfig,
) -> str:
    """Build the strategy-specific instruction block."""
    domain_cfg = get_domain_config(config.domain)

    if strategy == "gap":
        template = _load_template("strategy_gap.txt")
        under = memory.get_underrepresented_categories()
        if not under:
            under = domain_cfg.fallback_categories
        return template.format(
            underrepresented_categories=", ".join(under),
            domain=config.domain,
        )

    elif strategy == "inversion":
        template = _load_template("strategy_inversion.txt")
        recent = memory.get_recent(5)
        if recent:
            assumptions = "\n".join(
                f"- {item['name']}: likely assumes mainstream use case"
                for item in recent
            )
        else:
            assumptions = "- No recent ideas yet. Invert common assumptions about the domain."
        return template.format(assumptions=assumptions)

    elif strategy == "cross_industry":
        template = _load_template("strategy_cross_industry.txt")
        sampled = random.sample(domain_cfg.industries, 3)
        return template.format(
            stimulus_industries=", ".join(sampled),
            random_industry=sampled[0],
            another_industry=sampled[1],
            domain=config.domain,
        )

    elif strategy == "constraint":
        template = _load_template("strategy_constraint.txt")
        constraint = random.choice(domain_cfg.constraints)
        return template.format(constraint=constraint)

    return ""


def build_phase_instruction(phase: str, config: ExperimentConfig) -> str:
    """Build phase-specific instruction."""
    if phase == "exploration":
        return (
            "PHASE: Exploration. Cast a wide net. Prioritize breadth and novelty. "
            "Generate ideas from categories and perspectives not yet covered. "
            "Be as diverse as possible."
        )
    return (
        "PHASE: Gap-filling. Target underrepresented areas. "
        "Focus on categories with fewer ideas. Fill gaps in the collection. "
        "Prioritize coverage over raw novelty."
    )


def build_prompt(
    batch_number: int,
    memory: SemanticMemory,
    config: ExperimentConfig,
    method: str = "dce",
) -> str:
    """Build the full generation prompt for a given batch."""
    base = _load_template("base_generation.txt")

    # VTS instruction (used in vts_only, vts_dedup, dce)
    if method in ("vts_only", "vts_dedup", "dce"):
        vts_instruction = _load_template("vts_instruction.txt")
    else:
        vts_instruction = ""

    # Strategy instruction (used in dce only)
    if method == "dce":
        strategy = get_strategy_name(batch_number)
        strategy_instruction = build_strategy_instruction(strategy, memory, config)
    else:
        strategy = "none"
        strategy_instruction = ""

    # Phase instruction (used in dce only)
    if method == "dce":
        phase = get_phase(batch_number, config.total_batches, config.phase_threshold)
        phase_instruction = build_phase_instruction(phase, config)
    else:
        phase = "none"
        phase_instruction = ""

    # Recent ideas
    recent = memory.get_recent(config.recent_ideas_in_prompt)
    if recent:
        recent_text = "\n".join(f"- {item['name']}: {item['text']}" for item in recent)
    else:
        recent_text = "(None yet — this is the first batch.)"

    # Near-duplicates to avoid
    near_dupes = memory.get_near_duplicates(config.near_duplicates_shown)
    if near_dupes:
        near_dupes_text = "\n".join(f"- {d}" for d in near_dupes)
    else:
        near_dupes_text = "(None yet.)"

    # Category distribution
    dist = memory.category_distribution()
    if dist:
        dist_text = "\n".join(f"- {cat}: {count}" for cat, count in sorted(dist.items()))
    else:
        dist_text = "(No ideas generated yet.)"

    domain_cfg = get_domain_config(config.domain)

    return base.format(
        batch_size=config.batch_size,
        domain=config.domain,
        persona=domain_cfg.persona,
        vts_instruction=vts_instruction,
        strategy_instruction=strategy_instruction,
        phase_instruction=phase_instruction,
        recent_ideas=recent_text,
        near_duplicates=near_dupes_text,
        category_distribution=dist_text,
    )
