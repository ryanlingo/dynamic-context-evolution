"""Domain-specific configuration registry for DCE pipeline.

Maps domain names to persona, industries, constraints, fallback categories,
and category separators so the pipeline can run on any domain without
hardcoded assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainConfig:
    """All domain-specific knobs for the generation pipeline."""

    persona: str
    industries: list[str]
    constraints: list[str]
    fallback_categories: list[str]
    category_separators: list[str] = field(default_factory=lambda: ["/", "&"])


DOMAIN_REGISTRY: dict[str, DomainConfig] = {
    "sustainable packaging concepts": DomainConfig(
        persona="a creative product innovation expert",
        industries=[
            "hospitality", "agriculture", "gaming", "fashion", "finance",
            "healthcare", "construction", "logistics", "education", "entertainment",
            "aerospace", "marine", "automotive", "telecommunications", "energy",
        ],
        constraints=[
            "Must cost nothing to implement",
            "Unlimited budget available",
            "Must be deployable within 3 months",
            "Designed for a 10-year R&D timeline",
            "Must work without electricity",
            "Must be entirely edible or compostable within 30 days",
            "Must be manufactured from a single material",
            "Must serve a dual purpose beyond packaging",
            "Must be designed for reuse at least 100 times",
            "Must work in extreme temperatures (-40°C to 60°C)",
        ],
        fallback_categories=[
            "Marine", "Aerospace", "Lifestyle", "Agriculture", "Healthcare",
        ],
        category_separators=["/", "&"],
    ),
    "creative writing prompts": DomainConfig(
        persona="a creative writing instructor and published author",
        industries=[
            "psychology", "mythology", "journalism", "architecture",
            "marine biology", "archaeology", "culinary arts", "astronomy",
            "forensics", "anthropology", "music", "dance", "philosophy",
            "meteorology", "linguistics",
        ],
        constraints=[
            "Must be completable in under 500 words",
            "Must involve exactly two characters",
            "Must take place in a single room",
            "Must include a specific sensory detail as a central element",
            "Must subvert a common genre trope",
            "Must be suitable for middle school students",
            "Must incorporate a real historical event",
            "Must begin with dialogue",
            "Must involve a moral dilemma",
            "Must use second-person point of view",
        ],
        fallback_categories=[
            "Fiction", "Poetry", "Nonfiction", "Screenwriting", "Flash Fiction",
        ],
        category_separators=["/", "&", "-"],
    ),
    "educational exam questions for introductory biology and computer science": DomainConfig(
        persona="an expert educator and assessment designer",
        industries=[
            "K-12 education", "higher education", "corporate training",
            "medical education", "engineering education", "vocational training",
            "online learning platforms", "test preparation services",
            "educational publishing", "EdTech startups",
            "government certification", "military training",
            "language learning", "special education", "STEM outreach",
        ],
        constraints=[
            "Must be answerable without any external resources",
            "Must require synthesis across two or more topics",
            "Must be solvable in under 2 minutes",
            "Must require a diagram or figure to answer",
            "Must be accessible to non-native English speakers",
            "Must involve a real-world scenario or application",
            "Must test a common misconception",
            "Must be suitable for automated grading",
            "Must include quantitative reasoning",
            "Must be appropriate for open-book examination",
        ],
        fallback_categories=[
            "Cell Biology", "Genetics", "Data Structures",
            "Algorithms", "Ecology",
        ],
        category_separators=["/", "&", "-", ":"],
    ),
}


def get_domain_config(domain: str) -> DomainConfig:
    """Look up the DomainConfig for a domain string.

    Raises KeyError with a helpful message if the domain is unknown.
    """
    if domain not in DOMAIN_REGISTRY:
        available = ", ".join(f'"{k}"' for k in DOMAIN_REGISTRY)
        raise KeyError(
            f"Unknown domain {domain!r}. Available domains: {available}"
        )
    return DOMAIN_REGISTRY[domain]
