"""Pydantic models for DCE pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GeneratedIdea(BaseModel):
    """A single idea produced by the generator."""

    name: str = Field(description="Short name for the idea")
    description: str = Field(description="One-paragraph description")
    category: str = Field(description="Domain category (e.g. Marine, Aerospace, Food)")
    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-assessed probability: how likely is this idea among all possible responses (0-1)",
    )


class BatchOutput(BaseModel):
    """Structured output from a single generation call."""

    ideas: list[GeneratedIdea] = Field(description="List of generated ideas")


class StoredIdea(BaseModel):
    """An idea stored in the memory bank with its embedding."""

    id: str
    name: str
    description: str
    category: str
    probability: float
    batch_number: int
    session_id: str
    embedding: list[float] | None = None


class BatchResult(BaseModel):
    """Result of processing a single batch through the full pipeline."""

    batch_number: int
    generated: list[GeneratedIdea]
    after_vts: list[GeneratedIdea]
    after_dedup: list[GeneratedIdea]
    accepted: list[StoredIdea]
    strategy_used: str
    phase: str  # "exploration" or "exploitation"


class Checkpoint(BaseModel):
    """Experiment checkpoint for resumption."""

    experiment_name: str
    method: str
    last_completed_batch: int
    total_batches: int
    session_id: str
    ideas_accepted: int


class ExperimentConfig(BaseModel):
    """Configuration for a single experiment run."""

    domain: str = "sustainable packaging concepts"
    total_batches: int = 200
    batch_size: int = 5
    probability_threshold: float = 0.10
    similarity_threshold: float = 0.85
    recent_ideas_in_prompt: int = 10
    near_duplicates_shown: int = 5
    saturation_multiplier: float = 1.5
    phase_threshold: float = 0.40
    checkpoint_interval: int = 10
    generator_model: str = "gpt-5-mini-2025-08-07"
    embedding_model: str = "text-embedding-3-small"
    method: str = "dce"  # naive, vts_only, vts_dedup, dce
    output_dir: str = "data/raw"
    chroma_db_path: str = "data/chroma_db"
    session_id: str = ""
    cross_session: bool = True
