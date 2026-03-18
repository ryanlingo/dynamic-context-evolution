"""Experiment 1: Demonstrate cross-batch mode collapse with naive prompting.

Generates 200 batches of 5 ideas using the same prompt every batch, no memory.
Checkpoints every 10 batches.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import jsonlines
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from src.embeddings import EmbeddingClient
from src.generator import IdeaGenerator
from src.memory import SemanticMemory
from src.prompt_evolution import build_prompt
from src.schemas import Checkpoint, ExperimentConfig, StoredIdea

load_dotenv()

EXPERIMENT_NAME = "exp1_collapse"


def load_config(domain_override: str | None = None) -> ExperimentConfig:
    with open("config.yaml") as f:
        raw = yaml.safe_load(f)
    exp = raw["experiments"]["exp1_collapse"]
    return ExperimentConfig(
        domain=domain_override or raw["experiments"]["domain"],
        total_batches=exp["total_batches"],
        batch_size=raw["generator"]["batch_size"],
        probability_threshold=raw["vts"]["probability_threshold"],
        similarity_threshold=raw["memory"]["similarity_threshold"],
        recent_ideas_in_prompt=raw["memory"]["recent_ideas_in_prompt"],
        near_duplicates_shown=raw["memory"]["near_duplicates_shown"],
        saturation_multiplier=raw["memory"]["saturation_multiplier"],
        phase_threshold=raw["prompt_evolution"]["phase_threshold"],
        checkpoint_interval=raw["experiments"]["checkpoint_interval"],
        generator_model=raw["generator"]["model"],
        embedding_model=raw["embeddings"]["model"],
        method="naive",
        output_dir=raw["experiments"]["output_dir"],
        chroma_db_path=raw["memory"]["chroma_db_path"],
    )


def load_checkpoint(output_dir: Path) -> Checkpoint | None:
    cp_path = output_dir / "checkpoint.json"
    if cp_path.exists():
        return Checkpoint.model_validate_json(cp_path.read_text())
    return None


def save_checkpoint(checkpoint: Checkpoint, output_dir: Path) -> None:
    (output_dir / "checkpoint.json").write_text(checkpoint.model_dump_json(indent=2))


def _domain_suffix(domain: str) -> str:
    """Short directory-safe suffix from a domain string."""
    return domain.replace(" ", "_")[:40]


def run(domain_override: str | None = None):
    config = load_config(domain_override)
    suffix = _domain_suffix(config.domain)
    output_dir = Path(config.output_dir) / f"{EXPERIMENT_NAME}_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    session_id = str(uuid.uuid4())[:8]
    checkpoint = load_checkpoint(output_dir)
    start_batch = 0
    if checkpoint:
        start_batch = checkpoint.last_completed_batch + 1
        session_id = checkpoint.session_id
        print(f"Resuming from batch {start_batch} (session {session_id})")

    generator = IdeaGenerator(model=config.generator_model)
    embedding_client = EmbeddingClient(model=config.embedding_model)

    # For naive method: we still store embeddings for metrics, but don't use memory for dedup
    memory = SemanticMemory(
        db_path=str(output_dir / "chroma_db"),
        similarity_threshold=config.similarity_threshold,
        embedding_client=embedding_client,
    )

    results_path = output_dir / "results.jsonl"

    for batch_num in tqdm(range(start_batch, config.total_batches), desc="Exp1 batches"):
        # Naive: same prompt every time, no VTS, no memory dedup, no evolution
        prompt = build_prompt(batch_num, memory, config, method="naive")
        batch_output = generator.generate(prompt)

        # All ideas accepted (no filtering in naive mode)
        accepted = []
        for idea in batch_output.ideas:
            stored = StoredIdea(
                id=str(uuid.uuid4()),
                name=idea.name,
                description=idea.description,
                category=idea.category,
                probability=idea.probability,
                batch_number=batch_num,
                session_id=session_id,
            )
            accepted.append(stored)

        # Store in memory (for metrics tracking, not for dedup)
        memory.add(accepted)

        # Write results
        result = {
            "batch_number": batch_num,
            "ideas": [s.model_dump(exclude={"embedding"}) for s in accepted],
            "method": "naive",
        }
        with jsonlines.open(results_path, mode="a") as writer:
            writer.write(result)

        # Checkpoint
        if (batch_num + 1) % config.checkpoint_interval == 0:
            save_checkpoint(
                Checkpoint(
                    experiment_name=EXPERIMENT_NAME,
                    method="naive",
                    last_completed_batch=batch_num,
                    total_batches=config.total_batches,
                    session_id=session_id,
                    ideas_accepted=memory.count,
                ),
                output_dir,
            )

    # Final checkpoint
    save_checkpoint(
        Checkpoint(
            experiment_name=EXPERIMENT_NAME,
            method="naive",
            last_completed_batch=config.total_batches - 1,
            total_batches=config.total_batches,
            session_id=session_id,
            ideas_accepted=memory.count,
        ),
        output_dir,
    )
    print(f"Experiment 1 complete. {memory.count} ideas stored.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 1: mode collapse demo")
    parser.add_argument("--domain", type=str, default=None, help="Override domain from config.yaml")
    args = parser.parse_args()
    run(domain_override=args.domain)
