"""Sensitivity analysis: VTS threshold (tau) and dedup threshold (delta)."""

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
from src.prompt_evolution import build_prompt, get_strategy_name, get_phase
from src.schemas import Checkpoint, ExperimentConfig, StoredIdea
from src.vts import filter_by_probability

load_dotenv()

EXPERIMENT_NAME = "sensitivity_thresholds"


def load_config(domain_override: str | None = None) -> ExperimentConfig:
    with open("config.yaml") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig(
        domain=domain_override or raw["experiments"]["domain"],
        total_batches=raw["experiments"]["sensitivity"]["total_batches"],
        batch_size=raw["generator"]["batch_size"],
        probability_threshold=raw["vts"]["probability_threshold"],
        similarity_threshold=raw["memory"]["similarity_threshold"],
        recent_ideas_in_prompt=raw["memory"]["recent_ideas_in_prompt"],
        near_duplicates_shown=raw["memory"]["near_duplicates_shown"],
        saturation_multiplier=raw["memory"]["saturation_multiplier"],
        checkpoint_interval=raw["experiments"]["checkpoint_interval"],
        generator_model=raw["generator"]["model"],
        embedding_model=raw["embeddings"]["model"],
        method="dce",
        output_dir=raw["experiments"]["output_dir"],
        chroma_db_path=raw["memory"]["chroma_db_path"],
    )


def _domain_suffix(domain: str) -> str:
    """Short directory-safe suffix from a domain string."""
    return domain.replace(" ", "_")[:40]


def run_threshold(
    param: str,
    value: float,
    config: ExperimentConfig,
    domain_suffix: str = "",
):
    """Run DCE with a specific tau or delta override."""
    # Apply the override
    if param == "tau":
        config.probability_threshold = value
    elif param == "delta":
        config.similarity_threshold = value

    value_label = f"{param}_{value:.2f}"
    exp_dir = f"{EXPERIMENT_NAME}_{domain_suffix}" if domain_suffix else EXPERIMENT_NAME
    output_dir = Path(config.output_dir) / exp_dir / value_label
    output_dir.mkdir(parents=True, exist_ok=True)

    cp_path = output_dir / "checkpoint.json"
    start_batch = 0
    session_id = str(uuid.uuid4())[:8]
    if cp_path.exists():
        checkpoint = Checkpoint.model_validate_json(cp_path.read_text())
        start_batch = checkpoint.last_completed_batch + 1
        session_id = checkpoint.session_id
        if start_batch >= config.total_batches:
            print(f"  {value_label}: already complete, skipping.")
            return
        print(f"  {value_label}: resuming from batch {start_batch}")

    generator = IdeaGenerator(model=config.generator_model)
    embedding_client = EmbeddingClient(model=config.embedding_model)
    memory = SemanticMemory(
        db_path=str(output_dir / "chroma_db"),
        similarity_threshold=config.similarity_threshold,
        embedding_client=embedding_client,
    )

    results_path = output_dir / "results.jsonl"

    for batch_num in tqdm(range(start_batch, config.total_batches), desc=f"  {value_label}"):
        prompt = build_prompt(batch_num, memory, config, method="dce")
        batch_output = generator.generate(prompt)
        ideas = filter_by_probability(batch_output.ideas, config.probability_threshold)
        ideas = memory.check_duplicates(ideas)

        accepted = []
        for idea in ideas:
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

        memory.add(accepted)

        result = {
            "batch_number": batch_num,
            "ideas": [s.model_dump(exclude={"embedding"}) for s in accepted],
            "accepted_count": len(accepted),
            "param": param,
            "value": value,
            "strategy": get_strategy_name(batch_num),
            "phase": get_phase(batch_num, config.total_batches),
        }
        with jsonlines.open(results_path, mode="a") as writer:
            writer.write(result)

        if (batch_num + 1) % config.checkpoint_interval == 0:
            (output_dir / "checkpoint.json").write_text(
                Checkpoint(
                    experiment_name=EXPERIMENT_NAME,
                    method=f"dce_{value_label}",
                    last_completed_batch=batch_num,
                    total_batches=config.total_batches,
                    session_id=session_id,
                    ideas_accepted=memory.count,
                ).model_dump_json(indent=2)
            )

    # Final checkpoint
    (output_dir / "checkpoint.json").write_text(
        Checkpoint(
            experiment_name=EXPERIMENT_NAME,
            method=f"dce_{value_label}",
            last_completed_batch=config.total_batches - 1,
            total_batches=config.total_batches,
            session_id=session_id,
            ideas_accepted=memory.count,
        ).model_dump_json(indent=2)
    )

    # Save token usage
    with open(output_dir / "token_usage.json", "w") as f:
        json.dump(generator.token_usage, f, indent=2)

    print(f"  {value_label}: complete. {memory.count} ideas stored.")


def run(
    param: str,
    values: list[float],
    domain_override: str | None = None,
):
    config = load_config(domain_override)
    suffix = _domain_suffix(config.domain)

    print(f"Threshold sensitivity ({param}): {len(values)} values x {config.total_batches} batches")
    for value in values:
        print(f"\nRunning {param}={value:.2f}")
        # Reload config for each value so previous overrides don't persist
        fresh_config = load_config(domain_override)
        run_threshold(param, value, fresh_config, domain_suffix=suffix)
    print(f"\nThreshold sensitivity ({param}) complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sensitivity analysis: VTS threshold (tau) and dedup threshold (delta)"
    )
    parser.add_argument(
        "--param",
        type=str,
        required=True,
        choices=["tau", "delta"],
        help="Parameter to sweep: tau (VTS threshold) or delta (dedup threshold)",
    )
    parser.add_argument(
        "--values",
        type=float,
        nargs="+",
        required=True,
        help="Values to test (e.g. 0.05 0.10 0.20)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Override domain from config.yaml",
    )
    args = parser.parse_args()
    run(param=args.param, values=args.values, domain_override=args.domain)
