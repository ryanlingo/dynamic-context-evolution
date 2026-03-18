"""Experiment 2: DCE vs 3 baselines.

Methods: naive, vts_only, vts_dedup, dce
Each runs 200 batches of 5 ideas.
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
from src.prompt_evolution import build_prompt, get_strategy_name, get_phase
from src.schemas import Checkpoint, ExperimentConfig, StoredIdea
from src.vts import filter_by_probability

load_dotenv()

EXPERIMENT_NAME = "exp2_comparison"


def load_config(domain_override: str | None = None) -> ExperimentConfig:
    with open("config.yaml") as f:
        raw = yaml.safe_load(f)
    exp = raw["experiments"]["exp2_comparison"]
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
        output_dir=raw["experiments"]["output_dir"],
        chroma_db_path=raw["memory"]["chroma_db_path"],
    )


def _domain_suffix(domain: str) -> str:
    """Short directory-safe suffix from a domain string."""
    return domain.replace(" ", "_")[:40]


def run_method(method: str, config: ExperimentConfig, domain_suffix: str = ""):
    """Run a single method for the full experiment."""
    config.method = method
    exp_dir = f"{EXPERIMENT_NAME}_{domain_suffix}" if domain_suffix else EXPERIMENT_NAME
    output_dir = Path(config.output_dir) / exp_dir / method
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoint
    cp_path = output_dir / "checkpoint.json"
    start_batch = 0
    session_id = str(uuid.uuid4())[:8]
    if cp_path.exists():
        checkpoint = Checkpoint.model_validate_json(cp_path.read_text())
        start_batch = checkpoint.last_completed_batch + 1
        session_id = checkpoint.session_id
        if start_batch >= config.total_batches:
            print(f"  {method}: already complete, skipping.")
            return
        print(f"  {method}: resuming from batch {start_batch}")

    # Determine generator parameters based on method
    gen_kwargs: dict = dict(model=config.generator_model)
    if method == "temp_1.2_dedup":
        gen_kwargs["temperature"] = 1.2
    elif method == "nucleus_0.9_dedup":
        gen_kwargs["top_p"] = 0.9

    generator = IdeaGenerator(**gen_kwargs)
    embedding_client = EmbeddingClient(model=config.embedding_model)
    memory = SemanticMemory(
        db_path=str(output_dir / "chroma_db"),
        similarity_threshold=config.similarity_threshold,
        embedding_client=embedding_client,
    )

    results_path = output_dir / "results.jsonl"

    # Determine which pipeline components are active for this method
    use_vts = method in ("vts_only", "vts_dedup", "dce")
    use_dedup = method in ("vts_dedup", "dce", "dedup_only", "prompt_evo_dedup",
                           "temp_1.2_dedup", "nucleus_0.9_dedup")
    use_prompt_evo = method in ("dce", "prompt_evo_only", "prompt_evo_dedup")

    # For prompt building, map ablation methods to their closest base method
    prompt_method = method
    if method in ("dedup_only", "temp_1.2_dedup", "nucleus_0.9_dedup"):
        prompt_method = "naive"  # no VTS instruction, no strategy
    elif method in ("prompt_evo_only", "prompt_evo_dedup"):
        prompt_method = "dce"  # full prompt evolution

    for batch_num in tqdm(
        range(start_batch, config.total_batches), desc=f"  {method}"
    ):
        prompt = build_prompt(batch_num, memory, config, method=prompt_method)
        batch_output = generator.generate(prompt)
        ideas = batch_output.ideas

        # VTS filter
        if use_vts:
            ideas = filter_by_probability(ideas, config.probability_threshold)

        # Dedup filter
        if use_dedup:
            ideas = memory.check_duplicates(ideas)

        # Store accepted ideas
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

        strategy = get_strategy_name(batch_num) if use_prompt_evo else "none"
        phase = get_phase(batch_num, config.total_batches) if use_prompt_evo else "none"

        result = {
            "batch_number": batch_num,
            "ideas": [s.model_dump(exclude={"embedding"}) for s in accepted],
            "generated_count": len(batch_output.ideas),
            "after_vts_count": len(ideas) if method in ("vts_only", "vts_dedup", "dce") else len(batch_output.ideas),
            "accepted_count": len(accepted),
            "method": method,
            "strategy": strategy,
            "phase": phase,
        }
        with jsonlines.open(results_path, mode="a") as writer:
            writer.write(result)

        if (batch_num + 1) % config.checkpoint_interval == 0:
            Checkpoint(
                experiment_name=EXPERIMENT_NAME,
                method=method,
                last_completed_batch=batch_num,
                total_batches=config.total_batches,
                session_id=session_id,
                ideas_accepted=memory.count,
            ).model_dump_json(indent=2)
            (output_dir / "checkpoint.json").write_text(
                Checkpoint(
                    experiment_name=EXPERIMENT_NAME,
                    method=method,
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
            method=method,
            last_completed_batch=config.total_batches - 1,
            total_batches=config.total_batches,
            session_id=session_id,
            ideas_accepted=memory.count,
        ).model_dump_json(indent=2)
    )

    # Save token usage for cost analysis
    import json as _json
    with open(output_dir / "token_usage.json", "w") as f:
        _json.dump(generator.token_usage, f, indent=2)

    print(f"  {method}: complete. {memory.count} ideas stored.")


def run(methods: list[str] | None = None, domain_override: str | None = None, model_override: str | None = None):
    config = load_config(domain_override)
    if model_override:
        config.generator_model = model_override
    suffix = _domain_suffix(config.domain)
    # When using a non-default model, add model name to output dir
    if model_override:
        model_tag = model_override.replace("/", "_").replace(".", "_")[:30]
        suffix = f"{suffix}_{model_tag}"
    all_methods = [
        "naive", "vts_only", "vts_dedup", "dce",
        # Ablation methods (Step 5)
        "dedup_only", "prompt_evo_only", "prompt_evo_dedup",
        # Token-level baselines (Step 5)
        "temp_1.2_dedup", "nucleus_0.9_dedup",
    ]
    methods = methods or all_methods
    for m in methods:
        if m not in all_methods:
            print(f"Unknown method: {m}. Choose from {all_methods}")
            return
    print(f"Experiment 2: Running {len(methods)} methods × {config.total_batches} batches")
    print(f"  Model: {config.generator_model}")
    for method in methods:
        print(f"\nStarting method: {method}")
        run_method(method, config, domain_suffix=suffix)
    print("\nExperiment 2 complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 2: DCE vs baselines")
    parser.add_argument("--domain", type=str, default=None, help="Override domain from config.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override generator model (e.g. claude-haiku-4-5)")
    parser.add_argument("methods", nargs="*", default=None, help="Methods to run (default: all)")
    args = parser.parse_args()
    run(methods=args.methods or None, domain_override=args.domain, model_override=args.model)
