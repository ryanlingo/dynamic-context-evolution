"""Downstream validation: DeBERTa classifier trained on synthetic data."""

from __future__ import annotations

import json
from pathlib import Path

import jsonlines
import numpy as np
import yaml
from dotenv import load_dotenv

from src.domain_config import get_domain_config

load_dotenv()

EXPERIMENT_NAME = "downstream"


def _domain_suffix(domain: str) -> str:
    """Short directory-safe suffix from a domain string."""
    return domain.replace(" ", "_")[:40]


def load_training_data(method: str, max_ideas: int = 5000, data_dir: str = "data/raw/exp2_comparison") -> list[dict]:
    """Load generated ideas from an experiment run."""
    results_path = Path(data_dir) / method / "results.jsonl"
    ideas = []
    with jsonlines.open(results_path) as reader:
        for batch in reader:
            for idea in batch["ideas"]:
                ideas.append(idea)
                if len(ideas) >= max_ideas:
                    return ideas
    return ideas


def coarse_category(cat: str, separators: list[str] | None = None) -> str:
    """Extract a coarse top-level category from fine-grained labels."""
    import unicodedata
    if separators is None:
        separators = ["/", "&"]
    # Normalize unicode dashes/hyphens
    cat = unicodedata.normalize("NFKC", cat)
    # Take text before first separator as the coarse bucket
    for sep in separators:
        if sep in cat:
            cat = cat.split(sep)[0]
            break
    return cat.strip().lower()


def prepare_dataset(ideas: list[dict], min_count: int = 10, separators: list[str] | None = None, max_categories: int | None = None) -> tuple[list[str], list[str]]:
    """Convert ideas to (text, label) pairs for classification.

    Groups fine-grained categories into coarse buckets. Categories with
    fewer than min_count examples are dropped to ensure the classifier
    has enough signal. If max_categories is set, only the top-K most
    frequent categories are kept (controlling for class count across methods).
    """
    from collections import Counter

    texts = [f"{idea['name']}: {idea['description']}" for idea in ideas]
    labels = [coarse_category(idea["category"], separators=separators) for idea in ideas]

    # Only keep categories with enough examples
    counts = Counter(labels)
    filtered = [(t, l) for t, l in zip(texts, labels) if counts[l] >= min_count]
    if not filtered:
        raise ValueError("No categories with enough examples")

    # If max_categories is set, keep only the top-K most frequent
    if max_categories is not None:
        filtered_counts = Counter(l for _, l in filtered)
        top_cats = {cat for cat, _ in filtered_counts.most_common(max_categories)}
        filtered = [(t, l) for t, l in filtered if l in top_cats]

    texts, labels = zip(*filtered)
    return list(texts), list(labels)


def run(
    domain_override: str | None = None,
    data_dir_override: str | None = None,
    methods_override: list[str] | None = None,
    max_categories: int | None = None,
):
    try:
        import evaluate
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError:
        print("Downstream experiment requires: pip install dce[downstream]")
        return

    with open("config.yaml") as f:
        raw = yaml.safe_load(f)
    ds_config = raw["experiments"]["downstream"]
    domain = domain_override or raw["experiments"]["domain"]
    domain_cfg = get_domain_config(domain)
    suffix = _domain_suffix(domain)

    data_dir = Path(data_dir_override) if data_dir_override else Path("data/raw") / f"exp2_comparison_{suffix}"
    output_dir = Path("data/processed") / f"{EXPERIMENT_NAME}_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = methods_override or ["naive", "vts_dedup", "dce"]
    results = {}

    for method in methods:
        print(f"\nTraining on {method} data...")

        ideas = load_training_data(method, ds_config["training_size"], data_dir=str(data_dir))
        texts, labels = prepare_dataset(ideas, separators=domain_cfg.category_separators, max_categories=max_categories)

        # Create label mapping
        unique_labels = sorted(set(labels))
        label2id = {l: i for i, l in enumerate(unique_labels)}
        id2label = {i: l for l, i in label2id.items()}
        numeric_labels = [label2id[l] for l in labels]

        # Stratified 80/20 split
        from sklearn.model_selection import train_test_split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, numeric_labels, test_size=0.2, stratify=numeric_labels, random_state=42
        )

        tokenizer = AutoTokenizer.from_pretrained(ds_config["model"])

        def tokenize(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

        train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(tokenize, batched=True)
        val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels}).map(tokenize, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            ds_config["model"],
            num_labels=len(unique_labels),
            label2id=label2id,
            id2label=id2label,
        )

        f1_metric = evaluate.load("f1")

        def compute_metrics(eval_pred):
            logits, label_ids = eval_pred
            preds = np.argmax(logits, axis=-1)
            return f1_metric.compute(predictions=preds, references=label_ids, average="weighted")

        training_args = TrainingArguments(
            output_dir=str(output_dir / method),
            num_train_epochs=ds_config["epochs"],
            per_device_train_batch_size=ds_config["batch_size"],
            per_device_eval_batch_size=ds_config["batch_size"],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=50,
            report_to="none",
            learning_rate=ds_config["learning_rate"],
            use_cpu=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_result = trainer.evaluate()
        results[method] = eval_result
        print(f"  {method}: F1 = {eval_result.get('eval_f1', 'N/A')}")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDownstream results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Downstream validation: DeBERTa classifier")
    parser.add_argument("--domain", type=str, default=None, help="Override domain from config.yaml")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory (e.g. data/raw/sensitivity_thresholds_xxx/delta_0.90)")
    parser.add_argument("--methods", type=str, nargs="*", default=None, help="Methods to evaluate (default: naive vts_dedup dce)")
    parser.add_argument("--max-categories", type=int, default=None, help="Limit to top-K categories (controls for class count)")
    args = parser.parse_args()
    run(domain_override=args.domain, data_dir_override=args.data_dir, methods_override=args.methods, max_categories=args.max_categories)
