# Dynamic Context Evolution for Scalable Synthetic Data Generation

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

Large language models produce repetitive output when prompted independently across many batches — a phenomenon we term **cross-batch mode collapse**. We introduce **Dynamic Context Evolution (DCE)**, comprising three mechanisms: (1) *verbalized tail sampling*, which filters high-probability candidates via model self-assessment; (2) *semantic memory*, which maintains a persistent embedding index to reject near-duplicates across batches; and (3) *adaptive prompt evolution*, which reconstructs the generation prompt each batch using memory state and rotating diversity strategies. DCE achieves 0.0% collapse versus 5.6% for naive prompting across three domains and two model families, at ~$0.50 per 1,000 candidates using only standard API calls.

## Architecture

```
┌─────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│   Generator  │────▶│  Verbalized Tail    │────▶│  Semantic Memory  │
│  (LLM API)  │     │  Sampling (VTS)     │     │  (ChromaDB)       │
└─────────────┘     └─────────────────────┘     └────────┬─────────┘
       ▲                                                  │
       │            ┌─────────────────────┐               │
       └────────────│  Adaptive Prompt    │◀──────────────┘
                    │  Evolution          │
                    └─────────────────────┘
```

Each batch: the generator produces candidates → VTS filters obvious ideas (self-assessed probability > 0.10) → semantic memory rejects near-duplicates (cosine similarity > 0.85) → prompt evolution rewrites the next prompt using memory state and a rotating diversity strategy.

## Installation

```bash
git clone https://github.com/ryanlingo/dynamic-context-evolution.git
cd dynamic-context-evolution
pip install -e .

# Or with uv:
uv sync
```

For downstream evaluation (DeBERTa classifier):
```bash
pip install -e ".[downstream]"
```

## Quick Start

1. Copy the environment template and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your OPENAI_API_KEY and ANTHROPIC_API_KEY
   ```

2. Run a DCE generation session:
   ```bash
   python experiments/run_exp2_comparison.py
   ```

3. Configuration is in `config.yaml` — adjust domain, batch count, thresholds, etc.

## Reproducing Paper Experiments

**Experiment 1 — Cross-batch mode collapse:**
```bash
python experiments/run_exp1_collapse.py
```

**Experiment 2 — DCE vs. baselines (multi-seed):**
```bash
python experiments/run_multi_seed.py
```

**Sensitivity analysis:**
```bash
python experiments/run_sensitivity.py
python experiments/run_sensitivity_thresholds.py
```

**Downstream evaluation:**
```bash
python experiments/run_downstream.py
```

Analysis scripts in `analysis/` generate all paper figures and tables.

## Data

Experiment data (raw generation logs and processed embeddings) is available on the [GitHub Releases](https://github.com/ryanlingo/dynamic-context-evolution/releases/tag/v1.0) page.

## Citation

```bibtex
@article{lingo2026dynamic,
  title={Dynamic Context Evolution for Scalable Synthetic Data Generation},
  author={Lingo, Ryan and Chhajer, Rajeev},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

Copyright 2026 Honda Research Institute, USA, Inc.
