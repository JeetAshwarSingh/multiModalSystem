# Evaluation README

This directory contains scripts and utilities for evaluating the full **Engagement‑MAS** pipeline.

## Main scripts
| Script | Purpose |
|--------|---------|
| `metrics.py` | Functions to compute accuracy, macro‑F1, per‑class precision/recall, and confusion matrices (using **scikit‑learn**). |
| `evaluate.py` | Runs the entire pipeline on a chosen dataset split, applies a selected fusion strategy (`early`, `static`, `agentic`), and prints a summary of metrics. It also supports ablation flags to drop modalities or switch off the reasoning engine. |
| `robustness.py` | Randomly masks modalities during inference to measure robustness degradation (reports accuracy drop per missing‑modality ratio). |

## How to run
```bash
# Activate the appropriate environment first
conda activate engagement-mas-mac   # or ...-cloud for GPU

# Basic evaluation using the early‑fusion baseline on the test split
python -m evaluation.evaluate \
    --config configs/fusion.yaml \
    --dataset daisee \
    --split test \
    --fusion early

# Agentic late‑fusion with rule‑based reasoning (default)
python -m evaluation.evaluate \
    --fusion agentic

# Ablation: remove the rPPG modality
python -m evaluation.evaluate \
    --fusion agentic --disable-modality rppg

# Robustness test (10 random masks, 30 % missing each time)
python -m evaluation.robustness \
    --fusion static --mask-ratio 0.3 --iterations 10
```

All scripts write a CSV summary to `evaluation/results/` and also print a short table to the console.

## Requirements
* `scikit-learn` (already in the Conda env)
* `numpy`, `pandas`
* The perception blocks (`blocks/*`) must be importable (i.e., the project root is on `PYTHONPATH`).

---

**Note:** The evaluation uses the same deterministic inference code as the blocks themselves, so repeated runs on the same data produce identical results (subject to floating‑point rounding).
