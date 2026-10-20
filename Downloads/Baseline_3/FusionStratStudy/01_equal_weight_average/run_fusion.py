#!/usr/bin/env python3
"""
run_fusion.py - Strategy 01: Equal-Weight Late Fusion
Arithmetic mean of prediction probabilities across all available modalities.
"""

import sys
from pathlib import Path

# Ensure FusionStratStudy is importable
current_dir = Path(__file__).resolve().parent
study_dir = current_dir.parent
if str(study_dir) not in sys.path:
    sys.path.insert(0, str(study_dir))

from common import (
    load_modality_data,
    compute_metrics,
    print_evaluation_report,
    save_results
)

STRATEGY_SLUG = "01_equal_weight_average"
STRATEGY_NAME = "Equal-Weight Late Fusion (1/3, 1/3, 1/3)"

def fuse_equal_weight(samples):
    predictions = []
    probabilities = []

    for s in samples:
        pm = s["macro"]
        pr = s["rppg"]
        pmer = s["mer"]

        # Simple arithmetic average across all 4 classes
        fused = [(pm[c] + pr[c] + pmer[c]) / 3.0 for c in range(4)]
        pred = fused.index(max(fused))

        predictions.append(pred)
        probabilities.append(fused)

    return predictions, probabilities

def main():
    print(f"[*] Running {STRATEGY_NAME}...")
    samples = load_modality_data()
    print(f"[*] Loaded {len(samples)} clips.")

    predictions, probabilities = fuse_equal_weight(samples)
    trues = [s["true_y"] for s in samples]

    metrics = compute_metrics(predictions, trues, probabilities)
    print_evaluation_report(STRATEGY_NAME, metrics)

    save_results(
        strategy_slug=STRATEGY_SLUG,
        strategy_display_name=STRATEGY_NAME,
        samples=samples,
        predictions=predictions,
        probabilities=probabilities,
        metrics=metrics
    )

if __name__ == "__main__":
    main()
