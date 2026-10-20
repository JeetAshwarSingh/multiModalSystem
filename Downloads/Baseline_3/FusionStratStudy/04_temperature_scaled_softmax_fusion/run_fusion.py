#!/usr/bin/env python3
"""
run_fusion.py - Strategy 04: Calibrated Temperature-Scaled Late Fusion
Applies modality-specific temperature scaling to soften overconfident modalities and boost discriminative channels.
"""

import sys
import math
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

STRATEGY_SLUG = "04_temperature_scaled_softmax_fusion"
STRATEGY_NAME = "Calibrated Temperature-Scaled Late Fusion (T_macro:5.0, T_rppg:1.2, T_mer:1.2)"

TEMPERATURES = {
    "macro": 5.0,  # Flatten overconfident static prediction
    "rppg": 1.2,   # Smooth physiological confidence
    "mer": 1.2     # Smooth micro-expression confidence
}

WEIGHTS = {
    "macro": 0.20,
    "rppg": 0.45,
    "mer": 0.35
}

def apply_temperature_scaling(probs, temperature):
    """
    Transforms probability vector into logits, divides by temperature,
    and returns calibrated softmax probabilities.
    """
    eps = 1e-12
    # Invert softmax to pseudo-logits
    logits = [math.log(max(p, eps)) / temperature for p in probs]
    # Numerically stable softmax
    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]
    sum_exp = sum(exp_logits)
    return [e / sum_exp for e in exp_logits]

def fuse_temperature_scaled(samples, temps=TEMPERATURES, weights=WEIGHTS):
    wm = weights["macro"]
    wr = weights["rppg"]
    wmer = weights["mer"]
    total = wm + wr + wmer

    predictions = []
    probabilities = []

    for s in samples:
        calib_m = apply_temperature_scaling(s["macro"], temps["macro"])
        calib_r = apply_temperature_scaling(s["rppg"], temps["rppg"])
        calib_mer = apply_temperature_scaling(s["mer"], temps["mer"])

        fused = [
            (wm * calib_m[c] + wr * calib_r[c] + wmer * calib_mer[c]) / total
            for c in range(4)
        ]
        pred = fused.index(max(fused))

        predictions.append(pred)
        probabilities.append(fused)

    return predictions, probabilities

def main():
    print(f"[*] Running {STRATEGY_NAME}...")
    samples = load_modality_data()
    print(f"[*] Loaded {len(samples)} clips.")

    predictions, probabilities = fuse_temperature_scaled(samples)
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
