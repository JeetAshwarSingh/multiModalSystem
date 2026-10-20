#!/usr/bin/env python3
"""
run_fusion.py - Strategy 05: Shannon Entropy & Uncertainty-Gated Late Fusion
Dynamically weights modalities for each sample inversely proportional to prediction entropy.
"""

import sys
import math
import argparse
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

STRATEGY_SLUG = "05_entropy_uncertainty_gated_fusion"
STRATEGY_NAME = "Shannon Entropy & Uncertainty-Gated Late Fusion"

BASE_WEIGHTS = {
    "macro": 0.15,
    "rppg": 0.55,
    "mer": 0.30
}

def compute_entropy(probs):
    """Computes Shannon entropy in nats."""
    eps = 1e-12
    return -sum(p * math.log(max(p, eps)) for p in probs)

def fuse_entropy_gated(samples, alpha=1.0, base_weights=BASE_WEIGHTS):
    predictions = []
    probabilities = []

    for s in samples:
        em = compute_entropy(s["macro"])
        er = compute_entropy(s["rppg"])
        emer = compute_entropy(s["mer"])

        # Exponential decay with entropy
        wm = base_weights["macro"] * math.exp(-alpha * em)
        wr = base_weights["rppg"] * math.exp(-alpha * er)
        wmer = base_weights["mer"] * math.exp(-alpha * emer)
        total_w = wm + wr + wmer

        if total_w > 0:
            norm_wm = wm / total_w
            norm_wr = wr / total_w
            norm_wmer = wmer / total_w
        else:
            norm_wm = norm_wr = norm_wmer = 1.0 / 3.0

        fused = [
            norm_wm * s["macro"][c] + norm_wr * s["rppg"][c] + norm_wmer * s["mer"][c]
            for c in range(4)
        ]
        pred = fused.index(max(fused))

        predictions.append(pred)
        probabilities.append(fused)

    return predictions, probabilities

def main():
    parser = argparse.ArgumentParser(description=f"Run {STRATEGY_NAME}")
    parser.add_argument("--alpha", type=float, default=1.0, help="Uncertainty scaling factor (default: 1.0)")
    args = parser.parse_args()

    display_name = f"{STRATEGY_NAME} (alpha={args.alpha:.1f})"
    print(f"[*] Running {display_name}...")
    samples = load_modality_data()
    print(f"[*] Loaded {len(samples)} clips.")

    predictions, probabilities = fuse_entropy_gated(samples, alpha=args.alpha)
    trues = [s["true_y"] for s in samples]

    metrics = compute_metrics(predictions, trues, probabilities)
    print_evaluation_report(display_name, metrics)

    save_results(
        strategy_slug=STRATEGY_SLUG,
        strategy_display_name=display_name,
        samples=samples,
        predictions=predictions,
        probabilities=probabilities,
        metrics=metrics
    )

if __name__ == "__main__":
    main()
