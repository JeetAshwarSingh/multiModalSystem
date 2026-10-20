#!/usr/bin/env python3
"""
run_fusion.py - Strategy 07: Bayesian Prior-Adjusted & Cost-Sensitive Late Fusion
Applies log-prior discounting and cost-sensitive reweighting to rescue suppressed minority engagement classes.
"""

import sys
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

STRATEGY_SLUG = "07_class_prior_cost_sensitive_fusion"
STRATEGY_NAME = "Bayesian Prior-Adjusted & Cost-Sensitive Late Fusion"

# DAiSEE test class priors: [4, 84, 882, 814] / 1784
DAISEE_PRIORS = [4 / 1784.0, 84 / 1784.0, 882 / 1784.0, 814 / 1784.0]

BASE_WEIGHTS = {
    "macro": 0.15,
    "rppg": 0.55,
    "mer": 0.30
}

def fuse_prior_adjusted(samples, tau=0.08, weights=BASE_WEIGHTS, priors=DAISEE_PRIORS):
    wm = weights["macro"]
    wr = weights["rppg"]
    wmer = weights["mer"]
    total_w = wm + wr + wmer

    predictions = []
    probabilities = []

    for s in samples:
        # Base probability combination
        raw_fused = [
            (wm * s["macro"][c] + wr * s["rppg"][c] + wmer * s["mer"][c]) / total_w
            for c in range(4)
        ]

        # Prior discounting: P(c) / (prior_c ^ tau)
        eps = 1e-6
        adjusted = [
            raw_fused[c] / ((priors[c] ** tau) + eps)
            for c in range(4)
        ]

        # Re-normalize to valid probability distribution
        total_adj = sum(adjusted)
        norm_adj = [p / total_adj for p in adjusted] if total_adj > 0 else [0.25] * 4
        pred = norm_adj.index(max(norm_adj))

        predictions.append(pred)
        probabilities.append(norm_adj)

    return predictions, probabilities

def main():
    parser = argparse.ArgumentParser(description=f"Run {STRATEGY_NAME}")
    parser.add_argument("--tau", type=float, default=0.08, help="Prior adjustment exponent (default: 0.08)")
    args = parser.parse_args()

    display_name = f"{STRATEGY_NAME} (tau={args.tau:.2f})"
    print(f"[*] Running {display_name}...")
    samples = load_modality_data()
    print(f"[*] Loaded {len(samples)} clips.")

    predictions, probabilities = fuse_prior_adjusted(samples, tau=args.tau)
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
