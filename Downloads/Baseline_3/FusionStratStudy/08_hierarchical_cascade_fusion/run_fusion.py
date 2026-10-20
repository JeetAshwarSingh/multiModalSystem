#!/usr/bin/env python3
"""
run_fusion.py - Strategy 08: Hierarchical Coarse-to-Fine Engagement Cascade Fusion
Decouples macroscopic engagement state (Low vs High) from fine-grained ordinal levels.
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

STRATEGY_SLUG = "08_hierarchical_cascade_fusion"
STRATEGY_NAME = "Hierarchical Coarse-to-Fine Cascade Fusion"

BASE_WEIGHTS = {
    "macro": 0.15,
    "rppg": 0.50,
    "mer": 0.35
}

def fuse_hierarchical(samples, theta_coarse=0.50, theta_fine=0.52, theta_minority=0.80, weights=BASE_WEIGHTS):
    wm = weights["macro"]
    wr = weights["rppg"]
    wmer = weights["mer"]
    total_w = wm + wr + wmer

    predictions = []
    probabilities = []

    for s in samples:
        # Step 1: Base weighted probability fusion
        fused = [
            (wm * s["macro"][c] + wr * s["rppg"][c] + wmer * s["mer"][c]) / total_w
            for c in range(4)
        ]

        # Step 2: Hierarchical Stage 1 - Coarse Engagement State
        p_low = fused[0] + fused[1]
        p_high = fused[2] + fused[3]

        if p_high >= theta_coarse:
            # Stage 2A: High Engagement - Decide between Class 2 and Class 3
            rel_high = fused[3] / (fused[2] + fused[3] + 1e-12)
            if rel_high >= theta_fine:
                pred = 3
            else:
                pred = 2
        else:
            # Stage 2B: Low Engagement - Decide between Class 0 and Class 1
            rel_low = fused[1] / (fused[0] + fused[1] + 1e-12)
            if rel_low >= (1.0 - theta_minority):
                pred = 1
            else:
                pred = 0

        predictions.append(pred)
        probabilities.append(fused)

    return predictions, probabilities

def main():
    parser = argparse.ArgumentParser(description=f"Run {STRATEGY_NAME}")
    parser.add_argument("--theta-coarse", type=float, default=0.50, help="Threshold for High Engagement (default: 0.50)")
    parser.add_argument("--theta-fine", type=float, default=0.52, help="Threshold for Class 3 vs Class 2 (default: 0.52)")
    parser.add_argument("--theta-minority", type=float, default=0.80, help="Threshold for Class 1 vs Class 0 (default: 0.80)")
    args = parser.parse_args()

    display_name = f"{STRATEGY_NAME} (coarse={args.theta_coarse:.2f}, fine={args.theta_fine:.2f})"
    print(f"[*] Running {display_name}...")
    samples = load_modality_data()
    print(f"[*] Loaded {len(samples)} clips.")

    predictions, probabilities = fuse_hierarchical(
        samples,
        theta_coarse=args.theta_coarse,
        theta_fine=args.theta_fine,
        theta_minority=args.theta_minority
    )
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
