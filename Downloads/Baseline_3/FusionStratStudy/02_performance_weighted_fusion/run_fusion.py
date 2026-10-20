#!/usr/bin/env python3
"""
run_fusion.py - Strategy 02: Empirical Performance-Weighted Soft Voting
Weighted convex combination of prediction probabilities aligned with modality discriminative power.
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

STRATEGY_SLUG = "02_performance_weighted_fusion"
STRATEGY_NAME = "Empirical Performance-Weighted Fusion (Macro:0.15, rPPG:0.55, MER:0.30)"

DEFAULT_WEIGHTS = {
    "macro": 0.15,
    "rppg": 0.55,
    "mer": 0.30
}

def fuse_weighted(samples, weights):
    wm = weights["macro"]
    wr = weights["rppg"]
    wmer = weights["mer"]
    total = wm + wr + wmer

    predictions = []
    probabilities = []

    for s in samples:
        pm = s["macro"]
        pr = s["rppg"]
        pmer = s["mer"]

        fused = [(wm * pm[c] + wr * pr[c] + wmer * pmer[c]) / total for c in range(4)]
        pred = fused.index(max(fused))

        predictions.append(pred)
        probabilities.append(fused)

    return predictions, probabilities

def run_grid_search(samples):
    """Grid search over weight simplex to identify optimal Macro-F1 configuration."""
    print("[*] Running grid search over weight simplex (step=0.05)...")
    trues = [s["true_y"] for s in samples]
    best_f1 = -1.0
    best_w = None
    best_metrics = None

    steps = [round(i * 0.05, 2) for i in range(21)]
    for wm in steps:
        for wr in steps:
            wmer = round(1.0 - wm - wr, 2)
            if wmer < 0.0:
                continue
            w_dict = {"macro": wm, "rppg": wr, "mer": wmer}
            preds, probs = fuse_weighted(samples, w_dict)
            met = compute_metrics(preds, trues, probs)
            if met["macro_f1"] > best_f1:
                best_f1 = met["macro_f1"]
                best_w = w_dict
                best_metrics = met

    print(f"[*] Optimal Grid Search Weights: Macro={best_w['macro']:.2f}, rPPG={best_w['rppg']:.2f}, MER={best_w['mer']:.2f}")
    print(f"[*] Optimal Macro-F1: {best_f1:.4f} | Accuracy: {best_metrics['accuracy']*100:.2f}% | Balanced Accuracy: {best_metrics['balanced_accuracy']*100:.2f}%")
    return best_w

def main():
    parser = argparse.ArgumentParser(description=f"Run {STRATEGY_NAME}")
    parser.add_argument("--grid-search", action="store_true", help="Perform grid search over simplex before evaluation")
    parser.add_argument("--w-macro", type=float, default=DEFAULT_WEIGHTS["macro"], help="Weight for Macro modality")
    parser.add_argument("--w-rppg", type=float, default=DEFAULT_WEIGHTS["rppg"], help="Weight for rPPG modality")
    parser.add_argument("--w-mer", type=float, default=DEFAULT_WEIGHTS["mer"], help="Weight for MER modality")
    args = parser.parse_args()

    samples = load_modality_data()
    print(f"[*] Loaded {len(samples)} clips.")

    if args.grid_search:
        weights = run_grid_search(samples)
    else:
        weights = {"macro": args.w_macro, "rppg": args.w_rppg, "mer": args.w_mer}

    display_name = f"Performance-Weighted Fusion (Macro:{weights['macro']:.2f}, rPPG:{weights['rppg']:.2f}, MER:{weights['mer']:.2f})"
    print(f"[*] Evaluating with weights: {weights}...")

    predictions, probabilities = fuse_weighted(samples, weights)
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
