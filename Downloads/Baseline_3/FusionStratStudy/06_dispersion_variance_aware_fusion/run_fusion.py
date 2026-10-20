#!/usr/bin/env python3
"""
run_fusion.py - Strategy 06: Informational Dispersion & Modality Variance Gated Fusion
Measures dynamic range and signal variance across modalities to automatically suppress collapsed or frozen streams.
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

STRATEGY_SLUG = "06_dispersion_variance_aware_fusion"
STRATEGY_NAME = "Informational Dispersion & Variance-Aware Late Fusion"

def compute_modality_variances(samples):
    """Computes mean class probability variance for each modality across the dataset."""
    variances = {}
    N = len(samples)

    for mod in ["macro", "rppg", "mer"]:
        class_vars = []
        for c in range(4):
            vals = [s[mod][c] for s in samples]
            mean_v = sum(vals) / N
            var_v = sum((v - mean_v) ** 2 for v in vals) / N
            class_vars.append(var_v)
        variances[mod] = sum(class_vars) / 4.0

    return variances

def fuse_dispersion_aware(samples, gamma=0.5):
    variances = compute_modality_variances(samples)
    print(f"[*] Computed Informational Dispersions (Variances):")
    for mod, v in variances.items():
        print(f"    - {mod:<6}: {v:.8f}")

    # Compute weights proportional to variance^gamma
    eps = 1e-8
    raw_weights = {mod: (v + eps) ** gamma for mod, v in variances.items()}
    total_w = sum(raw_weights.values())
    norm_weights = {mod: w / total_w for mod, w in raw_weights.items()}

    print(f"[*] Auto-Derived Dispersion Weights (gamma={gamma:.2f}):")
    for mod, w in norm_weights.items():
        print(f"    - {mod:<6}: {w:.4f} ({w*100:.1f}%)")

    wm = norm_weights["macro"]
    wr = norm_weights["rppg"]
    wmer = norm_weights["mer"]

    predictions = []
    probabilities = []

    for s in samples:
        fused = [
            wm * s["macro"][c] + wr * s["rppg"][c] + wmer * s["mer"][c]
            for c in range(4)
        ]
        pred = fused.index(max(fused))

        predictions.append(pred)
        probabilities.append(fused)

    return predictions, probabilities, norm_weights

def main():
    parser = argparse.ArgumentParser(description=f"Run {STRATEGY_NAME}")
    parser.add_argument("--gamma", type=float, default=0.5, help="Dispersion sensitivity exponent (default: 0.5)")
    args = parser.parse_args()

    display_name = f"{STRATEGY_NAME} (gamma={args.gamma:.2f})"
    print(f"[*] Running {display_name}...")
    samples = load_modality_data()
    print(f"[*] Loaded {len(samples)} clips.")

    predictions, probabilities, weights = fuse_dispersion_aware(samples, gamma=args.gamma)
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
