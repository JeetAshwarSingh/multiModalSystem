#!/usr/bin/env python3
"""
run_fusion.py - Strategy 03: Non-Parametric Borda Count Rank Fusion
Converts probabilities to ordinal class rankings to eliminate calibration and scale distortions.
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

STRATEGY_SLUG = "03_borda_count_rank_fusion"
STRATEGY_NAME = "Borda Count Rank Fusion"

# Weighting for ranks: giving higher weight to informative modalities
WEIGHTS = {
    "macro": 0.20,
    "rppg": 0.50,
    "mer": 0.30
}

def get_class_ranks(probs):
    """
    Given a list of 4 probabilities, returns the rank for each class (0..3).
    Lowest probability gets rank 0, highest gets rank 3.
    """
    sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i])
    ranks = [0] * len(probs)
    for rank, class_idx in enumerate(sorted_indices):
        ranks[class_idx] = rank
    return ranks

def fuse_borda_count(samples, weights=WEIGHTS):
    wm = weights["macro"]
    wr = weights["rppg"]
    wmer = weights["mer"]

    predictions = []
    probabilities = []

    for s in samples:
        rm = get_class_ranks(s["macro"])
        rr = get_class_ranks(s["rppg"])
        rmer = get_class_ranks(s["mer"])

        borda_scores = [
            wm * rm[c] + wr * rr[c] + wmer * rmer[c] for c in range(4)
        ]

        # Normalized scores as pseudo-probabilities for output consistency
        total_score = sum(borda_scores)
        if total_score > 0:
            norm_probs = [score / total_score for score in borda_scores]
        else:
            norm_probs = [0.25] * 4

        # Deterministic tie breaking using average probability
        max_score = max(borda_scores)
        candidate_classes = [c for c in range(4) if borda_scores[c] == max_score]
        if len(candidate_classes) == 1:
            pred = candidate_classes[0]
        else:
            # Tie breaker: sum of raw probabilities
            avg_probs = [(s["macro"][c] + s["rppg"][c] + s["mer"][c]) for c in candidate_classes]
            best_cand_idx = avg_probs.index(max(avg_probs))
            pred = candidate_classes[best_cand_idx]

        predictions.append(pred)
        probabilities.append(norm_probs)

    return predictions, probabilities

def main():
    print(f"[*] Running {STRATEGY_NAME}...")
    samples = load_modality_data()
    print(f"[*] Loaded {len(samples)} clips.")

    predictions, probabilities = fuse_borda_count(samples)
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
