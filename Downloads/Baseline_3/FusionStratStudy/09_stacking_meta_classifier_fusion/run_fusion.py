#!/usr/bin/env python3
"""
run_fusion.py - Strategy 09: Decision Stacking & Meta-Learner Late Fusion
Trains a standardized, class-balanced multinomial meta-classifier on concatenated 12-dim modality probabilities.
"""

import sys
import math
import random
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

STRATEGY_SLUG = "09_stacking_meta_classifier_fusion"
STRATEGY_NAME = "Decision Stacking & Meta-Learner Late Fusion"

def extract_meta_features(samples):
    X = []
    y = []
    for s in samples:
        # Concatenate 4 + 4 + 4 = 12 probabilities
        feat = s["macro"] + s["rppg"] + s["mer"]
        X.append(feat)
        y.append(s["true_y"])
    return X, y

def standardize_features(X_train, X_test=None):
    """Zero-centers and normalizes variance to prevent constant modalities from dominating."""
    D = len(X_train[0])
    N = len(X_train)
    means = [sum(X_train[i][j] for i in range(N)) / N for j in range(D)]
    stds = [
        math.sqrt(sum((X_train[i][j] - means[j]) ** 2 for i in range(N)) / N) + 1e-5
        for j in range(D)
    ]
    X_train_norm = [[(X_train[i][j] - means[j]) / stds[j] for j in range(D)] for i in range(N)]
    if X_test is not None:
        X_test_norm = [[(X_test[i][j] - means[j]) / stds[j] for j in range(D)] for i in range(len(X_test))]
        return X_train_norm, X_test_norm
    return X_train_norm, means, stds

def softmax_probs(logits):
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    sum_e = sum(exps)
    return [e / sum_e for e in exps]

def train_meta_classifier(X_train, y_train, class_weights, lr=0.03, epochs=60, l2=0.02):
    """
    Trains a regularized multinomial logistic regression meta-model in pure Python.
    """
    D = len(X_train[0])  # 12
    C = 4                # 4 classes
    W = [[0.0] * D for _ in range(C)]
    b = [0.0] * C
    N = len(X_train)

    for epoch in range(epochs):
        for i in range(N):
            x = X_train[i]
            target = y_train[i]
            cw = class_weights[target]

            # Forward pass
            logits = [sum(W[c][j] * x[j] for j in range(D)) + b[c] for c in range(C)]
            probs = softmax_probs(logits)

            # Gradient descent with class weighting & L2 regularization
            for c in range(C):
                error = probs[c] - (1.0 if c == target else 0.0)
                grad = cw * error
                for j in range(D):
                    W[c][j] -= lr * (grad * x[j] + l2 * W[c][j])
                b[c] -= lr * grad

    return W, b

def predict_meta(W, b, x):
    D = len(x)
    C = len(W)
    logits = [sum(W[c][j] * x[j] for j in range(D)) + b[c] for c in range(C)]
    probs = softmax_probs(logits)
    pred = probs.index(max(probs))
    return pred, probs

def k_fold_cross_validation(X, y, k=5, epochs=60):
    """Performs stratified K-fold cross validation with standardized features."""
    random.seed(42)
    N = len(X)

    # Stratified split by class
    class_indices = {c: [] for c in range(4)}
    for idx, label in enumerate(y):
        class_indices[label].append(idx)

    for c in class_indices:
        random.shuffle(class_indices[c])

    folds = [[] for _ in range(k)]
    for c, indices in class_indices.items():
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)

    # Class weights: inverse frequency with power smoothing
    class_counts = [len(class_indices[c]) for c in range(4)]
    class_weights = [(N / (4.0 * max(cnt, 1))) ** 0.6 for cnt in class_counts]

    oof_predictions = [0] * N
    oof_probabilities = [[0.0] * 4 for _ in range(N)]

    for fold_idx in range(k):
        val_indices = set(folds[fold_idx])
        train_indices = [i for i in range(N) if i not in val_indices]

        X_train_raw = [X[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        X_val_raw = [X[i] for i in val_indices]

        # Standardize strictly within fold to avoid data leakage
        X_train_norm, X_val_norm = standardize_features(X_train_raw, X_val_raw)

        W, b = train_meta_classifier(X_train_norm, y_train, class_weights, lr=0.03, epochs=epochs, l2=0.02)

        for local_idx, val_i in enumerate(val_indices):
            pred, probs = predict_meta(W, b, X_val_norm[local_idx])
            oof_predictions[val_i] = pred
            oof_probabilities[val_i] = probs

    return oof_predictions, oof_probabilities

def main():
    parser = argparse.ArgumentParser(description=f"Run {STRATEGY_NAME}")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs (default: 60)")
    args = parser.parse_args()

    display_name = f"{STRATEGY_NAME} (Standardized {args.k_folds}-Fold CV)"
    print(f"[*] Running {display_name}...")
    samples = load_modality_data()
    print(f"[*] Loaded {len(samples)} clips.")

    X, y = extract_meta_features(samples)
    oof_preds, oof_probs = k_fold_cross_validation(X, y, k=args.k_folds, epochs=args.epochs)

    metrics = compute_metrics(oof_preds, y, oof_probs)
    print_evaluation_report(display_name, metrics)

    save_results(
        strategy_slug=STRATEGY_SLUG,
        strategy_display_name=display_name,
        samples=samples,
        predictions=oof_preds,
        probabilities=oof_probs,
        metrics=metrics
    )

if __name__ == "__main__":
    main()
