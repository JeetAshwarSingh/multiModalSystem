# Metrics utilities

"""metrics.py
Utility functions for computing common classification metrics for engagement estimation.
All functions accept NumPy arrays or PyTorch tensors (converted internally).
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from typing import Tuple, Dict, List

def _to_numpy(x):
    if isinstance(x, (list, tuple)):
        return np.array(x)
    if hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Overall accuracy (fraction of correctly predicted samples)."""
    return accuracy_score(_to_numpy(y_true), _to_numpy(y_pred))

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro‑averaged F1 across all classes."""
    return f1_score(_to_numpy(y_true), _to_numpy(y_pred), average="macro", zero_division=0)

def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, Dict[str, float]]:
    """Return precision, recall, F1 for each class label.
    Returns a dict ``{class_label: {"precision":…, "recall":…, "f1":…}}``.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {int(lbl): {"precision": float(p), "recall": float(r), "f1": float(f)}
            for lbl, p, r, f in zip(labels, precision, recall, f1)}

def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return the confusion matrix (rows = true, columns = pred)."""
    return confusion_matrix(_to_numpy(y_true), _to_numpy(y_pred))

def summary_dict(y_true, y_pred) -> Dict[str, float]:
    """Convenient one‑liner returning a dict of key metrics."""
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
    }
