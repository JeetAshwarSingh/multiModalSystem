"""
common.py - Shared utilities for FusionStratStudy
Provides unified data loading, metric computation, report printing,
and result persistence for all late fusion strategies.
"""

import os
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

CLASS_NAMES = ["Class_0", "Class_1", "Class_2", "Class_3"]
CLASS_LABELS = [
    "Very Low Engagement (0)",
    "Low Engagement (1)",
    "Engaged (2)",
    "Highly Engaged (3)"
]

def find_project_root() -> Path:
    """Find the root Baseline_3 directory containing resultAndAnalysis/."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "resultAndAnalysis").is_dir():
            return current
        if current.parent == current:
            break
        current = current.parent
    # Fallback to current working directory
    cwd = Path.cwd()
    if (cwd / "resultAndAnalysis").is_dir():
        return cwd
    return Path(__file__).resolve().parent.parent

def load_modality_data(root_dir: Optional[Path] = None, run_dir_name: str = "run_20261010_011802") -> List[Dict[str, Any]]:
    """
    Loads and aligns samples across macro, rppg, and mer prediction CSVs.
    Specifically loads from resultAndAnalysis/run_20261010_011802/ by default.
    Returns a list of dicts with keys:
      clip_id, clip_path, true_y, macro, rppg, mer
      where macro, rppg, mer are 4-element probability lists.
    """
    if root_dir is None:
        root_dir = find_project_root()
    
    # Priority 1: Specifically use the run directory requested (run_20261010_011802)
    res_dir = root_dir / "resultAndAnalysis" / run_dir_name
    if not res_dir.is_dir():
        # Fallback to root resultAndAnalysis if run directory is missing
        res_dir = root_dir / "resultAndAnalysis"
    
    print(f"[*] Loading modality predictions directly from: {res_dir}")

    macro_csv = res_dir / "macro_predictions.csv"
    rppg_csv = res_dir / "rppg_predictions.csv"
    mer_csv = res_dir / "mer_predictions.csv"

    for path, name in [(macro_csv, "macro"), (rppg_csv, "rppg"), (mer_csv, "mer")]:
        if not path.is_file():
            raise FileNotFoundError(f"Required prediction file not found: {path}")

    samples = []
    with open(macro_csv, mode="r", encoding="utf-8") as fm, \
         open(rppg_csv, mode="r", encoding="utf-8") as fr, \
         open(mer_csv, mode="r", encoding="utf-8") as fmer:
        
        rm = csv.DictReader(fm)
        rr = csv.DictReader(fr)
        rmer = csv.DictReader(fmer)

        for idx, (m_row, r_row, mer_row) in enumerate(zip(rm, rr, rmer)):
            true_y = int(m_row["true_engagement"])
            pm = [float(m_row[f"prob_class_{i}"]) for i in range(4)]
            pr = [float(r_row[f"prob_class_{i}"]) for i in range(4)]
            pmer = [float(mer_row[f"prob_class_{i}"]) for i in range(4)]

            samples.append({
                "clip_id": m_row.get("clip_id", str(idx)),
                "clip_path": m_row.get("clip_path", ""),
                "true_y": true_y,
                "macro": pm,
                "rppg": pr,
                "mer": pmer
            })

    return samples

def compute_metrics(preds: List[int], trues: List[int], probs: Optional[List[List[float]]] = None) -> Dict[str, Any]:
    """Computes comprehensive classification metrics in pure Python (no scikit-learn dependency required)."""
    total = len(trues)
    if total == 0:
        return {}

    correct = sum(1 for p, t in zip(preds, trues) if p == t)
    accuracy = correct / total

    # Confusion matrix: cm[true_class][pred_class]
    cm = [[0] * 4 for _ in range(4)]
    for p, t in zip(preds, trues):
        cm[t][p] += 1

    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for c in range(4):
        tp = cm[c][c]
        pred_c = sum(cm[r][c] for r in range(4))
        true_c = sum(cm[c])
        prec = tp / pred_c if pred_c > 0 else 0.0
        rec = tp / true_c if true_c > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(true_c)

        per_class[CLASS_NAMES[c]] = {
            "label": CLASS_LABELS[c],
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(true_c)
        }

    macro_f1 = sum(f1s) / 4.0
    macro_precision = sum(precisions) / 4.0
    macro_recall = sum(recalls) / 4.0
    balanced_accuracy = macro_recall

    # Weighted F1
    weighted_f1 = sum(f1 * sup for f1, sup in zip(f1s, supports)) / total

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "confusion_matrix": cm,
        "per_class": per_class,
        "total_samples": total
    }

def print_evaluation_report(strategy_name: str, metrics: Dict[str, Any]):
    """Pretty prints an evaluation summary table to the console."""
    sep = "=" * 76
    sub_sep = "-" * 76
    print("\n" + sep)
    print(f"  FUSION STRATEGY EVALUATION REPORT: {strategy_name.upper()}")
    print(sep)
    print(f"  Total Samples:      {metrics.get('total_samples', 0)}")
    print(f"  Overall Accuracy:   {metrics.get('accuracy', 0.0) * 100:.2f}%")
    print(f"  Balanced Accuracy:  {metrics.get('balanced_accuracy', 0.0) * 100:.2f}%")
    print(f"  Macro-F1 Score:     {metrics.get('macro_f1', 0.0):.4f}")
    print(f"  Weighted-F1 Score:  {metrics.get('weighted_f1', 0.0):.4f}")
    print(f"  Macro Precision:    {metrics.get('macro_precision', 0.0):.4f}")
    print(f"  Macro Recall:       {metrics.get('macro_recall', 0.0):.4f}")
    print(sub_sep)
    print(f"  {'Class':<26} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print(sub_sep)
    for c_name, c_data in metrics.get("per_class", {}).items():
        label = c_data.get("label", c_name)
        p = c_data.get("precision", 0.0)
        r = c_data.get("recall", 0.0)
        f = c_data.get("f1", 0.0)
        s = c_data.get("support", 0)
        print(f"  {label:<26} | {p:>10.4f} | {r:>10.4f} | {f:>10.4f} | {s:>8}")
    print(sub_sep)
    print("  Confusion Matrix (Rows: Ground Truth, Cols: Predicted):")
    cm = metrics.get("confusion_matrix", [])
    for row_idx, row in enumerate(cm):
        print(f"    Class {row_idx}: {row}")
    print(sep + "\n")

def save_results(
    strategy_slug: str,
    strategy_display_name: str,
    samples: List[Dict[str, Any]],
    predictions: List[int],
    probabilities: List[List[float]],
    metrics: Dict[str, Any],
    target_dir: Optional[Path] = None
) -> Tuple[Path, Path]:
    """
    Saves predictions CSV and metrics JSON into target directory
    (defaults to FusionStratStudy/result and mirrors to FusionStratStudy/results).
    Also updates all_strategies_summary.csv.
    """
    if target_dir is None:
        root = find_project_root()
        target_dir = root / "FusionStratStudy" / "result"
    
    target_dir.mkdir(parents=True, exist_ok=True)
    # Also ensure results/ exists
    mirror_dir = target_dir.parent / "results"
    mirror_dir.mkdir(parents=True, exist_ok=True)

    csv_path = target_dir / f"{strategy_slug}_predictions.csv"
    json_path = target_dir / f"{strategy_slug}_metrics.json"

    # Write predictions CSV
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "clip_id", "clip_path", "pred_engagement", "true_engagement",
            "prob_class_0", "prob_class_1", "prob_class_2", "prob_class_3"
        ])
        for s, pred, prob in zip(samples, predictions, probabilities):
            writer.writerow([
                s["clip_id"],
                s["clip_path"],
                pred,
                s["true_y"],
                round(prob[0], 6),
                round(prob[1], 6),
                round(prob[2], 6),
                round(prob[3], 6)
            ])

    # Write metrics JSON
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Mirror to results/
    mirror_csv = mirror_dir / f"{strategy_slug}_predictions.csv"
    mirror_json = mirror_dir / f"{strategy_slug}_metrics.json"
    import shutil
    shutil.copyfile(csv_path, mirror_csv)
    shutil.copyfile(json_path, mirror_json)

    # Update master summary CSV
    summary_csv = target_dir / "all_strategies_summary.csv"
    existing_records = []
    if summary_csv.is_file():
        with open(summary_csv, mode="r", encoding="utf-8") as sf:
            reader = csv.DictReader(sf)
            for r in reader:
                if r.get("Strategy_ID") != strategy_slug:
                    existing_records.append(r)

    new_record = {
        "Strategy_ID": strategy_slug,
        "Strategy_Name": strategy_display_name,
        "Accuracy": f"{metrics.get('accuracy', 0.0):.6f}",
        "Balanced_Accuracy": f"{metrics.get('balanced_accuracy', 0.0):.6f}",
        "Macro_F1": f"{metrics.get('macro_f1', 0.0):.6f}",
        "Weighted_F1": f"{metrics.get('weighted_f1', 0.0):.6f}",
        "Macro_Precision": f"{metrics.get('macro_precision', 0.0):.6f}",
        "Macro_Recall": f"{metrics.get('macro_recall', 0.0):.6f}",
        "F1_Class_0": f"{metrics.get('per_class', {}).get('Class_0', {}).get('f1', 0.0):.4f}",
        "F1_Class_1": f"{metrics.get('per_class', {}).get('Class_1', {}).get('f1', 0.0):.4f}",
        "F1_Class_2": f"{metrics.get('per_class', {}).get('Class_2', {}).get('f1', 0.0):.4f}",
        "F1_Class_3": f"{metrics.get('per_class', {}).get('Class_3', {}).get('f1', 0.0):.4f}",
    }
    existing_records.append(new_record)

    fieldnames = list(new_record.keys())
    with open(summary_csv, mode="w", newline="", encoding="utf-8") as sf:
        writer = csv.DictWriter(sf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_records)
    
    # Mirror summary CSV
    shutil.copyfile(summary_csv, mirror_dir / "all_strategies_summary.csv")

    print(f"  [+] Saved fused predictions to: {csv_path}")
    print(f"  [+] Saved metrics summary to:    {json_path}")
    print(f"  [+] Updated master summary at:  {summary_csv}")

    return csv_path, json_path
