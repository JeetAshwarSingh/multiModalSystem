"""generate_analysis.py
Standalone CLI tool to analyze prediction results and generate publication-ready figures.

Usage:
    python -m evaluation.generate_analysis [--dir ../resultAndAnalysis]
"""

import os
import sys
import json
import argparse
import pathlib
import numpy as np
import pandas as pd

from evaluation.analyze_and_plot import compute_detailed_metrics, generate_all_plots, generate_latex_table

def main():
    parser = argparse.ArgumentParser(description="Generate publication analysis and charts from evaluation results.")
    parser.add_argument("--dir", type=str, default="resultAndAnalysis", help="Directory containing prediction CSV files")
    args = parser.parse_args()

    target_dir = pathlib.Path(args.dir)
    if not target_dir.is_absolute():
        if (pathlib.Path("..") / args.dir).is_dir():
            target_dir = (pathlib.Path("..") / args.dir).resolve()
        elif (pathlib.Path(".") / args.dir).is_dir():
            target_dir = (pathlib.Path(".") / args.dir).resolve()
        elif (pathlib.Path("..") / "resultAndAnalysis").is_dir():
            target_dir = (pathlib.Path("..") / "resultAndAnalysis").resolve()
        else:
            target_dir = pathlib.Path(args.dir).resolve()

    if not target_dir.is_dir():
        print(f"Error: Directory '{target_dir}' does not exist.")
        sys.exit(1)

    # Check if target_dir has timestamped run directories and no direct CSVs
    run_subdirs = sorted([d for d in target_dir.glob("run_*") if d.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    has_direct_csvs = (target_dir / "fusion_agentic_predictions.csv").is_file() or (target_dir / "macro_predictions.csv").is_file()
    if not has_direct_csvs and run_subdirs:
        print(f"Target directory contains run directories. Using latest run: {run_subdirs[0].name}")
        target_dir = run_subdirs[0]

    analysis_dir = target_dir / "Analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    file_mapping = {
        "Macro Modality": target_dir / "macro_predictions.csv",
        "rPPG Modality": target_dir / "rppg_predictions.csv",
        "MER Modality": target_dir / "mer_predictions.csv",
        "Early Fusion": target_dir / "fusion_early_predictions.csv",
        "Static Late Fusion": target_dir / "fusion_static_predictions.csv",
        "Agentic Late Fusion": target_dir / "fusion_agentic_predictions.csv",
    }

    summary_records = []
    per_class_all = {}
    cm_all = {}
    json_metrics = {}

    for method_name, csv_path in file_mapping.items():
        if not csv_path.is_file():
            # Check fallback name
            fallback = target_dir / f"results_{method_name.lower().split()[0]}.csv"
            if fallback.is_file():
                csv_path = fallback
            else:
                continue

        df = pd.read_csv(csv_path)
        if "true_engagement" not in df.columns or "pred_engagement" not in df.columns:
            continue

        y_true = df["true_engagement"].astype(int).tolist()
        y_pred = df["pred_engagement"].astype(int).tolist()

        met = compute_detailed_metrics(y_true, y_pred)
        json_metrics[method_name] = met
        per_class_all[method_name] = met["per_class"]
        cm_all[method_name] = np.array(met["confusion_matrix"])

        summary_records.append({
            "Method": method_name,
            "Accuracy": met["accuracy"],
            "Macro-F1": met["macro_f1"],
            "Precision": met["precision"],
            "Recall": met["recall"]
        })

    if not summary_records:
        print(f"No prediction CSVs found in {target_dir}. Run evaluation first.")
        sys.exit(1)

    summary_df = pd.DataFrame(summary_records)
    summary_csv = target_dir / "overall_metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Updated summary CSV: {summary_csv}")

    json_file = target_dir / "metrics_summary.json"
    with open(json_file, "w") as f:
        json.dump(json_metrics, f, indent=2)
    print(f"Updated metrics JSON: {json_file}")

    # Generate charts
    agentic_weights = {"macro": [0.5], "rppg": [0.3], "mer": [0.2]}
    generate_all_plots(summary_df, per_class_all, cm_all, agentic_weights, analysis_dir)
    generate_latex_table(summary_df, analysis_dir / "paper_summary_table.tex")

    print("\nSuccessfully updated all charts and LaTeX table in:", analysis_dir)

if __name__ == "__main__":
    main()
