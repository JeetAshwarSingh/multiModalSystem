#!/usr/bin/env python3
"""
plot_balanced_accuracy_table.py
Generates publication-quality figures of the Balanced Accuracy Ranking Table
and saves them as PNG images in FusionStratStudy/result/, FusionStratStudy/results/,
and resultAndAnalysis/Analysis/.
"""

import os
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def load_data():
    project_root = Path(__file__).resolve().parent.parent
    base_json = project_root / "resultAndAnalysis" / "run_20261010_011802" / "metrics_summary.json"
    if not base_json.is_file():
        base_json = project_root / "resultAndAnalysis" / "metrics_summary.json"
    
    methods = []
    
    # 1. Load original baselines
    if base_json.is_file():
        with open(base_json) as f:
            raw_base = json.load(f)
            for name, data in raw_base.items():
                cm = data["confusion_matrix"]
                recalls = [cm[c][c] / sum(cm[c]) if sum(cm[c]) > 0 else 0.0 for c in range(4)]
                bal_acc = sum(recalls) / 4.0
                methods.append({
                    "name": f"{name} (Baseline)",
                    "type": "Baseline",
                    "bal_acc": bal_acc,
                    "r0": recalls[0],
                    "r1": recalls[1],
                    "r2": recalls[2],
                    "r3": recalls[3],
                    "acc": data.get("accuracy", 0.0),
                    "macro_f1": data.get("macro_f1", 0.0)
                })

    # 2. Load proposed strategies
    display_names = {
        "01_equal_weight_average": "01. Equal-Weight Average",
        "02_performance_weighted_fusion": "02. Performance-Weighted Fusion",
        "03_borda_count_rank_fusion": "03. Borda Count Rank Fusion",
        "04_temperature_scaled_softmax_fusion": "04. Temperature-Scaled Fusion",
        "05_entropy_uncertainty_gated_fusion": "05. Entropy Uncertainty Gating",
        "06_dispersion_variance_aware_fusion": "06. Dispersion-Aware Fusion",
        "07_class_prior_cost_sensitive_fusion": "07. Prior-Adjusted Cost-Sensitive",
        "08_hierarchical_cascade_fusion": "08. Hierarchical Cascade Fusion",
        "09_stacking_meta_classifier_fusion": "09. Stacking Meta-Classifier",
    }

    metrics_files = sorted(glob.glob(str(project_root / "FusionStratStudy" / "result" / "*_metrics.json")))
    for p in metrics_files:
        slug = os.path.basename(p).replace("_metrics.json", "")
        with open(p) as f:
            data = json.load(f)
            cm = data["confusion_matrix"]
            recalls = [cm[c][c] / sum(cm[c]) if sum(cm[c]) > 0 else 0.0 for c in range(4)]
            bal_acc = sum(recalls) / 4.0
            methods.append({
                "name": display_names.get(slug, slug),
                "type": "Proposed Strategy",
                "bal_acc": bal_acc,
                "r0": recalls[0],
                "r1": recalls[1],
                "r2": recalls[2],
                "r3": recalls[3],
                "acc": data.get("accuracy", 0.0),
                "macro_f1": data.get("macro_f1", 0.0)
            })

    # Sort descending by Balanced Accuracy
    methods.sort(key=lambda x: x["bal_acc"], reverse=True)
    return methods

def plot_publication_table(methods, save_paths):
    """Renders a beautifully styled tabular chart saved as PNG."""
    fig, ax = plt.subplots(figsize=(15, 9.5), dpi=300)
    ax.axis("off")

    headers = [
        "Rank", "Method / Fusion Strategy", "Type",
        "Balanced\nAccuracy", "Recall C0\n(Very Low)", "Recall C1\n(Low)",
        "Recall C2\n(Engaged)", "Recall C3\n(High)", "Overall\nAccuracy", "Macro-F1"
    ]

    cell_text = []
    cell_colors = []

    cmap_green = plt.get_cmap("YlGn")
    cmap_blue = plt.get_cmap("Blues")

    for idx, m in enumerate(methods, 1):
        is_top_3 = idx <= 3
        is_proposed = m["type"] == "Proposed Strategy"
        
        row = [
            f"#{idx}",
            m["name"],
            "Proposed" if is_proposed else "Baseline",
            f"{m['bal_acc']*100:.2f}%",
            f"{m['r0']*100:.1f}%",
            f"{m['r1']*100:.1f}%",
            f"{m['r2']*100:.1f}%",
            f"{m['r3']*100:.1f}%",
            f"{m['acc']*100:.2f}%",
            f"{m['macro_f1']:.4f}",
        ]
        cell_text.append(row)

        # Base row coloring
        if idx % 2 == 0:
            base_color = "#f8f9fa"
        else:
            base_color = "#ffffff"

        row_c = [base_color] * len(headers)

        # Highlight rank
        if is_top_3:
            row_c[0] = "#e8f5e9"
            row_c[1] = "#e8f5e9"

        # Highlight Balanced Accuracy with subtle green heat
        norm_val = min(max((m["bal_acc"] - 0.20) / 0.20, 0.0), 1.0)
        row_c[3] = mcolors.to_hex(cmap_green(0.15 + 0.65 * norm_val))

        # Color the per-class recalls subtly
        for c_idx, rec in enumerate([m['r0'], m['r1'], m['r2'], m['r3']], start=4):
            if rec > 0.0:
                c_norm = min(rec, 1.0)
                row_c[c_idx] = mcolors.to_hex(cmap_blue(0.08 + 0.40 * c_norm))

        cell_colors.append(row_c)

    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colors,
        colLabels=headers,
        colWidths=[0.06, 0.28, 0.09, 0.10, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.7)

    # Style header row
    for col_idx in range(len(headers)):
        cell = table[0, col_idx]
        cell.set_facecolor("#1a237e")
        cell.set_text_props(color="white", weight="bold", fontsize=10.5)

    # Left-align method names
    for row_idx in range(1, len(methods) + 1):
        cell = table[row_idx, 1]
        cell.set_text_props(ha="left")

    title_text = "Multimodal Engagement Classification: Class-Wise Balanced Accuracy Leaderboard"
    subtitle_text = "Evaluated on DAiSEE Test Set (1,784 Clips: Class 0: 4 | Class 1: 84 | Class 2: 882 | Class 3: 814)"
    plt.title(f"{title_text}\n{subtitle_text}", fontsize=13, weight="bold", pad=25, color="#1a237e")

    plt.tight_layout()
    for sp in save_paths:
        sp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(sp, bbox_inches="tight", dpi=300)
        print(f"[+] Saved table figure to: {sp}")
    plt.close()

def plot_horizontal_recall_chart(methods, save_paths):
    """Renders a companion grouped bar chart comparing Balanced Accuracy and class recalls."""
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)

    names = [m["name"] for m in reversed(methods)]
    bal_accs = [m["bal_acc"] * 100 for m in reversed(methods)]
    r0 = [m["r0"] * 100 for m in reversed(methods)]
    r1 = [m["r1"] * 100 for m in reversed(methods)]
    r2 = [m["r2"] * 100 for m in reversed(methods)]
    r3 = [m["r3"] * 100 for m in reversed(methods)]

    y_pos = np.arange(len(names))
    bar_height = 0.65

    # Plot Balanced Accuracy bars
    bars = ax.barh(y_pos, bal_accs, height=bar_height, color="#3949ab", alpha=0.88, label="Balanced Accuracy (%)")

    # Add reference line at 25.0% (random guess / single-class floor)
    ax.axvline(25.0, color="#d32f2f", linestyle="--", linewidth=1.5, label="25.0% Baseline Floor (All-Class-3 Mode Collapse)")

    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.6, bar.get_y() + bar.get_height()/2.0, f"{width:.1f}%",
                va="center", ha="left", fontsize=9, weight="bold", color="#1a237e")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Balanced Accuracy (%)", fontsize=11, weight="bold")
    ax.set_xlim(0, 48)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    ax.set_title("Engagement Detection Models Ranked by Balanced Accuracy\nDemonstrating Revival of Minority Classes over Mode-Collapsed Baselines",
                 fontsize=13, weight="bold", pad=15, color="#1a237e")

    plt.tight_layout()
    for sp in save_paths:
        sp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(sp, bbox_inches="tight", dpi=300)
        print(f"[+] Saved horizontal chart to: {sp}")
    plt.close()

def main():
    methods = load_data()
    print(f"[*] Loaded {len(methods)} methods.")

    project_root = Path(__file__).resolve().parent.parent

    # Target save paths
    table_paths = [
        project_root / "FusionStratStudy" / "result" / "balanced_accuracy_ranking_table.png",
        project_root / "FusionStratStudy" / "results" / "balanced_accuracy_ranking_table.png",
        project_root / "resultAndAnalysis" / "Analysis" / "balanced_accuracy_ranking_table.png",
    ]

    chart_paths = [
        project_root / "FusionStratStudy" / "result" / "balanced_accuracy_comparison_chart.png",
        project_root / "FusionStratStudy" / "results" / "balanced_accuracy_comparison_chart.png",
        project_root / "resultAndAnalysis" / "Analysis" / "balanced_accuracy_comparison_chart.png",
    ]

    plot_publication_table(methods, table_paths)
    plot_horizontal_recall_chart(methods, chart_paths)
    print("[*] All publication figures generated successfully!")

if __name__ == "__main__":
    main()
