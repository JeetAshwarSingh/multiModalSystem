"""analyze_and_plot.py
Comprehensive analysis and publication-ready visualization generation for Engagement-MAS.
Generates metrics CSVs, JSONs, LaTeX table, and publication-ready Seaborn/Matplotlib figures.
"""

import os
import json
import pathlib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Setup publication style formatting with Matplotlib and Seaborn
def _setup_plot_style():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

def compute_detailed_metrics(y_true: List[int], y_pred: List[int], num_classes: int = 4) -> Dict[str, Any]:
    """Computes comprehensive classification metrics across all classes."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
    
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    
    acc = float(accuracy_score(y_t, y_p))
    balanced_acc = float(balanced_accuracy_score(y_t, y_p))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_t, y_p, average="weighted", zero_division=0))
    prec = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    rec = float(recall_score(y_t, y_p, average="macro", zero_division=0))
    
    cm = confusion_matrix(y_t, y_p, labels=list(range(num_classes)))
    
    per_class = {}
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[f"Class_{c}"] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "support": int(cm[c, :].sum())
        }
        
    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "total_samples": len(y_true)
    }

def generate_all_plots(summary_df: pd.DataFrame, 
                       per_class_data: Dict[str, Dict[str, Any]], 
                       confusion_matrices: Dict[str, np.ndarray], 
                       agentic_weights: Dict[str, List[float]], 
                       analysis_dir: pathlib.Path):
    """Generates all publication-quality research paper figures."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    _setup_plot_style()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------
    # Fig 1: Modality vs Fusion Accuracy Benchmark (All modalities + Fusion)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    df_sorted = summary_df.copy()
    
    # Define custom palette: cool colors for single modalities, warm/vibrant for fusions
    palette = []
    for m in df_sorted["Method"]:
        if "Fusion" in m:
            palette.append("#e74c3c" if "Agentic" in m else ("#f39c12" if "Static" in m else "#9b59b6"))
        else:
            palette.append("#3498db" if "Macro" in m else ("#1abc9c" if "rPPG" in m else "#2ecc71"))
            
    bars = ax.bar(df_sorted["Method"], df_sorted["Accuracy"] * 100, color=palette, width=0.55, edgecolor="black", linewidth=0.8)
    
    # Add values on top of bars
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold", fontsize=10)
        
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Modality vs. Fusion Strategy Performance Benchmark", fontweight="bold", pad=15)
    ax.set_ylim(0, max(df_sorted["Accuracy"] * 100) * 1.18 + 5)
    plt.xticks(rotation=18, ha="right", fontweight="semibold")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    fig1_path = analysis_dir / "fig1_modality_vs_fusion_accuracy.png"
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved: {fig1_path}")

    # -------------------------------------------------------------
    # Fig 2: Fusion Methods Comparison (Multi-metric grouped bar chart)
    # -------------------------------------------------------------
    fusion_rows = df_sorted[df_sorted["Method"].str.contains("Fusion")].copy()
    if not fusion_rows.empty:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        metrics_to_plot = ["Accuracy", "Macro-F1", "Precision", "Recall"]
        fusion_long = pd.melt(fusion_rows, id_vars=["Method"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
        
        sns.barplot(data=fusion_long, x="Metric", y="Score", hue="Method", palette=["#9b59b6", "#f39c12", "#e74c3c"], ax=ax, edgecolor="black", linewidth=0.7)
        ax.set_ylabel("Score (0.0 – 1.0)", fontweight="bold")
        ax.set_title("Head-to-Head Comparison of Multi-Modal Fusion Strategies", fontweight="bold", pad=15)
        ax.set_ylim(0, 1.15)
        ax.legend(title="Strategy", frameon=True, facecolor="white", edgecolor="none")
        plt.tight_layout()
        fig2_path = analysis_dir / "fig2_fusion_methods_comparison.png"
        plt.savefig(fig2_path)
        plt.close()
        print(f"Saved: {fig2_path}")

    # -------------------------------------------------------------
    # Fig 3: Per-Class Engagement F1-Scores
    # -------------------------------------------------------------
    class_rows = []
    class_labels = ["0 (Very Low)", "1 (Low)", "2 (Engaged)", "3 (High)"]
    for method_name, class_dict in per_class_data.items():
        for idx in range(4):
            c_key = f"Class_{idx}"
            if c_key in class_dict:
                class_rows.append({
                    "Method": method_name,
                    "Class": class_labels[idx],
                    "F1-Score": class_dict[c_key]["f1"]
                })
    if class_rows:
        df_class = pd.DataFrame(class_rows)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        sns.barplot(data=df_class, x="Class", y="F1-Score", hue="Method", ax=ax, palette="tab10", edgecolor="black", linewidth=0.6)
        ax.set_ylabel("F1-Score", fontweight="bold")
        ax.set_xlabel("Engagement Level", fontweight="bold")
        ax.set_title("Per-Class Engagement Recognition F1-Score across Methods", fontweight="bold", pad=15)
        ax.set_ylim(0, 1.15)
        ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        fig3_path = analysis_dir / "fig3_per_class_f1_comparison.png"
        plt.savefig(fig3_path)
        plt.close()
        print(f"Saved: {fig3_path}")

    # -------------------------------------------------------------
    # Fig 4: Confusion Matrices Multi-panel Heatmaps
    # -------------------------------------------------------------
    selected_methods = [m for m in ["Macro Modality", "Static Late Fusion", "Agentic Late Fusion"] if m in confusion_matrices]
    if not selected_methods:
        selected_methods = list(confusion_matrices.keys())[:3]
        
    n_panels = len(selected_methods)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.0 * n_panels, 4.5), sharey=True)
    if n_panels == 1:
        axes = [axes]
        
    for ax, m_name in zip(axes, selected_methods):
        cm = confusion_matrices[m_name]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["0", "1", "2", "3"], yticklabels=["0", "1", "2", "3"],
                    annot_kws={"size": 11, "weight": "bold"})
        ax.set_title(m_name, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontweight="semibold")
        if ax == axes[0]:
            ax.set_ylabel("Ground Truth Label", fontweight="semibold")
            
    fig.suptitle("Confusion Matrix Heatmaps for Baseline and Fusion Systems", fontweight="bold", y=1.03)
    plt.tight_layout()
    fig4_path = analysis_dir / "fig4_confusion_matrices.png"
    plt.savefig(fig4_path)
    plt.close()
    print(f"Saved: {fig4_path}")

    # -------------------------------------------------------------
    # Fig 5: Agentic Modality Weight Adaptations
    # -------------------------------------------------------------
    if agentic_weights and any(len(v) > 0 for v in agentic_weights.values()):
        fig, ax = plt.subplots(figsize=(8, 5))
        w_df = pd.DataFrame(agentic_weights)
        sns.boxplot(data=w_df, palette=["#3498db", "#1abc9c", "#2ecc71"], ax=ax, width=0.45)
        sns.stripplot(data=w_df, color="black", alpha=0.6, jitter=0.2, size=5, ax=ax)
        ax.set_ylabel("Adaptive Weight Assigned", fontweight="bold")
        ax.set_xlabel("Perception Modality", fontweight="bold")
        ax.set_title("Agentic Fusion Adaptive Modality Weight Distributions", fontweight="bold", pad=15)
        ax.set_ylim(0.0, 1.0)
        plt.tight_layout()
        fig5_path = analysis_dir / "fig5_agentic_modality_weights.png"
        plt.savefig(fig5_path)
        plt.close()
        print(f"Saved: {fig5_path}")

    # -------------------------------------------------------------
    # Fig 6: Radar Comparison Profile
    # -------------------------------------------------------------
    try:
        categories = ["Accuracy", "Macro-F1", "Precision", "Recall"]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        color_map = {"Macro Modality": "#3498db", "Static Late Fusion": "#f39c12", "Agentic Late Fusion": "#e74c3c"}
        
        for idx, row in df_sorted.iterrows():
            m = row["Method"]
            if m in color_map:
                values = [row[c] for c in categories]
                values += values[:1]
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=m, color=color_map[m])
                ax.fill(angles, values, color=color_map[m], alpha=0.15)
                
        plt.xticks(angles[:-1], categories, color='grey', size=11, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_title("Comprehensive Performance Radar Profile", fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        fig6_path = analysis_dir / "fig6_radar_performance_profile.png"
        plt.savefig(fig6_path)
        plt.close()
        print(f"Saved: {fig6_path}")
    except Exception as e:
        print(f"Radar plot skipped: {e}")

def generate_latex_table(summary_df: pd.DataFrame, out_path: pathlib.Path):
    """Generates an IEEE/ACM-ready LaTeX table snippet."""
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Performance Comparison of Individual Modalities and Multi-Modal Fusion Strategies on DAiSEE}",
        r"\label{tab:engagement_results}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"\textbf{Method / Architecture} & \textbf{Accuracy (\%)} & \textbf{Macro-F1} & \textbf{Precision} & \textbf{Recall} \\",
        r"\hline"
    ]
    for _, row in summary_df.iterrows():
        name = row["Method"]
        acc = f"{row['Accuracy']*100:.2f}"
        f1 = f"{row['Macro-F1']:.4f}"
        p = f"{row['Precision']:.4f}"
        r = f"{row['Recall']:.4f}"
        if "Agentic" in name:
            tex_lines.append(f"\\textbf{{{name}}} & \\textbf{{{acc}}} & \\textbf{{{f1}}} & \\textbf{{{p}}} & \\textbf{{{r}}} \\\\")
        else:
            tex_lines.append(f"{name} & {acc} & {f1} & {p} & {r} \\\\")
            
    tex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}"
    ])
    with open(out_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Saved LaTeX table: {out_path}")
