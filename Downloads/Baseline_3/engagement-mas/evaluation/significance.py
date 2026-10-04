# Statistical significance script

"""significance.py
Run the full evaluation across multiple random seeds for the macro‑training phase and compare fusion methods using paired t‑tests.

Usage example:
```bash
python -m evaluation.significance \
    --seeds 0 1 2 3 4 \
    --fusion-methods early static agentic \
    --dataset daisee \
    --split test
```
"""

import argparse
import pathlib
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from evaluation import evaluate  # reuse the evaluate script's main logic

def parse_args():
    parser = argparse.ArgumentParser(description="Statistical significance analysis for fusion methods")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="List of random seeds to use for macro training")
    parser.add_argument("--fusion-methods", type=str, nargs="+", choices=["early", "static", "agentic"], required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=str, default="evaluation/results")
    return parser.parse_args()

def run_one_seed(seed, fusion, args):
    # Set seed for reproducibility (affects macro training checkpoint loading)
    import torch, random, numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Call evaluate.main with a fabricated argparse.Namespace
    ns = argparse.Namespace()
    ns.config = "configs/fusion.yaml"
    ns.dataset = args.dataset
    ns.split = args.split
    ns.fusion = fusion
    ns.disable_modality = []
    ns.output_dir = args.output_dir
    # The evaluate.main function prints summary and writes CSV; we capture the returned metrics via a helper
    # We'll monkey‑patch evaluate.main to return the macro_f1 value.
    # For simplicity, we directly call evaluate.main and then read the generated CSV.
    evaluate.main(ns)
    # Load the CSV produced for this fusion method
    csv_path = pathlib.Path(args.output_dir) / f"results_{fusion}.csv"
    df = pd.read_csv(csv_path)
    # Compute macro_f1 using the evaluation.metrics utilities
    from evaluation.metrics import macro_f1
    return macro_f1(df["true_engagement"].values, df["pred_engagement"].values)

def main():
    args = parse_args()
    results = {fusion: [] for fusion in args.fusion_methods}
    for seed in args.seeds:
        for fusion in args.fusion_methods:
            f1 = run_one_seed(seed, fusion, args)
            results[fusion].append(f1)
    # Perform paired t‑tests between each pair of methods
    report = []
    for i, f1 in enumerate(args.fusion_methods):
        for j in range(i + 1, len(args.fusion_methods)):
            f2 = args.fusion_methods[j]
            t_stat, p_val = ttest_rel(results[f1], results[f2])
            report.append({
                "method_a": f1,
                "method_b": f2,
                "mean_a": np.mean(results[f1]),
                "mean_b": np.mean(results[f2]),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
            })
    # Save report JSON
    out_path = pathlib.Path(args.output_dir) / "significance_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("Significance report saved to", out_path)
    for entry in report:
        print(f"{entry['method_a']} vs {entry['method_b']}: p={entry['p_value']:.4g}, mean_a={entry['mean_a']:.4f}, mean_b={entry['mean_b']:.4f}")

if __name__ == "__main__":
    main()
