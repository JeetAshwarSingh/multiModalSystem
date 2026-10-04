# Robustness evaluation script

"""robustness.py
Randomly mask modalities during inference to evaluate robustness of each fusion method.

Usage example:
```bash
python -m evaluation.robustness \
    --fusion agentic \
    --mask-ratio 0.3 \
    --iterations 10
```
"""

import argparse
import pathlib
import random
import numpy as np
from evaluation.evaluate import main as evaluate_main

def parse_args():
    parser = argparse.ArgumentParser(description="Robustness test with random modality masking")
    parser.add_argument("--fusion", type=str, choices=["early", "static", "agentic"], required=True)
    parser.add_argument("--mask-ratio", type=float, default=0.3, help="Probability of dropping each modality per iteration")
    parser.add_argument("--iterations", type=int, default=10, help="Number of random masking runs")
    parser.add_argument("--output-dir", type=str, default="evaluation/results")
    return parser.parse_args()

def run_random_mask(fusion, mask_ratio, iteration, args):
    # Determine which modalities to disable based on mask_ratio
    modalities = ["macro", "rppg", "mer"]
    disabled = [m for m in modalities if random.random() < mask_ratio]
    # Reuse evaluate script with custom args via function call
    # Build a namespace mimicking argparse.Namespace expected by evaluate_main
    class ArgsNamespace:
        pass
    ns = ArgsNamespace()
    ns.config = "configs/fusion.yaml"
    ns.dataset = "daisee"
    ns.split = "test"
    ns.fusion = fusion
    ns.disable_modality = disabled
    ns.output_dir = args.output_dir
    # Run evaluation directly (it returns None, but writes CSV and prints summary)
    evaluate_main(ns)
    return disabled

def main():
    args = parse_args()
    results = []
    for i in range(args.iterations):
        disabled = run_random_mask(args.fusion, args.mask_ratio, i, args)
        results.append({"iteration": i, "disabled": disabled})
    # Summarize the masking schedule
    import json
    schedule_path = pathlib.Path(args.output_dir) / f"mask_schedule_{args.fusion}.json"
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    with open(schedule_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved masking schedule to {schedule_path}")

if __name__ == "__main__":
    main()
