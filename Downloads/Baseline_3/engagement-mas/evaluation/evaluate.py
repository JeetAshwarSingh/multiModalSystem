# Evaluation script

"""evaluate.py
Run the full Engagement‑MAS pipeline on a dataset split and report metrics for a chosen fusion strategy.

Usage example:
```bash
python -m evaluation.evaluate \
    --config configs/fusion.yaml \
    --dataset daisee \
    --split test \
    --fusion agentic
```
"""

import os
import argparse
import time
import pathlib
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Project imports (assumes the repository root is in PYTHONPATH)
from blocks.macro.macro_block import MacroBlock
from blocks.rppg.rppg_block import RPPGBlock
from blocks.mer.mer_block import MERBlock
from gate.sync_gate import SyncGate
from fusion.early_fusion import EarlyFusion
from fusion.static_late_fusion import StaticLateFusion
from fusion.agentic_fusion import AgenticFusion
from evaluation.metrics import summary_dict, confusion, per_class_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation for Engagement‑MAS")
    parser.add_argument("--config", type=str, required=True, help="Path to fusion config YAML")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (matches entry in configs/datasets.yaml)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (e.g. test, val, train, test_quick)")
    parser.add_argument("--fusion", type=str, choices=["early", "static", "agentic"], required=True)
    parser.add_argument("--disable-modality", type=str, nargs="*", default=[], help="Modality names to mask (macro, rppg, mer)")
    parser.add_argument("--max-clips", type=int, default=None, help="Optional max number of clips to evaluate")
    parser.add_argument("--output-dir", type=str, default="evaluation/results")
    return parser.parse_args()

def load_dataset(csv_path: pathlib.Path, root_dir: pathlib.Path = None):
    import pandas as pd
    df = pd.read_csv(csv_path)
    paths = []
    for p in df["clip_path"]:
        p_str = str(p)
        if root_dir is not None and not os.path.isabs(p_str):
            paths.append(str(root_dir / p_str))
        else:
            paths.append(p_str)
    label_cols = [c for c in ["Engagement", "Boredom", "Confusion", "Frustration"] if c in df.columns]
    return paths, df[label_cols].values

def main(args=None):
    if args is None:
        args = parse_args()
    
    # Load dataset config if available
    datasets_cfg_path = pathlib.Path("configs/datasets.yaml")
    root_dir = None
    csv_path = None
    if datasets_cfg_path.is_file():
        import yaml
        with open(datasets_cfg_path) as f:
            ds_all = yaml.safe_load(f).get("datasets", {})
            if args.dataset in ds_all:
                ds_cfg = ds_all[args.dataset]
                cfg_root = ds_cfg.get("root_dir", "")
                if cfg_root and cfg_root != "/path/to/daisee_root":
                    root_dir = pathlib.Path(cfg_root).expanduser().resolve()
                    csv_key = f"{args.split}_csv"
                    csv_name = str(ds_cfg.get(csv_key, f"{args.split}.csv"))
                    csv_p = pathlib.Path(csv_name)
                    if csv_p.is_absolute() and csv_p.is_file():
                        csv_path = csv_p
                    elif (root_dir / csv_name).is_file():
                        csv_path = root_dir / csv_name
                    elif (pathlib.Path("..") / csv_name).is_file():
                        csv_path = pathlib.Path("..") / csv_name
                    elif csv_p.is_file():
                        csv_path = csv_p
                    else:
                        csv_path = root_dir / csv_name

    if csv_path is None or not csv_path.is_file():
        if (pathlib.Path("..") / f"{args.split}.csv").is_file():
            csv_path = pathlib.Path("..") / f"{args.split}.csv"
        else:
            csv_path = pathlib.Path("data") / args.dataset / f"{args.split}.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Could not find dataset CSV for split '{args.split}' at '{csv_path}'. "
            f"Please verify 'configs/datasets.yaml' or place '{args.split}.csv' in 'data/{args.dataset}/'."
        )

    print(f"Loading dataset from {csv_path} (root_dir={root_dir})...")
    clip_paths, labels = load_dataset(csv_path, root_dir)
    if args.max_clips is not None and args.max_clips > 0:
        clip_paths = clip_paths[:args.max_clips]
        labels = labels[:args.max_clips]
        print(f"Truncated to first {len(clip_paths)} clips (--max-clips {args.max_clips}).")
    else:
        print(f"Found {len(clip_paths)} clips.")

    # Initialise blocks
    macro = MacroBlock()
    rppg = RPPGBlock()
    mer = MERBlock()
    gate = SyncGate()

    # Run blocks in parallel per clip
    start_time = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {}
        for idx, clip_path in enumerate(clip_paths):
            future = executor.submit(run_all_modalities, idx, clip_path, macro, rppg, mer, args.disable_modality)
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            clip_idx, clip_id, modality_results = future.result()
            # Feed each modality result to the gate
            for res in modality_results:
                gate.receive(res)
            # After all results for a clip have been pushed, check for release
            bundle = gate.check_and_release(clip_id)
            if bundle is not None:
                results.append((clip_idx, clip_id, bundle))

    # After processing all clips, flush any remaining pending bundles (timeout)
    gate.finalize()
    for pending_id in gate.pending_ids():
        bundle = gate.check_and_release(pending_id)
        if bundle:
            # pending_id is str(clip_idx)
            try:
                c_idx = int(pending_id)
            except ValueError:
                c_idx = len(results)
            results.append((c_idx, pending_id, bundle))

    elapsed = time.perf_counter() - start_time

    # Sort results by dataset order
    results.sort(key=lambda x: x[0])

    # Prepare fusion
    if args.fusion == "early":
        fusion_model = EarlyFusion(args.config)
    elif args.fusion == "static":
        fusion_model = StaticLateFusion(args.config)
    else:
        fusion_model = AgenticFusion(args.config)

    # Apply fusion to each bundle
    y_true = []
    y_pred = []
    for (clip_idx, clip_id, bundle) in results:
        pred = fusion_model.run(bundle)
        eng_probs = pred.get("engagement", pred.get("predictions", {}).get("engagement"))
        y_true.append(labels[clip_idx][0])  # primary engagement label
        y_pred.append(int(np.argmax(eng_probs)))

    # Compute metrics
    metrics = summary_dict(y_true, y_pred)
    per_class = per_class_metrics(y_true, y_pred)
    conf = confusion(y_true, y_pred)

    # Write results CSV
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"results_{args.fusion}.csv"
    import pandas as pd
    pd.DataFrame({
        "clip_id": [r[1] for r in results],
        "clip_path": [clip_paths[r[0]] for r in results],
        "pred_engagement": y_pred,
        "true_engagement": y_true,
    }).to_csv(results_path, index=False)

    # Print summary
    print("\n--- Evaluation Summary ---")
    print(f"Fusion method: {args.fusion}")
    print(f"Total clips processed: {len(y_true)}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro‑F1: {metrics['macro_f1']:.4f}")
    print("Per‑class metrics:")
    for cls, vals in per_class.items():
        print(f"  Class {cls}: P={vals['precision']:.3f} R={vals['recall']:.3f} F1={vals['f1']:.3f}")
    print("Confusion matrix (rows= true, cols= pred):")
    print(conf)

def run_all_modalities(clip_idx, clip_path, macro, rppg, mer, disabled):
    """Run the three perception blocks for a single clip.
    Returns (clip_idx, clip_id, [result_dicts...]) where each result_dict follows the common block interface.
    """
    clip_id = str(clip_idx)
    results = []
    # Macro
    if "macro" not in disabled:
        res = macro.run(clip_path)
        res["clip_id"] = clip_id
        results.append(res)
    else:
        results.append({"clip_id": clip_id, "modality": "macro", "status": "absent"})
    # rPPG
    if "rppg" not in disabled:
        res = rppg.run(clip_path)
        res["clip_id"] = clip_id
        results.append(res)
    else:
        results.append({"clip_id": clip_id, "modality": "rppg", "status": "absent"})
    # MER
    if "mer" not in disabled:
        res = mer.run(clip_path)
        res["clip_id"] = clip_id
        results.append(res)
    else:
        results.append({"clip_id": clip_id, "modality": "mer", "status": "absent"})
    return clip_idx, clip_id, results

if __name__ == "__main__":
    main()
