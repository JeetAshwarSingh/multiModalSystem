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

try:
    from evaluation.analyze_and_plot import compute_detailed_metrics, generate_all_plots, generate_latex_table
except ImportError:
    try:
        from analyze_and_plot import compute_detailed_metrics, generate_all_plots, generate_latex_table
    except ImportError:
        compute_detailed_metrics = None
        generate_all_plots = None
        generate_latex_table = None

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation for Engagement‑MAS")
    parser.add_argument("--config", type=str, required=True, help="Path to fusion config YAML")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (matches entry in configs/datasets.yaml)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (e.g. test, val, train, test_quick)")
    parser.add_argument("--fusion", type=str, choices=["early", "static", "agentic"], default="agentic", help="Primary fusion strategy")
    parser.add_argument("--disable-modality", type=str, nargs="*", default=[], help="Modality names to mask (macro, rppg, mer)")
    parser.add_argument("--max-clips", type=int, default=None, help="Optional max number of clips to evaluate")
    parser.add_argument("--output-dir", type=str, default="resultAndAnalysis", help="Directory to save predictions, metrics, and analysis charts")
    return parser.parse_args()

def load_dataset(csv_path: pathlib.Path, root_dir: pathlib.Path = None):
    import pandas as pd
    df = pd.read_csv(csv_path)
    paths = []
    for p in df["clip_path"]:
        p_str = str(p)
        if root_dir is not None and not os.path.isabs(p_str):
            target = root_dir / p_str
            if not target.is_file():
                if p_str.startswith("DataSet/") and (root_dir / p_str[8:]).is_file():
                    target = root_dir / p_str[8:]
                elif (root_dir / "DataSet" / p_str).is_file():
                    target = root_dir / "DataSet" / p_str
            paths.append(str(target))
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
                root_candidates = [
                    os.environ.get("DAISEE_ROOT", ""),
                    cfg_root,
                    "/mnt/c/Users/puneet/Downloads/DAiSEE",
                    "C:/Users/puneet/Downloads/DAiSEE",
                    "/Users/jeetashwar/Downloads/DAiSEE",
                    "../DAiSEE",
                    "DAiSEE",
                    "../DataSet",
                    "DataSet",
                ]
                for cand in root_candidates:
                    if cand and cand != "/path/to/daisee_root":
                        p = pathlib.Path(cand).expanduser()
                        if p.is_dir():
                            root_dir = p.resolve()
                            break
                if root_dir is None and cfg_root and cfg_root != "/path/to/daisee_root":
                    root_dir = pathlib.Path(cfg_root).expanduser().resolve()
                csv_key = f"{args.split}_csv"
                csv_name = str(ds_cfg.get(csv_key, f"{args.split}.csv"))
                candidates = [
                    pathlib.Path(csv_name),
                    pathlib.Path("..") / os.path.basename(csv_name),
                    pathlib.Path(os.path.basename(csv_name)),
                    pathlib.Path("..") / csv_name,
                    pathlib.Path("data") / args.dataset / os.path.basename(csv_name),
                ]
                if root_dir is not None:
                    candidates.extend([root_dir / csv_name, root_dir / os.path.basename(csv_name)])
                for cand in candidates:
                    if cand.is_file():
                        csv_path = cand.resolve()
                        break

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

    results.sort(key=lambda x: x[0])

    # Resolve output directory (resultAndAnalysis)
    out_dir = pathlib.Path(args.output_dir)
    if not out_dir.is_absolute():
        if (pathlib.Path("..") / args.output_dir).is_dir():
            out_dir = (pathlib.Path("..") / args.output_dir).resolve()
        elif (pathlib.Path(".") / args.output_dir).is_dir():
            out_dir = (pathlib.Path(".") / args.output_dir).resolve()
        elif (pathlib.Path("..") / "resultAndAnalysis").is_dir():
            out_dir = (pathlib.Path("..") / "resultAndAnalysis").resolve()
        elif pathlib.Path("configs").is_dir():
            out_dir = (pathlib.Path("..") / args.output_dir).resolve()
        else:
            out_dir = pathlib.Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = out_dir / "Analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Prepare fusion models
    early_model = EarlyFusion(args.config)
    static_model = StaticLateFusion(args.config)
    agentic_model = AgenticFusion(args.config)

    methods = [
        "Macro Modality",
        "rPPG Modality",
        "MER Modality",
        "Early Fusion",
        "Static Late Fusion",
        "Agentic Late Fusion",
    ]
    preds_dict = {m: [] for m in methods}
    probs_dict = {m: [] for m in methods}
    y_true = []
    agentic_weights = {"macro": [], "rppg": [], "mer": []}

    # Evaluate each modality and fusion across all bundles
    for (clip_idx, clip_id, bundle) in results:
        target = int(labels[clip_idx][0])
        y_true.append(target)
        p_bundle = bundle.get("predictions", {})

        # 1. Macro Modality
        m_p = np.zeros(4, dtype=np.float32)
        if "macro" in p_bundle and isinstance(p_bundle["macro"], dict):
            eng = p_bundle["macro"].get("engagement")
            if eng is not None:
                m_p = np.asarray(eng, dtype=np.float32)
        preds_dict["Macro Modality"].append(int(np.argmax(m_p)) if np.any(m_p > 0) else 0)
        probs_dict["Macro Modality"].append(m_p)

        # 2. rPPG Modality
        r_p = np.zeros(4, dtype=np.float32)
        if "rppg" in p_bundle and isinstance(p_bundle["rppg"], dict):
            eng = p_bundle["rppg"].get("engagement")
            if eng is not None:
                r_p = np.asarray(eng, dtype=np.float32)
        preds_dict["rPPG Modality"].append(int(np.argmax(r_p)) if np.any(r_p > 0) else 0)
        probs_dict["rPPG Modality"].append(r_p)

        # 3. MER Modality
        mer_p = np.zeros(4, dtype=np.float32)
        if "mer" in p_bundle and isinstance(p_bundle["mer"], dict):
            eng = p_bundle["mer"].get("engagement")
            if eng is not None:
                mer_p = np.asarray(eng, dtype=np.float32)
        preds_dict["MER Modality"].append(int(np.argmax(mer_p)) if np.any(mer_p > 0) else 0)
        probs_dict["MER Modality"].append(mer_p)

        # 4. Early Fusion
        ef_res = early_model.run(bundle)
        ef_p = np.asarray(ef_res.get("engagement", np.zeros(4)), dtype=np.float32)
        preds_dict["Early Fusion"].append(int(np.argmax(ef_p)) if np.any(ef_p > 0) else 0)
        probs_dict["Early Fusion"].append(ef_p)

        # 5. Static Late Fusion
        sf_res = static_model.run(bundle)
        sf_p = np.asarray(sf_res.get("engagement", np.zeros(4)), dtype=np.float32)
        preds_dict["Static Late Fusion"].append(int(np.argmax(sf_p)) if np.any(sf_p > 0) else 0)
        probs_dict["Static Late Fusion"].append(sf_p)

        # 6. Agentic Late Fusion
        af_res = agentic_model.run(bundle)
        af_p = np.asarray(af_res.get("engagement", np.zeros(4)), dtype=np.float32)
        preds_dict["Agentic Late Fusion"].append(int(np.argmax(af_p)) if np.any(af_p > 0) else 0)
        probs_dict["Agentic Late Fusion"].append(af_p)

        # Log agentic quality weights
        for k in ["macro", "rppg", "mer"]:
            agentic_weights[k].append(float(agentic_model.base_weights.get(k, 0.33)))

    import pandas as pd
    file_prefix_map = {
        "Macro Modality": "macro",
        "rPPG Modality": "rppg",
        "MER Modality": "mer",
        "Early Fusion": "fusion_early",
        "Static Late Fusion": "fusion_static",
        "Agentic Late Fusion": "fusion_agentic",
    }

    # Save detailed prediction CSVs for each modality and fusion
    for m in methods:
        p_list = preds_dict[m]
        prob_arr = np.array(probs_dict[m])
        if prob_arr.ndim != 2 or prob_arr.shape[1] < 4:
            prob_arr = np.zeros((len(p_list), 4), dtype=np.float32)
        df_out = pd.DataFrame({
            "clip_id": [r[1] for r in results],
            "clip_path": [clip_paths[r[0]] for r in results],
            "pred_engagement": p_list,
            "true_engagement": y_true,
            "prob_class_0": prob_arr[:, 0],
            "prob_class_1": prob_arr[:, 1],
            "prob_class_2": prob_arr[:, 2],
            "prob_class_3": prob_arr[:, 3],
        })
        prefix = file_prefix_map.get(m, m.lower().replace(" ", "_"))
        csv_file = out_dir / f"{prefix}_predictions.csv"
        df_out.to_csv(csv_file, index=False)
        print(f"Saved predictions CSV: {csv_file}")

    # Backwards compatibility: results_<fusion>.csv
    primary_prefix = f"fusion_{args.fusion}" if args.fusion in ["early", "static", "agentic"] else args.fusion
    if (out_dir / f"{primary_prefix}_predictions.csv").is_file():
        import shutil
        shutil.copy(out_dir / f"{primary_prefix}_predictions.csv", out_dir / f"results_{args.fusion}.csv")

    # Compute comprehensive metrics
    summary_records = []
    per_class_all = {}
    cm_all = {}
    json_metrics = {}

    for m in methods:
        p_list = preds_dict[m]
        if compute_detailed_metrics is not None:
            met = compute_detailed_metrics(y_true, p_list)
        else:
            s_dict = summary_dict(y_true, p_list)
            met = {
                "accuracy": s_dict["accuracy"],
                "macro_f1": s_dict["macro_f1"],
                "weighted_f1": s_dict["macro_f1"],
                "precision": 0.0,
                "recall": 0.0,
                "confusion_matrix": confusion(y_true, p_list).tolist(),
                "per_class": {}
            }
        json_metrics[m] = met
        per_class_all[m] = met.get("per_class", {})
        cm_all[m] = np.array(met["confusion_matrix"])

        summary_records.append({
            "Method": m,
            "Accuracy": met["accuracy"],
            "Macro-F1": met["macro_f1"],
            "Precision": met.get("precision", 0.0),
            "Recall": met.get("recall", 0.0),
        })

    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = out_dir / "overall_metrics_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved overall metrics summary: {summary_csv_path}")

    json_path = out_dir / "metrics_summary.json"
    with open(json_path, "w") as f:
        json.dump(json_metrics, f, indent=2)
    print(f"Saved detailed metrics JSON: {json_path}")

    # Generate publication-ready figures & LaTeX table in Analysis/
    if generate_all_plots is not None:
        try:
            generate_all_plots(summary_df, per_class_all, cm_all, agentic_weights, analysis_dir)
        except Exception as e:
            print(f"Plot generation notice: {e}")
    if generate_latex_table is not None:
        try:
            generate_latex_table(summary_df, analysis_dir / "paper_summary_table.tex")
        except Exception as e:
            pass

    # Print summary table to console
    print("\n" + "=" * 70)
    print("           ENGAGEMENT-MAS COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total clips processed: {len(y_true)} | Time elapsed: {elapsed:.2f}s")
    print("-" * 70)
    print(f"{'Method':<22} | {'Accuracy':<10} | {'Macro-F1':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 70)
    for _, r in summary_df.iterrows():
        print(f"{r['Method']:<22} | {r['Accuracy']*100:>8.2f}% | {r['Macro-F1']:>10.4f} | {r['Precision']:>10.4f} | {r['Recall']:>10.4f}")
    print("=" * 70)
    print(f"All predictions, metrics, and charts saved to: {out_dir}")

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
