"""train_rppg_classifier.py
Trains a lightweight machine learning classifier on extracted rPPG features
to predict engagement levels (4 classes), addressing Problem 3 & Bonus (class imbalance).
"""

import os
import sys
import argparse
import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
import joblib

# Add project root to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from blocks.rppg.rppg_block import RPPGBlock
from evaluation.evaluate import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train rPPG Engagement Classifier")
    parser.add_argument("--csv", type=str, default="train.csv", help="Training CSV path")
    parser.add_argument("--root-dir", type=str, default="/Users/jeetashwar/Downloads/DAiSEE", help="DAiSEE root dir")
    parser.add_argument("--max-clips", type=int, default=100, help="Max clips to extract features for training")
    parser.add_argument("--output", type=str, default="weights/rppg/classifier.pkl", help="Output path for trained model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Resolve paths
    base_name = os.path.basename(str(args.csv).strip("/\\"))
    clean_rel = str(args.csv).lstrip("/\\")
    baseline_dir = pathlib.Path(__file__).resolve().parent.parent.parent
    csv_candidates = [
        pathlib.Path(args.csv),
        pathlib.Path("..") / base_name,
        pathlib.Path(base_name),
        baseline_dir / base_name,
        pathlib.Path("..") / clean_rel,
        pathlib.Path(clean_rel),
        baseline_dir / clean_rel,
    ]
    csv_path = None
    for cand in csv_candidates:
        if cand.is_file():
            csv_path = cand.resolve()
            break
    if csv_path is None:
        raise FileNotFoundError(f"Could not find training CSV at {args.csv}")

    root_dir = pathlib.Path(args.root_dir).expanduser().resolve()
    print(f"Loading training data from {csv_path} with root {root_dir}...")
    clip_paths, labels = load_dataset(csv_path, root_dir)

    # Subsample if needed, maintaining stratified balance where possible
    df = pd.DataFrame({"path": clip_paths, "label": labels[:, 0]})
    if args.max_clips is not None and args.max_clips < len(df):
        # Sample proportionally or ensure minority classes are included
        sampled = []
        n_per_class = max(1, args.max_clips // 4)
        for c in range(4):
            c_df = df[df["label"] == c]
            if len(c_df) > 0:
                sampled.append(c_df.sample(min(len(c_df), n_per_class), random_state=args.seed))
        df = pd.concat(sampled).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled {len(df)} clips across classes: {df['label'].value_counts().to_dict()}")

    rppg = RPPGBlock(seed=args.seed)
    X, y = [], []

    print(f"Extracting rPPG features from {len(df)} clips...")
    for idx, row in df.iterrows():
        p = row["path"]
        lbl = int(row["label"])
        res = rppg.run(p)
        if res.get("status") == "success":
            hr = float(res["predictions"]["hr"])
            q = res["quality_metrics"]
            bvp = res["features"]
            feat = rppg._extract_feature_vector(hr, q, bvp)
            X.append(feat)
            y.append(lbl)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"Successfully extracted features for {len(X)} clips. Classes: {np.bincount(y, minlength=4)}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest classifier with balanced weights
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight="balanced",
        random_state=args.seed
    )
    clf.fit(X_scaled, y)

    y_pred = clf.predict(X_scaled)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    print(f"\nTraining complete: Accuracy = {acc * 100:.2f}%, Macro-F1 = {f1:.4f}")
    print(classification_report(y, y_pred, zero_division=0))

    # Save
    out_path = pathlib.Path(args.output)
    if not out_path.is_absolute():
        if (pathlib.Path("..") / args.output).parent.is_dir():
            out_path = pathlib.Path("..") / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler}, out_path)
    print(f"Saved trained rPPG classifier to {out_path.resolve()}")


if __name__ == "__main__":
    main()
