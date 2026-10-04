import os
import sys
import yaml
import argparse
import pathlib
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Validate dataset configuration and clips")
    parser.add_argument("--config", type=str, default="daisee", help="Dataset key in configs/datasets.yaml")
    parser.add_argument("--config-file", type=str, default="configs/datasets.yaml", help="Path to datasets.yaml")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"])
    return parser.parse_args()

def validate_split(root_dir: pathlib.Path, csv_file: str, label_cols: list, num_classes: int):
    csv_p = pathlib.Path(csv_file)
    if csv_p.is_absolute() and csv_p.is_file():
        csv_path = csv_p
    elif (root_dir / csv_file).is_file():
        csv_path = root_dir / csv_file
    elif csv_p.is_file():
        csv_path = csv_p
    elif (pathlib.Path("..") / csv_file).is_file():
        csv_path = pathlib.Path("..") / csv_file
    else:
        csv_path = root_dir / csv_file

    if not csv_path.is_file():
        print(f"❌ Error: CSV file does not exist: {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading CSV {csv_path}: {e}")
        return False

    if "clip_path" not in df.columns:
        print(f"❌ Error: 'clip_path' column missing in {csv_path}")
        return False

    missing_labels = [c for c in label_cols if c not in df.columns]
    if missing_labels:
        print(f"❌ Error: Missing label columns in {csv_path}: {missing_labels}")
        return False

    print(f"\n--- Checking {csv_path.name} ({len(df)} records) ---")
    missing_clips = 0
    for idx, rel_p in enumerate(df["clip_path"]):
        full_p = root_dir / str(rel_p)
        if not full_p.is_file():
            if missing_clips < 5:
                print(f"  Missing clip [{idx}]: {full_p}")
            missing_clips += 1

    if missing_clips > 0:
        print(f"❌ Total missing/unreadable clips: {missing_clips}/{len(df)}")
        return False

    print("✅ All video clips verified!")
    print("Class distributions:")
    for col in label_cols:
        counts = df[col].value_counts().to_dict()
        print(f"  {col}: {counts}")

    return True

def main():
    args = parse_args()
    cfg_file = pathlib.Path(args.config_file)
    if not cfg_file.is_file():
        print(f"Error: Config file not found: {cfg_file}")
        sys.exit(1)

    with open(cfg_file, "r") as f:
        datasets_cfg = yaml.safe_load(f).get("datasets", {})

    if args.config not in datasets_cfg:
        print(f"Error: Dataset '{args.config}' not defined in {cfg_file}")
        sys.exit(1)

    cfg = datasets_cfg[args.config]
    root_dir = pathlib.Path(cfg.get("root_dir", "")).expanduser().resolve()
    label_cols = cfg.get("label_columns", ["Engagement", "Boredom", "Confusion", "Frustration"])
    num_classes = cfg.get("num_classes", 4)

    all_passed = True
    for split in args.splits:
        csv_key = f"{split}_csv"
        if csv_key in cfg:
            csv_file = cfg[csv_key]
            ok = validate_split(root_dir, csv_file, label_cols, num_classes)
            if not ok:
                all_passed = False

    if not all_passed:
        print("\n❌ Dataset validation FAILED.")
        sys.exit(1)
    else:
        print("\n🎉 Dataset validation PASSED!")

if __name__ == "__main__":
    main()
