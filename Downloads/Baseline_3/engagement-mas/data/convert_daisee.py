import os
import argparse
import pathlib
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Convert DAiSEE dataset labels to unified format")
    parser.add_argument("--source-dir", type=str, required=True, help="Directory containing original DAiSEE labels/files")
    parser.add_argument("--output-dir", type=str, required=True, help="Target root directory where train.csv, val.csv, test.csv will be created")
    return parser.parse_args()

def find_label_file(source_dir: pathlib.Path, split: str):
    alias_map = {
        "train": ["train", "training"],
        "val": ["validation", "val", "valid"],
        "test": ["test", "testing"],
    }
    aliases = alias_map.get(split.lower(), [split.lower()])

    candidates = []
    for p in source_dir.rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix not in [".csv", ".txt", ".mat"]:
            continue
        stem_lower = p.stem.lower()
        for alias in aliases:
            if alias in stem_lower:
                # Prefer files that explicitly contain "label" or equal the alias
                priority = 0
                if "label" in stem_lower:
                    priority += 2
                if stem_lower.startswith(alias):
                    priority += 1
                candidates.append((priority, p))
                break

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    return None

def build_video_index(search_dirs: list, base_dir: pathlib.Path):
    """Index video files by filename stem for instant lookup."""
    index = {}
    valid_exts = {".mp4", ".avi", ".mov", ".mkv"}
    print("Indexing video files for fast matching...")
    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for p in s_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in valid_exts:
                stem = p.stem.lower()
                if stem not in index:
                    try:
                        # Path relative to base_dir
                        rel = str(p.relative_to(base_dir))
                    except ValueError:
                        # If outside base_dir, relative to s_dir
                        rel = str(p.relative_to(s_dir))
                    index[stem] = rel
    print(f"Indexed {len(index)} video files.")
    return index

def process_split(label_file: pathlib.Path, output_dir: pathlib.Path, split: str, video_index: dict):
    print(f"\nProcessing split '{split}' from {label_file}...")
    sep = "," if label_file.suffix.lower() == ".csv" else None
    try:
        df = pd.read_csv(label_file, sep=sep, engine="python")
    except Exception as e:
        print(f"Error reading {label_file}: {e}")
        return

    df.columns = [c.strip() for c in df.columns]

    clip_col = None
    for col in df.columns:
        if any(k in col.lower() for k in ["clip", "video", "id", "name"]):
            clip_col = col
            break
    if clip_col is None:
        clip_col = df.columns[0]

    target_labels = ["Engagement", "Boredom", "Confusion", "Frustration"]
    col_map = {}
    for req in target_labels:
        for c in df.columns:
            if c.lower() == req.lower():
                col_map[c] = req
                break

    records = []
    matched = 0
    for _, row in df.iterrows():
        raw_clip = str(row[clip_col]).strip()
        stem = pathlib.Path(raw_clip).stem.lower()

        if stem in video_index:
            rel_path = video_index[stem]
            matched += 1
        else:
            # Fallback path if videos are not yet indexed
            ext = pathlib.Path(raw_clip).suffix or ".avi"
            clean_name = f"{stem}{ext}"
            rel_path = f"DataSet/{split.capitalize()}/{clean_name}"

        record = {"clip_path": rel_path}
        for orig_col, target_col in col_map.items():
            record[target_col] = int(row[orig_col])
        records.append(record)

    out_df = pd.DataFrame(records)
    out_csv = output_dir / f"{split}.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Saved {len(out_df)} records to {out_csv} ({matched}/{len(out_df)} matched to local video files).")

def main():
    args = parse_args()
    source_dir = pathlib.Path(args.source_dir).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search for videos in source_dir and output_dir, using source_dir as the base root
    video_index = build_video_index([source_dir, output_dir], base_dir=source_dir)

    splits = ["train", "val", "test"]
    for split in splits:
        lf = find_label_file(source_dir, split)
        if lf:
            process_split(lf, output_dir, split, video_index)
        else:
            print(f"Warning: No label file found for split '{split}' in {source_dir}")

if __name__ == "__main__":
    main()
