# Data Interface

This directory contains utilities for loading datasets in a **dataset‑agnostic** way. The system expects a root folder that contains video clips and three CSV files (`train.csv`, `val.csv`, `test.csv`).

## CSV Schema
| Column      | Description |
|------------|-------------|
| `clip_path` | Path **relative** to the dataset root that points to a video file (e.g., `videos/clip_001.mp4`). |
| `<label>`   | One column per label defined in the dataset configuration. For DAiSEE the default columns are `Engagement`, `Boredom`, `Confusion`, `Frustration`. Each entry is an integer in the range `[0, 3]` (four severity levels). |

Only the columns listed in the configuration file are required; extra columns are ignored.

## Configuration (`configs/datasets.yaml`)
The dataset configuration lives in `configs/datasets.yaml`. Example:
```yaml
datasets:
  daisee:
    root_dir: /path/to/daisee_root
    train_csv: train.csv
    val_csv: val.csv
    test_csv: test.csv
    label_columns: [Engagement, Boredom, Confusion, Frustration]
    num_classes: 4  # number of classes per label (0‑3)
```
* `root_dir` – absolute or relative path to the folder that contains the video clips.
* `train_csv`, `val_csv`, `test_csv` – file names (relative to `root_dir`).
* `label_columns` – list of label column names.
* `num_classes` – number of discrete classes for each label (default `4`).

Any new dataset only needs a new entry here and matching CSV files; **no code changes** are required.

## Conversion Script – `convert_daisee.py`
The official DAiSEE release provides separate label files (MAT/CSV). `convert_daisee.py` converts them into the unified CSV format.
```bash
python -m data.convert_daisee \
    --source-dir /path/to/daisee_original \
    --output-dir /path/to/daisee_root
```
Arguments:
* `--source-dir` – directory containing the original DAiSEE label files (`train_label.txt`, etc.).
* `--output-dir` – target root directory where the video clips reside. The script creates `train.csv`, `val.csv`, `test.csv` inside this directory.

The script parses the original files, builds the required columns, and writes the CSVs using **pandas**.

## Validation Script – `validate_dataset.py`
Run this script to sanity‑check a dataset configuration:
```bash
python -m data.validate_dataset \
    --config daisee   # name of the dataset entry in configs/datasets.yaml
```
The validator:
1. Loads the CSV specified for the chosen split (by default it checks all three).  
2. Verifies that each `clip_path` resolves to an existing, readable file under `root_dir`.  
3. Reports class distribution for each label (counts per class).  
4. Exits with a non‑zero status if any missing/ unreadable clip is found.

The script is deterministic and can be used in CI pipelines.

## Usage in Code
A typical loader (e.g., in `data/loader.py`) does:
```python
import yaml, pandas as pd, pathlib
cfg = yaml.safe_load(open('configs/datasets.yaml'))['datasets'][args.dataset]
root = pathlib.Path(cfg['root_dir'])
train_df = pd.read_csv(root / cfg['train_csv'])
# access clips & labels via `train_df['clip_path']` and the label columns
```
All perception blocks expect a **clip identifier** (`clip_id`) that matches the CSV entry, and they receive a shared timestamp index derived from the video frames.

---

**Note:** The conversion script assumes the original DAiSEE label files follow the standard format used in the literature. If your copy differs, adjust the parsing logic in `convert_daisee.py` accordingly.
