"""train_early_fusion.py
Trains the Early Fusion projection head on concatenated multimodal features
using class-weighted cross-entropy loss, addressing Problem 4 & Bonus (class imbalance).
"""

import os
import sys
import argparse
import pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Add project root to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from blocks.macro.macro_block import MacroBlock
from blocks.rppg.rppg_block import RPPGBlock
from blocks.mer.mer_block import MERBlock
from evaluation.evaluate import load_dataset


class TrainableEarlyFusion(nn.Module):
    """Multimodal early fusion classifier network."""
    def __init__(self, input_dim=1152, hidden_dim=256, num_classes=4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


def pad_or_truncate(feat: np.ndarray, target_dim: int) -> np.ndarray:
    flat = np.asarray(feat, dtype=np.float32).flatten()
    if flat.size < target_dim:
        res = np.zeros(target_dim, dtype=np.float32)
        res[:flat.size] = flat
        return res
    return flat[:target_dim]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Early Fusion Head")
    parser.add_argument("--csv", type=str, default="train.csv", help="Training CSV path")
    parser.add_argument("--root-dir", type=str, default=None, help="DAiSEE root dir (auto-detected if omitted)")
    parser.add_argument("--max-clips", type=int, default=None, help="Max clips to extract multimodal features from (default: all clips)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", type=str, default="weights/fusion/early_fusion.pt", help="Output path for weights")
    parser.add_argument("--device", type=str, default="auto", help="Compute device for vision backbone: auto, cuda, or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    # Auto-resolve root_dir across environments
    cfg_root = ""
    ds_yaml = pathlib.Path("configs/datasets.yaml")
    if not ds_yaml.is_file():
        cand_yaml = pathlib.Path(__file__).resolve().parent.parent / "configs" / "datasets.yaml"
        if cand_yaml.is_file():
            ds_yaml = cand_yaml
    if ds_yaml.is_file():
        import yaml
        with open(ds_yaml) as f:
            cfg_root = yaml.safe_load(f).get("datasets", {}).get("daisee", {}).get("root_dir", "")

    root_candidates = [
        args.root_dir,
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
    root_dir = None
    for cand in root_candidates:
        if cand and cand != "/path/to/daisee_root":
            p = pathlib.Path(cand).expanduser()
            if p.is_dir():
                root_dir = p.resolve()
                break
    if root_dir is None:
        root_dir = pathlib.Path(args.root_dir or "/path/to/daisee_root").expanduser().resolve()

    print(f"Loading training data from {csv_path} with root {root_dir}...")
    clip_paths, labels = load_dataset(csv_path, root_dir)

    df = pd.DataFrame({"path": clip_paths, "label": labels[:, 0]})
    if args.max_clips is not None and args.max_clips > 0 and args.max_clips < len(df):
        sampled = []
        n_per_class = max(1, args.max_clips // 4)
        for c in range(4):
            c_df = df[df["label"] == c]
            if len(c_df) > 0:
                sampled.append(c_df.sample(min(len(c_df), n_per_class), random_state=args.seed))
        df = pd.concat(sampled).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled {len(df)} clips across classes: {df['label'].value_counts().to_dict()}")

    macro = MacroBlock(device=args.device)
    rppg = RPPGBlock(seed=args.seed)
    mer = MERBlock(seed=args.seed)

    X_list, y_list = [], []
    dims = {"macro": 768, "rppg": 128, "mer": 256}
    total_dim = sum(dims.values())  # 1152

    print(f"Extracting multi-modal features from {len(df)} clips...")
    for idx, row in df.iterrows():
        p = row["path"]
        lbl = int(row["label"])
        res_m = macro.run(p)
        res_r = rppg.run(p)
        res_mer = mer.run(p)

        feat_m = pad_or_truncate(res_m.get("features", np.zeros(dims["macro"])), dims["macro"])
        feat_r = pad_or_truncate(res_r.get("features", np.zeros(dims["rppg"])), dims["rppg"])
        feat_mer = pad_or_truncate(res_mer.get("features", np.zeros(dims["mer"])), dims["mer"])

        concat_feat = np.concatenate([feat_m, feat_r, feat_mer])
        X_list.append(concat_feat)
        y_list.append(lbl)

    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int64)

    # Compute inverse frequency class weights to resolve class imbalance
    class_counts = np.bincount(y_arr, minlength=4)
    total_samples = len(y_arr)
    weights = []
    for c in range(4):
        cnt = max(1, class_counts[c])
        weights.append(total_samples / (4.0 * cnt))
    dev = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    class_weights = torch.tensor(weights, dtype=torch.float32).to(dev)
    print(f"Computed inverse frequency class weights: {weights} on device {dev}")

    dataset = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = TrainableEarlyFusion(input_dim=total_dim, hidden_dim=256, num_classes=4).to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    print(f"Training Early Fusion model for {args.epochs} epochs on {dev}...")
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch [{epoch+1}/{args.epochs}] Loss: {epoch_loss/total:.4f} Acc: {correct/total*100:.2f}%")

    # Save trained PyTorch model safely
    target_weights = pathlib.Path("weights")
    if target_weights.is_symlink() and not target_weights.exists():
        target_weights.unlink()
    elif target_weights.is_file():
        target_weights.unlink()

    out_path = pathlib.Path(args.output)
    if not out_path.is_absolute():
        if (pathlib.Path("..") / "weights").is_dir():
            out_path = pathlib.Path("..") / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": total_dim,
        "hidden_dim": 256,
        "num_classes": 4,
        "modality_dims": dims,
    }, out_path)
    print(f"Saved trained Early Fusion checkpoint to {out_path.resolve()}")


if __name__ == "__main__":
    main()
