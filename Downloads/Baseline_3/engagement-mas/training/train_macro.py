# Training script for Macro block (PyTorch Lightning)

"""train_macro.py

Usage (local Mac – CPU/MPS):
    conda activate engagement-mas-mac
    python -m training.train_macro --config configs/training.yaml --dataset daisee --split train

Usage (cloud – CUDA):
    conda activate engagement-mas-cloud
    python -m training.train_macro --config configs/training.yaml --dataset daisee --split train

The script:
* Loads the CSV for the chosen dataset (via `configs/datasets.yaml`).
* Builds a `torch.utils.data.Dataset` that reads video clips, runs the **MacroBlock** in *feature‑extraction mode* (appearance backbone frozen) and returns the per‑clip embedding plus label vector.
* Caches the appearance embeddings on disk (`cache_dir`) to avoid recomputing them every epoch.
* Trains the temporal transformer and classification heads (including optional LoRA) using **PyTorch Lightning**.
* Saves checkpoints (`best.pt`, `last.pt`) under `weights/macro/` and writes a copy of the training config (`config.yaml`).
"""

import argparse
import os
import yaml
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from blocks.macro.macro_block import MacroBlock

class MacroDataset(Dataset):
    """Dataset that returns cached appearance embeddings and labels.
    On first epoch it extracts embeddings via MacroBlock (appearance path) and stores them.
    """
    def __init__(self, csv_path: str, label_cols: list, root_dir: str, cache_dir: str, device: str = "cpu"):
        candidates = [
            csv_path,
            os.path.join(root_dir, csv_path),
            os.path.join("..", os.path.basename(csv_path)),
            os.path.join(root_dir, os.path.basename(csv_path)),
            os.path.join("data", "daisee", os.path.basename(csv_path))
        ]
        csv_file = None
        for cand in candidates:
            if os.path.isfile(cand):
                csv_file = cand
                break
        if csv_file is None:
            raise FileNotFoundError(f"Could not find CSV file '{csv_path}'. Checked candidates: {candidates}")
        self.df = pd.read_csv(csv_file)
        self.label_cols = label_cols
        self.root = root_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.device = device
        self.block = MacroBlock(use_lora=False, device=device)
        # Pre‑compute deterministic hash for each clip for caching
        self.clip_ids = [self.block._clip_id_from_path(os.path.join(self.root, p)) for p in self.df['clip_path']]

    def __len__(self):
        return len(self.df)

    def _cache_path(self, clip_id: str) -> str:
        return os.path.join(self.cache_dir, f"{clip_id}.pt")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row['clip_path']
        clip_path = os.path.join(self.root, rel_path)
        clip_id = self.clip_ids[idx]
        cache_path = self._cache_path(clip_id)
        if os.path.isfile(cache_path):
            embedding = torch.load(cache_path)
        else:
            result = self.block.run(clip_path)
            if isinstance(result, dict) and "features" in result and result["features"] is not None:
                embedding = torch.from_numpy(result['features']).float()
            else:
                embedding = torch.zeros(832, dtype=torch.float32)
            torch.save(embedding, cache_path)
        # Labels as integer tensor
        labels = torch.tensor(row[self.label_cols].values.astype(np.int64))
        return embedding, labels

class MacroLitModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):
            self.save_hyperparameters(hparams)
        else:
            self.hparams.update(hparams)
        lora_cfg = self.hparams.get('lora', {})
        lora_enabled = lora_cfg.get('enabled', False) if isinstance(lora_cfg, dict) else getattr(lora_cfg, 'enabled', False)
        lora_rank = lora_cfg.get('rank', 4) if isinstance(lora_cfg, dict) else getattr(lora_cfg, 'rank', 4)
        dev = self.hparams.get('device', 'cpu')
        
        self.block = MacroBlock(use_lora=lora_enabled, lora_rank=lora_rank, device=dev)
        self.transformer = self.block.transformer
        self.head_engagement = self.block.head_engagement
        self.head_boredom = self.block.head_boredom
        self.head_confusion = self.block.head_confusion
        self.head_frustration = self.block.head_frustration
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embedding):
        seq = embedding.unsqueeze(1)
        transformed = self.transformer(seq)
        pooled = transformed.squeeze(1)
        return pooled

    def training_step(self, batch, batch_idx):
        embedding, labels = batch
        feats = self.forward(embedding)
        loss = 0.0
        logits_eng = self.head_engagement(feats)
        loss += float(self.hparams.get('loss_weights', {}).get('engagement', 1.0)) * self.loss_fn(logits_eng, labels[:, 0])
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr = float(self.hparams.get('learning_rate', 1e-3))
        wd = float(self.hparams.get('weight_decay', 1e-5))
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training.yaml', help='Path to training.yaml')
    parser.add_argument('--dataset', type=str, default='daisee', help='Dataset key from configs/datasets.yaml')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)['training']
    with open('configs/datasets.yaml') as f:
        ds_cfg = yaml.safe_load(f)['datasets'][args.dataset]

    seed_everything(42)
    ckpt_dir = os.path.abspath(cfg.get('checkpoint_dir', 'weights/macro'))
    cache_dir = os.path.abspath(cfg.get('cache_dir', 'data/cache'))
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    device_setting = cfg.get("device", "auto")
    if device_setting == "auto" or device_setting == "cuda" or device_setting == "gpu":
        if torch.cuda.is_available():
            accel = "gpu"
            dev = "cuda"
        elif torch.backends.mps.is_available():
            accel = "mps"
            dev = "mps"
        else:
            accel = "cpu"
            dev = "cpu"
    else:
        accel = device_setting
        dev = device_setting

    dataset = MacroDataset(csv_path=ds_cfg[f"{args.split}_csv"],
                           label_cols=ds_cfg['label_columns'],
                           root_dir=ds_cfg['root_dir'],
                           cache_dir=cache_dir,
                           device=dev)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)

    lit_module = MacroLitModule(cfg)

    checkpoint_cb = ModelCheckpoint(dirpath=ckpt_dir, filename='best', monitor='train_loss', mode='min', save_top_k=1)
    trainer = Trainer(max_epochs=cfg['num_epochs'],
                      default_root_dir=ckpt_dir,
                      callbacks=[checkpoint_cb],
                      accelerator=accel,
                      devices=1 if accel in ["gpu", "mps"] else "auto")
    trainer.fit(lit_module, loader)

    final_path = os.path.join(ckpt_dir, 'last.pt')
    torch.save(lit_module.state_dict(), final_path)
    print(f"Saved final trained weights to {final_path}")
    yaml.safe_dump(cfg, open(os.path.join(ckpt_dir, 'config.yaml'), 'w'))

if __name__ == "__main__":
    main()
