import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class TrainableEarlyFusionModule(nn.Module):
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


class EarlyFusion:
    """Early Fusion: Concatenates feature vectors from present modalities and maps to 4 classes."""

    def __init__(self, config_path: str = "configs/fusion.yaml", weights_path: Optional[str] = None):
        cfg_path = os.path.abspath(config_path)
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
            self.cfg = cfg.get("fusion", {}).get("early_fusion", {})
        else:
            self.cfg = {}

        self.modality_dims = {
            "macro": 768,
            "rppg": 128,
            "mer": 256,
        }
        total_dim = sum(self.modality_dims.values())  # 1152

        # Check for trained PyTorch model
        self.model = None
        candidates = [
            weights_path,
            os.path.abspath("weights/fusion/early_fusion.pt"),
            os.path.abspath("../weights/fusion/early_fusion.pt"),
        ]
        for p in candidates:
            if p and os.path.isfile(p):
                try:
                    ckpt = torch.load(p, map_location="cpu")
                    in_dim = ckpt.get("input_dim", total_dim)
                    h_dim = ckpt.get("hidden_dim", 256)
                    n_cls = ckpt.get("num_classes", 4)
                    m = TrainableEarlyFusionModule(input_dim=in_dim, hidden_dim=h_dim, num_classes=n_cls)
                    m.load_state_dict(ckpt["state_dict"])
                    m.eval()
                    self.model = m
                    print(f"Loaded trained Early Fusion model from {p}")
                    break
                except Exception as e:
                    print(f"Warning: Failed to load Early Fusion model from {p}: {e}")

        # Deterministic fallback projection matrix if no trained model
        rng = np.random.RandomState(42)
        self.proj_weights = {
            mod: rng.randn(dim, 4) * (1.0 / np.sqrt(dim)) for mod, dim in self.modality_dims.items()
        }
        self.bias = np.zeros(4, dtype=np.float64)

    def run(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        features = bundle.get("features", {})
        availability = bundle.get("availability", {})

        # If trained PyTorch model is loaded, use it
        if self.model is not None:
            feat_parts = []
            has_feat = False
            for mod, expected_dim in self.modality_dims.items():
                is_avail = availability.get(mod, mod in features)
                if is_avail and mod in features and features[mod] is not None:
                    feat = np.asarray(features[mod], dtype=np.float32).flatten()
                    if feat.size < expected_dim:
                        padded = np.zeros(expected_dim, dtype=np.float32)
                        padded[:feat.size] = feat
                        feat = padded
                    elif feat.size > expected_dim:
                        feat = feat[:expected_dim]
                    feat_parts.append(feat)
                    has_feat = True
                else:
                    feat_parts.append(np.zeros(expected_dim, dtype=np.float32))

            if has_feat:
                concat_vec = np.concatenate(feat_parts).astype(np.float32)
                with torch.no_grad():
                    t_in = torch.from_numpy(concat_vec).unsqueeze(0)
                    out_logits = self.model(t_in).squeeze(0).numpy()
                exp_l = np.exp(out_logits - np.max(out_logits))
                probs = exp_l / (np.sum(exp_l) + 1e-12)
                return {
                    "clip_id": bundle.get("clip_id", ""),
                    "modality": "fusion",
                    "engagement": probs.astype(np.float32),
                    "predictions": {"engagement": probs.astype(np.float32)},
                    "status": "success",
                }

        logits = np.copy(self.bias)
        has_feature = False

        for mod, expected_dim in self.modality_dims.items():
            is_avail = availability.get(mod, mod in features)
            if is_avail and mod in features and features[mod] is not None:
                feat = np.asarray(features[mod], dtype=np.float64).flatten()
                if feat.size > 0:
                    # Pad or truncate if needed to match expected_dim
                    if feat.size < expected_dim:
                        padded = np.zeros(expected_dim, dtype=np.float64)
                        padded[:feat.size] = feat
                        feat = padded
                    elif feat.size > expected_dim:
                        feat = feat[:expected_dim]
                    logits += feat @ self.proj_weights[mod]
                    has_feature = True

        if not has_feature:
            # Fallback to predictions if features missing
            preds = bundle.get("predictions", {})
            for mod in ["macro", "rppg", "mer"]:
                if mod in preds and preds[mod] is not None:
                    p = preds[mod]
                    if isinstance(p, dict) and "engagement" in p:
                        return {
                            "clip_id": bundle.get("clip_id", ""),
                            "modality": "fusion",
                            "engagement": np.asarray(p["engagement"], dtype=np.float64),
                            "predictions": {"engagement": np.asarray(p["engagement"], dtype=np.float64)},
                            "status": "success",
                        }

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-12)

        return {
            "clip_id": bundle.get("clip_id", ""),
            "modality": "fusion",
            "engagement": probs,
            "predictions": {"engagement": probs},
            "status": "success",
        }
