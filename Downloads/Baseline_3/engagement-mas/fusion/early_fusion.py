import os
import yaml
import numpy as np
from typing import Dict, Any

class EarlyFusion:
    """Early Fusion: Concatenates feature vectors from present modalities and maps to 4 classes."""

    def __init__(self, config_path: str = "configs/fusion.yaml"):
        cfg_path = os.path.abspath(config_path)
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
            self.cfg = cfg.get("fusion", {}).get("early_fusion", {})
        else:
            self.cfg = {}

        # Modality expected feature dimensions for deterministic linear mapping
        self.modality_dims = {
            "macro": 768,
            "rppg": 128,
            "mer": 256,
        }
        total_dim = sum(self.modality_dims.values())
        # Fixed deterministic pseudo-random projection matrix
        rng = np.random.RandomState(42)
        self.proj_weights = {
            mod: rng.randn(dim, 4) * 0.05 for mod, dim in self.modality_dims.items()
        }
        self.bias = np.zeros(4, dtype=np.float64)

    def run(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        features = bundle.get("features", {})
        availability = bundle.get("availability", {})

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
