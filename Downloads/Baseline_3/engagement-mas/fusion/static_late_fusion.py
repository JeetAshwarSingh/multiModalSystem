import os
import yaml
import numpy as np
from typing import Dict, Any

class StaticLateFusion:
    """Static Late Fusion: Weighted average of per-modality predictions."""

    def __init__(self, config_path: str = "configs/fusion.yaml"):
        cfg_path = os.path.abspath(config_path)
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
            self.weights = cfg.get("fusion", {}).get("static_late_fusion", {}).get("weights", {
                "macro": 0.5,
                "rppg": 0.3,
                "mer": 0.2
            })
        else:
            self.weights = {"macro": 0.5, "rppg": 0.3, "mer": 0.2}

    def _extract_prob(self, pred: Any) -> np.ndarray:
        if isinstance(pred, dict):
            if "engagement" in pred:
                return np.asarray(pred["engagement"], dtype=np.float32)
            for v in pred.values():
                if isinstance(v, (list, np.ndarray)):
                    return np.asarray(v, dtype=np.float32)
            return np.zeros(4, dtype=np.float32)
        elif isinstance(pred, (list, np.ndarray)):
            return np.asarray(pred, dtype=np.float32)
        return np.zeros(4, dtype=np.float32)

    def run(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        preds = bundle.get("predictions", {})
        availability = bundle.get("availability", {})
        
        weighted_sum = np.zeros(4, dtype=np.float64)
        total_weight = 0.0

        for mod, weight in self.weights.items():
            is_avail = availability.get(mod, mod in preds)
            if is_avail and mod in preds and preds[mod] is not None:
                prob = self._extract_prob(preds[mod])
                if prob.shape[0] == 4 and np.any(prob > 0):
                    weighted_sum += weight * prob
                    total_weight += weight

        if total_weight > 0:
            fused = weighted_sum / total_weight
            sum_fused = np.sum(fused)
            if sum_fused > 0:
                fused = fused / sum_fused
        else:
            fused = np.ones(4, dtype=np.float64) / 4.0

        return {
            "clip_id": bundle.get("clip_id", ""),
            "modality": "fusion",
            "engagement": fused,
            "predictions": {"engagement": fused},
            "status": "success",
        }
