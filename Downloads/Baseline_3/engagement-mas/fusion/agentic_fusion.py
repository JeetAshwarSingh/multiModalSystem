import os
import yaml
import collections
import numpy as np
from typing import Dict, Any

class AgenticFusion:
    """Agentic Late Fusion:
    - Modality Monitor: Evaluates reliability from quality metrics.
    - Reasoning Engine: Adjusts weights based on deterministic rule thresholds.
    - State Controller: Buffers recent confidences for temporal smoothing.
    """

    def __init__(self, config_path: str = "configs/fusion.yaml"):
        cfg_path = os.path.abspath(config_path)
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
            self.cfg = cfg.get("fusion", {}).get("agentic_fusion", {})
        else:
            self.cfg = {}

        self.thresholds = self.cfg.get("reliability_thresholds", {
            "rppg_snr": 5.0,
            "mer_flow_noise": 0.2,
            "macro_blur": 100.0,
        })
        self.adj = self.cfg.get("weight_adjustment", {
            "increase": 0.2,
            "decrease": 0.2,
        })
        self.max_weight = self.cfg.get("max_weight", 0.8)
        self.min_weight = self.cfg.get("min_weight", 0.1)
        self.smoothing_window = self.cfg.get("smoothing_window", 5)

        self.base_weights = {"macro": 0.5, "rppg": 0.3, "mer": 0.2}
        self.history = collections.deque(maxlen=self.smoothing_window)

    def _extract_prob(self, pred: Any) -> np.ndarray:
        if isinstance(pred, dict):
            if "engagement" in pred:
                return np.asarray(pred["engagement"], dtype=np.float64)
            for v in pred.values():
                if isinstance(v, (list, np.ndarray)):
                    return np.asarray(v, dtype=np.float64)
            return np.zeros(4, dtype=np.float64)
        elif isinstance(pred, (list, np.ndarray)):
            return np.asarray(pred, dtype=np.float64)
        return np.zeros(4, dtype=np.float64)

    def run(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        preds = bundle.get("predictions", {})
        quality = bundle.get("quality_metrics", {})
        availability = bundle.get("availability", {})

        weights = dict(self.base_weights)
        explanations = []

        # Modality Monitor & Reasoning Engine
        # Macro check
        if availability.get("macro", "macro" in preds):
            blur = quality.get("macro", {}).get("blur", 100.0)
            if blur >= self.thresholds.get("macro_blur", 100.0):
                weights["macro"] = min(self.max_weight, weights["macro"] + self.adj.get("increase", 0.2))
                explanations.append(f"Macro blur ({blur:.1f}) good -> increased weight to {weights['macro']:.2f}")
            else:
                weights["macro"] = max(self.min_weight, weights["macro"] - self.adj.get("decrease", 0.2))
                explanations.append(f"Macro blur ({blur:.1f}) poor -> decreased weight to {weights['macro']:.2f}")
        else:
            weights["macro"] = 0.0

        # rPPG check
        if availability.get("rppg", "rppg" in preds):
            snr = quality.get("rppg", {}).get("snr", 10.0)
            if snr >= self.thresholds.get("rppg_snr", 5.0):
                weights["rppg"] = min(self.max_weight, weights["rppg"] + self.adj.get("increase", 0.2))
                explanations.append(f"rPPG SNR ({snr:.1f}dB) high -> increased weight to {weights['rppg']:.2f}")
            else:
                weights["rppg"] = max(self.min_weight, weights["rppg"] - self.adj.get("decrease", 0.2))
                explanations.append(f"rPPG SNR ({snr:.1f}dB) low -> decreased weight to {weights['rppg']:.2f}")
        else:
            weights["rppg"] = 0.0

        # MER check
        if availability.get("mer", "mer" in preds):
            flow_noise = quality.get("mer", {}).get("flow_noise", 0.1)
            if flow_noise <= self.thresholds.get("mer_flow_noise", 0.2):
                weights["mer"] = min(self.max_weight, weights["mer"] + self.adj.get("increase", 0.2))
                explanations.append(f"MER flow noise ({flow_noise:.2f}) low -> increased weight to {weights['mer']:.2f}")
            else:
                weights["mer"] = max(self.min_weight, weights["mer"] - self.adj.get("decrease", 0.2))
                explanations.append(f"MER flow noise ({flow_noise:.2f}) high -> decreased weight to {weights['mer']:.2f}")
        else:
            weights["mer"] = 0.0

        # Fuse predictions
        weighted_sum = np.zeros(4, dtype=np.float64)
        total_weight = 0.0

        for mod in ["macro", "rppg", "mer"]:
            w = weights.get(mod, 0.0)
            if w > 0 and mod in preds and preds[mod] is not None:
                prob = self._extract_prob(preds[mod])
                if prob.shape[0] == 4 and np.any(prob > 0):
                    weighted_sum += w * prob
                    total_weight += w

        if total_weight > 0:
            current_fused = weighted_sum / total_weight
            sum_prob = np.sum(current_fused)
            if sum_prob > 0:
                current_fused = current_fused / sum_prob
        else:
            current_fused = np.ones(4, dtype=np.float64) / 4.0

        # State Controller: temporal smoothing
        self.history.append(current_fused)
        fused = np.mean(self.history, axis=0)
        fused = fused / np.sum(fused)

        return {
            "clip_id": bundle.get("clip_id", ""),
            "modality": "fusion",
            "engagement": fused,
            "predictions": {"engagement": fused},
            "explanation": " | ".join(explanations),
            "status": "success",
        }
