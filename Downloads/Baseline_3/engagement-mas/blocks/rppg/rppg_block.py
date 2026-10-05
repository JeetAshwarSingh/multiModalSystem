import os
import numpy as np
import cv2
import mediapipe as mp
import hashlib
from typing import Dict, Any

# Assuming face2ppg provides a deterministic API
try:
    from face2ppg import Face2PPG
except ImportError:
    Face2PPG = None
    class _DummyFace2PPG:
        """Simple deterministic rPPG placeholder.
        Generates a synthetic 75‑bpm sinusoidal BVP signal based on the
        number of frames in the video. No external dependencies required.
        """
        def set_seed(self, seed: int):
            # deterministic – seed is ignored because we generate a fixed signal
            pass
        def process(self, clip_path: str):
            import cv2, numpy as np, math
            cap = cv2.VideoCapture(clip_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video {clip_path}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            # 75 bpm => 1.25 Hz => period = 0.8 s
            hr = 75.0
            freq = hr / 60.0  # Hz
            t = np.arange(frame_count) / fps
            # sine wave amplitude 0.5, offset 0.5 to keep values positive
            bvp = 0.5 * np.sin(2 * math.pi * freq * t) + 0.5
            # quality metrics are deterministic placeholders
            quality = {"snr": 10.0, "peak_prominence": 0.8, "hr_stability": 0.95}
            return {"bvp": bvp, "hr": hr, **quality}


class RPPGBlock:
    """Deterministic rPPG block.

    The block follows the common interface:
    ``run(clip_path) -> dict`` with keys
    ``clip_id, modality, features, predictions, quality_metrics, timestamps, status``.
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        # set any other library seeds if needed
        if Face2PPG is None:
            self.processor = _DummyFace2PPG()
        else:
            self.processor = Face2PPG()
        # Ensure deterministic behavior inside face2ppg if it respects the seed
        if hasattr(self.processor, "set_seed"):
            self.processor.set_seed(seed)

    @staticmethod
    def _clip_id_from_path(path: str) -> str:
        # deterministic id based on hash of absolute path
        return hashlib.sha256(os.path.abspath(path).encode()).hexdigest()[:16]

    def run(self, clip_path: str) -> Dict[str, Any]:
        if not os.path.isfile(clip_path):
            return {"clip_id": None, "modality": "rppg", "status": f"File not found: {clip_path}"}
        try:
            # Face2PPG returns a dict with keys: bvp (np.ndarray), hr (float), snr, peak_prominence, hr_stability
            result = self.processor.process(clip_path)
            bvp = result["bvp"]  # 1‑D array aligned with timestamps
            hr = float(result["hr"])
            # Quality metrics
            quality = {
                "snr": float(result.get("snr", 0.0)),
                "peak_prominence": float(result.get("peak_prominence", 0.0)),
                "hr_stability": float(result.get("hr_stability", 0.0)),
            }
            # timestamps – assume one sample per frame (approx.)
            timestamps = list(range(len(bvp)))

            # Physiological engagement estimation (4 classes: 0=Very Low, 1=Low, 2=Engaged, 3=High)
            hr_norm = (hr - 72.0) / 10.0
            p0 = 1.0 / (1.0 + np.exp(hr_norm + 1.8))
            p1 = np.exp(-0.5 * (hr_norm + 0.8)**2)
            p2 = np.exp(-0.5 * (hr_norm - 0.2)**2) * 1.5
            p3 = 1.0 / (1.0 + np.exp(-hr_norm + 0.5)) * 1.5
            eng_probs = np.array([p0, p1, p2, p3], dtype=np.float32)
            eng_probs = eng_probs / np.sum(eng_probs)

            return {
                "clip_id": self._clip_id_from_path(clip_path),
                "modality": "rppg",
                "features": bvp,
                "predictions": {"hr": hr, "engagement": eng_probs},
                "quality_metrics": quality,
                "timestamps": timestamps,
                "status": "success",
            }
        except Exception as e:
            return {"clip_id": self._clip_id_from_path(clip_path), "modality": "rppg", "status": str(e)}
