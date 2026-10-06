import os
import math
import hashlib
import numpy as np
import cv2
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
from typing import Dict, Any, Optional
import joblib

try:
    from face2ppg import Face2PPG
except ImportError:
    Face2PPG = None


class RealRPPGProcessor:
    """Real Remote Photoplethysmography (rPPG) processor.
    Extracts blood volume pulse (BVP) from facial video using the Green channel
    and Chrominance (CHROM) method, performs bandpass filtering, and estimates
    heart rate (HR), signal-to-noise ratio (SNR), peak prominence, and HR stability.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        try:
            from mediapipe.python.solutions.face_mesh import FaceMesh
            self.face_mesh = FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception:
            self.face_mesh = None

    def _extract_face_roi(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.face_mesh is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                # Use forehead & cheek region (landmarks: 10, 151, 9, 8, 107, 336, 117, 346)
                xs = [p.x for p in lm]
                ys = [p.y for p in lm]
                x_min, x_max = max(0, int(min(xs) * w)), min(w, int(max(xs) * w))
                y_min, y_max = max(0, int(min(ys) * h)), min(h, int(max(ys) * h))
                # Focus on upper 65% of face (forehead + cheeks) to avoid mouth motion
                roi_h = y_max - y_min
                roi = frame[y_min:y_min + int(roi_h * 0.65), x_min:x_max]
                if roi.size > 0:
                    return roi
        # Fallback: central upper-middle region
        return frame[int(h * 0.2):int(h * 0.65), int(w * 0.3):int(w * 0.7)]

    def process(self, clip_path: str) -> Dict[str, Any]:
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video {clip_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        green_means = []
        red_means = []
        blue_means = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            roi = self._extract_face_roi(frame)
            if roi.size > 0:
                # Mean colors in ROI
                b_m = float(np.mean(roi[:, :, 0]))
                g_m = float(np.mean(roi[:, :, 1]))
                r_m = float(np.mean(roi[:, :, 2]))
            else:
                b_m, g_m, r_m = 0.0, 0.0, 0.0
            blue_means.append(b_m)
            green_means.append(g_m)
            red_means.append(r_m)

        cap.release()
        num_frames = len(green_means)
        if num_frames < 10:
            raise ValueError(f"Video {clip_path} contains too few frames ({num_frames})")

        # CHROM signal construction: Xs = 3R - 2G, Ys = 1.5R + G - 1.5B
        r_arr = np.array(red_means, dtype=np.float64)
        g_arr = np.array(green_means, dtype=np.float64)
        b_arr = np.array(blue_means, dtype=np.float64)

        # Standardize color signals
        r_norm = (r_arr - np.mean(r_arr)) / (np.std(r_arr) + 1e-6)
        g_norm = (g_arr - np.mean(g_arr)) / (np.std(g_arr) + 1e-6)
        b_norm = (b_arr - np.mean(b_arr)) / (np.std(b_arr) + 1e-6)

        xs = 3.0 * r_norm - 2.0 * g_norm
        ys = 1.5 * r_norm + g_norm - 1.5 * b_norm
        std_xs = np.std(xs) + 1e-6
        std_ys = np.std(ys) + 1e-6
        alpha = std_xs / std_ys
        chrom_signal = xs - alpha * ys

        # Bandpass filter (0.7 Hz to 4.0 Hz -> 42 to 240 BPM)
        nyq = fps / 2.0
        low = max(0.01, min(0.7 / nyq, 0.95))
        high = max(low + 0.05, min(4.0 / nyq, 0.99))
        b, a = butter(3, [low, high], btype="band")
        filtered_bvp = filtfilt(b, a, chrom_signal)

        # Spectral analysis via FFT
        freqs = rfftfreq(len(filtered_bvp), d=1.0 / fps)
        psd = np.abs(rfft(filtered_bvp)) ** 2
        cardiac_mask = (freqs >= 0.7) & (freqs <= 4.0)

        if np.any(cardiac_mask) and np.sum(psd[cardiac_mask]) > 0:
            cardiac_freqs = freqs[cardiac_mask]
            cardiac_psd = psd[cardiac_mask]
            peak_idx = int(np.argmax(cardiac_psd))
            peak_freq = float(cardiac_freqs[peak_idx])
            hr = float(peak_freq * 60.0)

            # Signal-to-Noise Ratio (SNR) in dB
            # Signal power: power within +/- 0.2 Hz of peak
            sig_mask = (cardiac_freqs >= peak_freq - 0.2) & (cardiac_freqs <= peak_freq + 0.2)
            sig_power = float(np.sum(cardiac_psd[sig_mask]))
            noise_power = float(np.sum(cardiac_psd[~sig_mask]))
            snr = float(10.0 * np.log10((sig_power + 1e-8) / (noise_power + 1e-8)))
            snr = max(-10.0, min(20.0, snr))

            # Peak prominence (relative to average spectral noise)
            mean_noise = float(np.mean(cardiac_psd))
            peak_power = float(cardiac_psd[peak_idx])
            peak_prominence = float((peak_power - mean_noise) / (peak_power + 1e-6))
            peak_prominence = max(0.0, min(1.0, peak_prominence))

            # HR Stability: compare HR on first half vs second half
            half = num_frames // 2
            if half > 15:
                freqs1 = rfftfreq(half, d=1.0 / fps)
                psd1 = np.abs(rfft(filtered_bvp[:half])) ** 2
                mask1 = (freqs1 >= 0.7) & (freqs1 <= 4.0)
                hr1 = float(freqs1[mask1][np.argmax(psd1[mask1])] * 60.0) if np.any(mask1) else hr

                freqs2 = rfftfreq(len(filtered_bvp[half:]), d=1.0 / fps)
                psd2 = np.abs(rfft(filtered_bvp[half:])) ** 2
                mask2 = (freqs2 >= 0.7) & (freqs2 <= 4.0)
                hr2 = float(freqs2[mask2][np.argmax(psd2[mask2])] * 60.0) if np.any(mask2) else hr
                hr_stability = float(1.0 / (1.0 + abs(hr1 - hr2) / 10.0))
            else:
                hr_stability = 0.8
        else:
            hr = 72.0
            snr = 0.0
            peak_prominence = 0.1
            hr_stability = 0.5

        # Normalize BVP to 128 dimensions for consistent fusion features
        t_orig = np.linspace(0, 1, len(filtered_bvp))
        t_128 = np.linspace(0, 1, 128)
        bvp_128 = np.interp(t_128, t_orig, filtered_bvp).astype(np.float32)

        return {
            "bvp": bvp_128,
            "raw_bvp": filtered_bvp.astype(np.float32),
            "hr": hr,
            "snr": snr,
            "peak_prominence": peak_prominence,
            "hr_stability": hr_stability,
        }


class RPPGBlock:
    """Deterministic, robust rPPG block.
    Extracts physiological BVP and Heart Rate signals, and predicts 4-class engagement.
    """

    def __init__(self, seed: int = 42, weights_path: Optional[str] = None):
        np.random.seed(seed)
        self.processor = RealRPPGProcessor(seed=seed)
        self.classifier = None
        self.scaler = None

        # Try to load trained classifier
        candidates = [
            weights_path,
            os.path.abspath("weights/rppg/classifier.pkl"),
            os.path.abspath("../weights/rppg/classifier.pkl"),
        ]
        for p in candidates:
            if p and os.path.isfile(p):
                try:
                    data = joblib.load(p)
                    if isinstance(data, dict):
                        self.classifier = data.get("model")
                        self.scaler = data.get("scaler")
                    else:
                        self.classifier = data
                    print(f"Loaded trained rPPG classifier from {p}")
                    break
                except Exception as e:
                    print(f"Warning: Failed to load rPPG classifier from {p}: {e}")

    @staticmethod
    def _clip_id_from_path(path: str) -> str:
        return hashlib.sha256(os.path.abspath(path).encode()).hexdigest()[:16]

    def _extract_feature_vector(self, hr: float, quality: Dict[str, float], bvp: np.ndarray) -> np.ndarray:
        """Extract compact feature vector for engagement classification."""
        return np.array([
            hr,
            quality["snr"],
            quality["peak_prominence"],
            quality["hr_stability"],
            float(np.mean(bvp)),
            float(np.std(bvp)),
            float(np.max(bvp) - np.min(bvp)),
        ], dtype=np.float32)

    def run(self, clip_path: str) -> Dict[str, Any]:
        if not os.path.isfile(clip_path):
            print(f"[RPPGBlock Warning] Video file not found: {clip_path}")
            return {"clip_id": None, "modality": "rppg", "status": f"File not found: {clip_path}"}
        try:
            result = self.processor.process(clip_path)
            bvp = result["bvp"]
            hr = float(result["hr"])
            quality = {
                "snr": float(result.get("snr", 0.0)),
                "peak_prominence": float(result.get("peak_prominence", 0.0)),
                "hr_stability": float(result.get("hr_stability", 0.0)),
            }
            timestamps = list(range(len(bvp)))

            # Predict engagement
            if self.classifier is not None:
                feat_vec = self._extract_feature_vector(hr, quality, bvp).reshape(1, -1)
                if self.scaler is not None:
                    feat_vec = self.scaler.transform(feat_vec)
                eng_probs = self.classifier.predict_proba(feat_vec)[0].astype(np.float32)
            else:
                # Balanced physiological heuristic without artificial class-2 bias
                # HR mapping: <55: Very Low (0), 55-70: Low (1), 70-85: Engaged (2), >85: High (3)
                hr_norm = (hr - 72.0) / 12.0
                logits = np.array([
                    -1.2 * hr_norm - 0.8,
                    -0.6 * (hr_norm + 0.4) ** 2,
                    -0.6 * (hr_norm - 0.3) ** 2,
                    1.2 * hr_norm - 0.8,
                ], dtype=np.float64)
                exp_l = np.exp(logits - np.max(logits))
                eng_probs = (exp_l / np.sum(exp_l)).astype(np.float32)

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

