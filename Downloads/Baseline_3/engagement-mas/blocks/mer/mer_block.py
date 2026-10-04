import os
import cv2
import numpy as np
import hashlib
from typing import Dict, Any, List

# Robust MediaPipe FaceMesh import
FaceMesh = None
try:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
    FaceMesh = mp_face_mesh.FaceMesh
except Exception:
    try:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
        FaceMesh = mp_face_mesh.FaceMesh
    except Exception:
        try:
            import mediapipe as mp
            FaceMesh = getattr(getattr(mp, "solutions", None), "face_mesh", None).FaceMesh
        except Exception:
            FaceMesh = None

# Simple deterministic Bi-WOOF implementation (histogram of oriented optical flow)
def bi_woof(flow: np.ndarray, num_bins: int = 8) -> np.ndarray:
    """Compute a Bi‑WOOF descriptor for a given optical‑flow field."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    ang = np.mod(ang, 2 * np.pi)
    bin_edges = np.linspace(0, 2 * np.pi, num_bins + 1)
    hist, _ = np.histogram(ang, bins=bin_edges, weights=mag)
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist.astype(np.float32)

class MERBlock:
    """Deterministic Micro‑Expression (MER) block.

    Provides ``run(clip_path)`` returning a dict compatible with the
    common block interface.
    """

    def __init__(self, seed: int = 42, num_bins: int = 8):
        np.random.seed(seed)
        self.seed = seed
        self.num_bins = num_bins
        if FaceMesh is not None:
            try:
                self.mp_face_mesh = FaceMesh(static_image_mode=True,
                                             max_num_faces=1,
                                             refine_landmarks=True,
                                             min_detection_confidence=0.5)
            except Exception:
                self.mp_face_mesh = None
        else:
            self.mp_face_mesh = None

    @staticmethod
    def _clip_id_from_path(path: str) -> str:
        return hashlib.sha256(os.path.abspath(path).encode()).hexdigest()[:16]

    def _extract_face_roi(self, frame: np.ndarray) -> np.ndarray:
        """Crop the face ROI using MediaPipe landmarks.
        Returns a resized (112,112,3) RGB array.
        """
        if self.mp_face_mesh is None:
            return cv2.resize(frame, (112, 112))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = self.mp_face_mesh.process(rgb)
        except Exception:
            return cv2.resize(frame, (112, 112))
        if not results.multi_face_landmarks:
            return cv2.resize(frame, (112, 112))
        # Get bounding box of landmarks
        h, w, _ = frame.shape
        xs = [lm.x for lm in results.multi_face_landmarks[0].landmark]
        ys = [lm.y for lm in results.multi_face_landmarks[0].landmark]
        x_min, x_max = max(min(xs) - 0.05, 0.0), min(max(xs) + 0.05, 1.0)
        y_min, y_max = max(min(ys) - 0.05, 0.0), min(max(ys) + 0.05, 1.0)
        left, right = int(x_min * w), int(x_max * w)
        top, bottom = int(y_min * h), int(y_max * h)
        roi = frame[top:bottom, left:right]
        if roi.size == 0:
            return cv2.resize(frame, (112, 112))
        return cv2.resize(roi, (112, 112))

    def run(self, clip_path: str) -> Dict[str, Any]:
        if not os.path.isfile(clip_path):
            return {"clip_id": None, "modality": "mer", "status": f"File not found: {clip_path}"}
        try:
            cap = cv2.VideoCapture(clip_path)
            frame_idx = 0
            rois: List[np.ndarray] = []
            timestamps: List[int] = []
            stride = 5
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % stride == 0:
                    roi = self._extract_face_roi(frame)
                    rois.append(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                    timestamps.append(frame_idx)
                frame_idx += 1
            cap.release()
            if len(rois) < 2:
                raise ValueError("Video too short for optical flow.")

            # Compute optical flow between consecutive ROI frames
            flows = []
            for i in range(len(rois) - 1):
                if hasattr(cv2, 'optflow') and hasattr(cv2.optflow, 'DualTVL1OpticalFlow_create'):
                    try:
                        flow = cv2.optflow.DualTVL1OpticalFlow_create().calc(rois[i], rois[i + 1], None)
                    except Exception:
                        flow = cv2.calcOpticalFlowFarneback(rois[i], rois[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                else:
                    flow = cv2.calcOpticalFlowFarneback(rois[i], rois[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flows.append(flow)

            # Compute average magnitude per flow to find apex frame
            mags = [float(np.mean(np.linalg.norm(f, axis=-1))) for f in flows]
            apex_idx = int(np.argmax(mags))  # index of flow leading to apex
            # Onset is the frame before the apex flow start
            onset_idx = max(apex_idx - 1, 0)

            # Concatenate flows from onset to apex (inclusive)
            selected_flows = flows[onset_idx:apex_idx + 1]
            # Bi‑WOOF descriptor for each flow, then average
            descriptors = [bi_woof(f, self.num_bins) for f in selected_flows]
            feature_vec = np.mean(descriptors, axis=0) if descriptors else np.zeros(self.num_bins, dtype=np.float32)

            # Quality metrics
            flow_noise = float(np.std(mags))
            # Face‑track stability: std of ROI bounding box centers (approx.)
            centers = []
            for roi in rois:
                h, w = roi.shape[:2]
                centers.append((w / 2, h / 2))
            centers = np.array(centers)
            track_stability = float(np.mean(np.std(centers, axis=0)))
            # Frame‑rate adequacy: ratio of actual fps to 30
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            framerate_factor = float(min(fps, 30.0) / 30.0)
            # Compression/blur estimate via Laplacian variance of median frame
            median_frame = np.median(np.stack(rois, axis=0), axis=0).astype(np.uint8)
            blur_est = cv2.Laplacian(median_frame, cv2.CV_64F).var()

            quality = {
                "flow_noise": flow_noise,
                "track_stability": track_stability,
                "framerate_factor": framerate_factor,
                "blur_estimate": blur_est,
            }

            return {
                "clip_id": self._clip_id_from_path(clip_path),
                "modality": "mer",
                "features": feature_vec,
                "predictions": {},  # no classification head
                "quality_metrics": quality,
                "timestamps": timestamps,
                "status": "success",
            }
        except Exception as e:
            return {"clip_id": self._clip_id_from_path(clip_path), "modality": "mer", "status": str(e)}
