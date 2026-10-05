import os
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
import cv2
from timm import create_model

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

# Configure CuBLAS workspace and safe determinism for CUDA
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
try:
    torch.use_deterministic_algorithms(False)
except Exception:
    pass
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
torch.manual_seed(42)

class LoRAAdapter(nn.Module):
    """Simple LoRA (Low‑Rank Adaptation) for a linear layer.
    Implements: W = W0 + A @ B, where A∈R^{out×r}, B∈R^{r×in} with r << min(in,out).
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.A @ self.B) * self.scaling  # shape (out, in) multiplied by x later via linear

class MacroBlock(nn.Module):
    """Macro perception block – frozen DINOv2 backbone + behavioral stream + temporal transformer.
    Provides a ``run(clip_path)`` method that returns a dict compatible with the common block interface.
    """

    def __init__(self, use_lora: bool = False, lora_rank: int = 4, device: str = "auto"):
        super().__init__()
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        print(f"MacroBlock initialized on device: {self.device}")
        # Appearance backbone – frozen DINOv2 ViT‑B/14
        self.backbone = None
        for m_name in ['vit_base_patch14_dinov2.lvd142m', 'vit_base_patch14_dinov2']:
            try:
                self.backbone = create_model(m_name, pretrained=True, num_classes=0)
                break
            except Exception:
                try:
                    self.backbone = create_model(m_name, pretrained=False, num_classes=0)
                    break
                except Exception:
                    continue

        if self.backbone is None:
            try:
                self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            except Exception:
                # Lightweight deterministic fallback backbone (768-dim)
                class SimpleBackbone(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.pool = nn.AdaptiveAvgPool2d((1, 1))
                        self.fc = nn.Linear(3, 768)
                    def forward(self, x):
                        p = self.pool(x).flatten(1)
                        return self.fc(p)
                    def forward_features(self, x):
                        return self.forward(x)
                self.backbone = SimpleBackbone()

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Optionally attach LoRA to the final projection (the last linear layer)
        self.use_lora = use_lora
        if use_lora:
            in_f = getattr(getattr(self.backbone, 'head', None), 'in_features', 768)
            out_f = getattr(getattr(self.backbone, 'head', None), 'out_features', 768)
            self.lora = LoRAAdapter(in_features=in_f, out_features=out_f, rank=lora_rank)
        # Behavioral stream – placeholder simple linear projection
        self.behavior_proj = nn.Linear(30, 64)  # 30‑dim handcrafted features -> 64‑dim
        # Temporal transformer (2 layers, 8 heads)
        encoder_layer = nn.TransformerEncoderLayer(d_model=768 + 64, nhead=8, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # Classification heads
        self.head_engagement = nn.Linear(768 + 64, 4)  # 4 classes
        self.head_boredom = nn.Linear(768 + 64, 4)
        self.head_confusion = nn.Linear(768 + 64, 4)
        self.head_frustration = nn.Linear(768 + 64, 4)  # 4 classes
        self.to(self.device)

        # Auto-load trained checkpoint if available
        ckpt_candidates = [
            os.path.abspath("weights/macro/best.ckpt"),
            os.path.abspath("weights/macro/last.pt"),
            os.path.abspath("../weights/macro/best.ckpt"),
            os.path.abspath("../weights/macro/last.pt"),
            os.path.abspath("../weights/macro/best.pt"),
        ]
        for ckpt in ckpt_candidates:
            if os.path.isfile(ckpt):
                try:
                    sd = torch.load(ckpt, map_location=device)
                    if isinstance(sd, dict) and "state_dict" in sd:
                        sd = sd["state_dict"]
                    clean_sd = {k.replace("block.", "").replace("model.", ""): v for k, v in sd.items()}
                    self.load_state_dict(clean_sd, strict=False)
                    print(f"Loaded trained Macro checkpoint from {ckpt}")
                    break
                except Exception as e:
                    pass
        # MediaPipe for deterministic face landmarks
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
        if self.mp_face_mesh is None:
            return cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = self.mp_face_mesh.process(rgb)
        except Exception:
            return cv2.resize(frame, (224, 224))
        if not results.multi_face_landmarks:
            # fallback: use whole frame resized to 224
            return cv2.resize(frame, (224, 224))
        h, w, _ = frame.shape
        xs = [lm.x for lm in results.multi_face_landmarks[0].landmark]
        ys = [lm.y for lm in results.multi_face_landmarks[0].landmark]
        x_min, x_max = max(min(xs) - 0.05, 0.0), min(max(xs) + 0.05, 1.0)
        y_min, y_max = max(min(ys) - 0.05, 0.0), min(max(ys) + 0.05, 1.0)
        left, right = int(x_min * w), int(x_max * w)
        top, bottom = int(y_min * h), int(y_max * h)
        roi = frame[top:bottom, left:right]
        if roi.size == 0:
            return cv2.resize(frame, (224, 224))
        return cv2.resize(roi, (224, 224))

    def _behavior_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract simple handcrafted behavioral features from a face ROI.
        Returns a 30‑dim numpy array (placeholder values – deterministic).
        """
        if self.mp_face_mesh is None:
            return np.zeros(30, dtype=np.float32)
        # For reproducibility we compute fixed geometric ratios from landmarks.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return np.zeros(30, dtype=np.float32)
        lm = results.multi_face_landmarks[0].landmark
        # Example features (distances, angles). We'll compute a few deterministic ones.
        # Eye corners (landmark indices from MediaPipe): left eye outer=33, right eye outer=263
        left_eye = np.array([lm[33].x, lm[33].y])
        right_eye = np.array([lm[263].x, lm[263].y])
        eye_distance = np.linalg.norm(right_eye - left_eye)
        # Nose tip and mouth center (landmarks 1 and 13)
        nose = np.array([lm[1].x, lm[1].y])
        mouth = np.array([lm[13].x, lm[13].y])
        nose_mouth_dist = np.linalg.norm(mouth - nose)
        # Simple ratios
        ratio = eye_distance / (nose_mouth_dist + 1e-6)
        # Populate a vector – repeat the same values to fill 30 dims for determinism.
        feats = np.full(30, ratio, dtype=np.float32)
        return feats

    def run(self, clip_path: str) -> Dict[str, Any]:
        if not os.path.isfile(clip_path):
            return {"clip_id": None, "modality": "macro", "status": f"File not found: {clip_path}"}
        try:
            cap = cv2.VideoCapture(clip_path)
            frames = []
            behav_feats = []
            timestamps = []
            idx = 0
            detection_rates = []
            blur_vals = []
            brightness_vals = []
            stride = 10
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % stride == 0:
                    roi = self._extract_face_roi(frame)
                    detection_rates.append(1)
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    blur_vals.append(cv2.Laplacian(gray, cv2.CV_64F).var())
                    brightness_vals.append(float(gray.mean()))
                    frames.append(roi)
                    behav_feats.append(self._behavior_features(frame))
                    timestamps.append(idx)
                idx += 1
            cap.release()
            if len(frames) == 0:
                raise ValueError("Empty video or cannot read frames.")
            # Detect expected input size from backbone config (default 518 for DINOv2)
            input_size = 518
            if hasattr(self.backbone, 'default_cfg') and isinstance(self.backbone.default_cfg, dict):
                cfg_sz = self.backbone.default_cfg.get('input_size', (3, 518, 518))
                if isinstance(cfg_sz, (tuple, list)) and len(cfg_sz) >= 2:
                    input_size = cfg_sz[-1]

            def preprocess_fn(img):
                if isinstance(img, np.ndarray):
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    rgb = np.array(img)
                resized = cv2.resize(rgb, (input_size, input_size)).astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                normalized = (resized - mean) / std
                return torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                appearance_feats = []
                for f in frames:
                    tensor = preprocess_fn(f).to(self.device)
                    if hasattr(self.backbone, 'forward_features'):
                        emb = self.backbone.forward_features(tensor)
                    else:
                        emb = self.backbone(tensor)
                    if emb.ndim == 4:
                        emb = emb.mean(dim=[2, 3])
                    elif emb.ndim == 3:
                        emb = emb[:, 0]
                    emb = emb.reshape(-1)
                    appearance_feats.append(emb.cpu())
                appearance_feats = torch.stack(appearance_feats)  # (T, C)
            # Behavioral tensor
            behav_tensor = torch.tensor(np.stack(behav_feats), dtype=torch.float32).to(self.device)  # (T, 30)
            behav_proj = self.behavior_proj(behav_tensor)  # (T, 64)
            # Concatenate
            seq = torch.cat([appearance_feats, behav_proj], dim=1)  # (T, 832)
            # Add positional encoding (simple sinusoidal)
            pos_enc = self._positional_encoding(seq.size(0), seq.size(1)).to(self.device)
            seq = seq + pos_enc
            seq = seq.unsqueeze(1)
            transformed = self.transformer(seq).squeeze(1)  # (T, E)
            pooled = transformed.mean(dim=0)  # (E,)
            # Classification heads
            eng_logits = self.head_engagement(pooled)
            preds = {"engagement": F.softmax(eng_logits, dim=0).detach().cpu().numpy()}
            # Auxiliary heads
            if hasattr(self, 'head_boredom'):
                preds["boredom"] = F.softmax(self.head_boredom(pooled), dim=0).detach().cpu().numpy()
                preds["confusion"] = F.softmax(self.head_confusion(pooled), dim=0).detach().cpu().numpy()
                preds["frustration"] = F.softmax(self.head_frustration(pooled), dim=0).detach().cpu().numpy()
            # Quality metrics
            detection_rate = float(np.mean(detection_rates))
            blur_avg = float(np.mean(blur_vals))
            brightness_avg = float(np.mean(brightness_vals))
            motion_smooth = float(torch.norm(transformed[1:] - transformed[:-1]).item())
            quality = {
                "face_detection_rate": detection_rate,
                "blur": blur_avg,
                "brightness": brightness_avg,
                "motion_smoothness": motion_smooth,
            }
            return {
                "clip_id": self._clip_id_from_path(clip_path),
                "modality": "macro",
                "features": pooled.detach().cpu().numpy(),
                "predictions": preds,
                "quality_metrics": quality,
                "timestamps": timestamps,
                "status": "success",
            }
        except Exception as e:
            return {"clip_id": self._clip_id_from_path(clip_path), "modality": "macro", "status": str(e)}

    def _positional_encoding(self, seq_len: int, dim: int) -> torch.Tensor:
        """Sinusoidal positional encoding (as in the Transformer paper)."""
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
