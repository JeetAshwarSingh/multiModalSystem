 # Macro Block

The **Macro** block is the only trainable component in the Engagement‑MAS pipeline. It combines a frozen appearance backbone (DINOv2 ViT‑B/14) with a lightweight behavioral stream (gaze, head pose, AU intensities) and a small temporal transformer to predict engagement and optional auxiliary affective states.

## Architecture Overview
1. **Face detection & cropping** – MediaPipe‑FaceMesh is used to locate the face in each frame deterministicly. The cropped face is resized to 224×224.
2. **Appearance stream** – A frozen DINOv2 ViT‑B/14 backbone (from `timm`) extracts a 768‑dim per‑frame embedding. An optional **LoRA** module can be attached for low‑rank fine‑tuning.
3. **Behavioral stream** – For each frame we compute:
   * **Gaze direction** (unit vector) via eye‑landmark geometry.
   * **Head pose** (Euler angles) with `cv2.solvePnP` on a canonical 3‑D face model.
   * **Action‑Unit (AU) intensities** – approximated from MediaPipe landmark distances (e.g., AU12 – lip corner puller).
   These are concatenated into a 30‑dim behavioral vector.
4. **Temporal transformer** – A 2‑layer `torch.nn.TransformerEncoder` processes the sequence of `[appearance; behavior]` vectors (dimension ≈ 800). Positional encodings are added.
5. **Classification heads** – A primary head predicts **Engagement** (4 classes). Auxiliary heads (Boredom, Confusion, Frustration) are toggled via config.
6. **Quality metrics** – Face‑detection rate, average blur (Laplacian variance), brightness, and a simple motion‑smoothness score are computed for downstream gating.

## Inference Interface
```python
from blocks.macro.macro_block import MacroBlock

block = MacroBlock()
result = block.run("/abs/path/to/video.mp4")
```
`result` follows the common interface:
* `clip_id` – deterministic hash of the video path.
* `modality` – ``"macro"``.
* `features` – final temporal embedding (1‑D np.ndarray).
* `predictions` – dict with keys `engagement`, optional `boredom`, `confusion`, `frustration` (class probabilities).
* `quality_metrics` – detection rate, blur, brightness, motion smoothness.
* `timestamps` – frame indices.
* `status` – ``"success"`` or error description.

## Determinism (Inference)
* Global RNG seeds are fixed (`np.random.seed(42)`, `torch.manual_seed(42)`).
* PyTorch deterministic flags are set:
  ```python
  torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```
* No dropout or stochastic augmentation is applied during ``run``.

## Training Notes (see `training/`)
* The appearance backbone is **frozen** by default; LoRA can be enabled to fine‑tune a low‑rank adaptation.
* Class imbalance is handled via a **weighted Cross‑Entropy loss** (weights computed from the training CSV) and optional **balanced sampling**.
* Checkpoints (`best.pt`, `last.pt`) are saved under `weights/macro/` and a config snapshot is stored alongside.

---

### Installation
All required packages are listed in `env/environment-mac.yml` and `env/environment-cloud.yml` (torch, timm, mediapipe, opencv, pandas, PyTorch‑Lightning, torchmetrics). No CUDA packages appear in the Mac environment.
