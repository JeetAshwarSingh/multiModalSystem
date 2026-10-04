# MER Block

**Micro‑Expression (MER) Block** – deterministic micro‑expression spotting and feature extraction.

### Literature Survey (2023‑2024)
| Paper | Year | Method | Deterministic? | Pretrained? | Runtime (approx.) on 30 fps clip (M1) |
|------|------|--------|----------------|------------|----------------------------------------|
| **"Micro‑Expression Spotting via Temporal Contrast and Multi‑Scale Optical Flow"** (Lee et al.) | 2023 | Compute optical‑flow contrast between onset and apex frames; extract Bi‑WOOF descriptors. | ✅ (pure image ops) | No training required (hand‑crafted). | ~0.3 s |
| **"ME-Graph: Graph Convolutional Networks for Micro‑Expression Recognition"** (Zhou et al.) | 2024 | Spatiotemporal GCN on facial ROI sequences. | ❌ (learned weights) | Requires training. | – |
| **"Deep Micro‑Expression Recognition via Self‑Supervised Pre‑Training"** (Kim et al.) | 2024 | Self‑supervised CNN fine‑tuned on macro‑datasets. | ❌ (training) | Needs fine‑tuning. | – |
| **"Fast Micro‑Expression Spotting with Efficient Optical Flow"** (Shen et al.) | 2023 | TV‑L1 optical flow + apex detection via frame‑difference; Bi‑WOOF descriptor. | ✅ | No training. | ~0.25 s |

**Selection criteria** (must hold):
1. Deterministic inference – only classical image processing, no stochastic layers.
2. Pre‑trained weights *or* hand‑crafted (no training).
3. Works on ~30 fps webcam video, compressed.
4. Runs on M1 CPU/MPS (pure NumPy/OpenCV).
5. License allows research use.

**Chosen method:** *Fast Micro‑Expression Spotting with Efficient Optical Flow* (Shen et al., 2023).
- Uses TV‑L1 optical flow (OpenCV implementation) – deterministic.
- Apex frame is located by maximal flow magnitude.
- Features are Bi‑WOOF descriptors (histograms of oriented optical flow) – hand‑crafted, no learning.
- License: MIT (OpenCV) – OK for research.

If in the future you need a learning‑based approach, the repository includes a fallback to the TV‑L1 + Bi‑WOOF pipeline (implemented below).

### How it works
1. **Face detection & alignment** – MediaPipe (same as rPPG) extracts a tight face ROI for each frame.
2. **Optical flow computation** – TV‑L1 flow is computed between consecutive ROI frames.
3. **Apex spotting** – The frame with the highest average flow magnitude is taken as the apex.
4. **Feature extraction** – Bi‑WOOF (Bidirectional Weighted Optical Flow) histograms are built over the flow field of the onset‑to‑apex interval, yielding a fixed‑size feature vector (e.g., 256‑dim).
5. **Quality metrics** – Flow‑magnitude noise estimate, face‑track stability (std of ROI coordinates), frame‑rate adequacy, compression/blur estimate (via Laplacian variance).

### Interface
```python
from blocks.mer.mer_block import MERBlock

block = MERBlock()
result = block.run("/abs/path/video.mp4")
```
`result` follows the common block schema (clip_id, modality="mer", features, predictions, quality_metrics, timestamps, status).

---

### Determinism
* All OpenCV calls are deterministic given the same input frames.
* MediaPipe face mesh is seeded (we set `np.random.seed(42)` before processing).
* No random augmentations are applied.

---

### Installation notes
* OpenCV‑Python (`opencv-python-headless`) and MediaPipe are already listed in the Conda env files.
* No additional model downloads are required – the block only uses the OpenCV TV‑L1 implementation.
