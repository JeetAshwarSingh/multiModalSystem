# rPPG Block

The **rPPG** block provides a deterministic wrapper around the `face2ppg` library. It extracts physiological signals from a video clip without any training.

## How it works
1. **Face detection & ROI extraction** – `face2ppg` tracks the face region in each frame using MediaPipe (CPU‑only) and crops the ROI.
2. **Signal extraction** – The RGB intensity changes inside the ROI are processed with a built‑in POS (Plane‑Orthogonal‑to‑Skin) algorithm to obtain a Blood Volume Pulse (BVP) waveform.
3. **Post‑processing** – From the BVP we compute:
   * **Heart‑Rate (HR) estimate** (beats‑per‑minute) via peak detection in the frequency domain.
   * **Signal‑quality metrics**:
     * **SNR** in the typical HR band (0.75‑4 Hz).
     * **Spectral peak prominence** (how dominant the HR peak is).
     * **HR stability** – variation of HR over sliding windows.

All steps are fully deterministic given a fixed random seed and the same video input.

## Interface
```python
from blocks.rppg.rppg_block import RPPGBlock

block = RPPGBlock()
result = block.run(clip_path="/absolute/path/to/video.mp4")
```
`result` is a dictionary with the following keys (matching the common block interface):
* `clip_id` – identifier derived from the filename.
* `modality` – string ``"rppg"``.
* `features` – the raw BVP signal (1‑D NumPy array).
* `predictions` – a dict `{"hr": float}`.
* `quality_metrics` – dict containing ``snr``, ``peak_prominence`` and ``hr_stability``.
* `timestamps` – frame indices (list of ints) corresponding to each BVP sample.
* `status` – ``"success"`` or error description.

## Determinism
* The `face2ppg` library runs deterministically when the underlying MediaPipe face mesh is seeded (we set `np.random.seed(42)` and `torch.manual_seed(42)`).
* No stochastic augmentations or dropout are used.

---

### Installation notes
`face2ppg` is listed in `env/environment-mac.yml` and `environment-cloud.yml`. It pulls in MediaPipe and a small PyTorch‑based helper, all of which work on CPU/MPS.
