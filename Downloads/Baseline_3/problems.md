# Engagement-MAS: Problems & Solutions

> This document lists every critical problem identified in the current codebase, explains **why** it causes bad results, and provides a concrete **solution** for each.

---

## Table of Contents

1. [Problem 1: rPPG Dummy Processor — Identical Output for Every Video](#problem-1-rppg-dummy-processor--identical-output-for-every-video)
2. [Problem 2: Video File Paths Not Accessible — Macro Block Fails Silently](#problem-2-video-file-paths-not-accessible--macro-block-fails-silently)
3. [Problem 3: rPPG & MER Use Hardcoded Heuristics Instead of Trained Classifiers](#problem-3-rppg--mer-use-hardcoded-heuristics-instead-of-trained-classifiers)
4. [Problem 4: Early Fusion Uses Random Projection — Never Trained](#problem-4-early-fusion-uses-random-projection--never-trained)
5. [Problem 5: Macro Checkpoint Loaded with strict=False — Mismatched Weights](#problem-5-macro-checkpoint-loaded-with-strictfalse--mismatched-weights)
6. [Problem 6: Behavioral Features Are Meaningless — Same Number Repeated 30 Times](#problem-6-behavioral-features-are-meaningless--same-number-repeated-30-times)
7. [Problem 7: Fusion Cascade Failure — Garbage In, Garbage Out](#problem-7-fusion-cascade-failure--garbage-in-garbage-out)
8. [Bonus: Severe Class Imbalance in Test Set](#bonus-severe-class-imbalance-in-test-set)
9. [Priority Summary](#priority-summary)

---

## Problem 1: rPPG Dummy Processor — Identical Output for Every Video

### What's Wrong

The rPPG block depends on an external library called `face2ppg` to extract heart rate from facial video. However, **`face2ppg` is not installed** — it is not listed in any `requirements.txt` or `environment.yml` file despite the README claiming it is.

When `face2ppg` is missing, the code silently falls back to a `_DummyFace2PPG` class (defined in `blocks/rppg/rppg_block.py`, lines 13–37). This dummy:

- **Always returns HR = 75.0 BPM** regardless of video content
- Generates a fixed synthetic sine wave as the BVP signal
- Returns hardcoded quality metrics (`snr=10.0`, `peak_prominence=0.8`, `hr_stability=0.95`)

The engagement prediction heuristic (lines 82–88) then converts this constant HR into a **fixed probability vector** for every single video:

```
[0.0386, 0.1934, 0.5287, 0.2392]  →  Always predicts Class 2 (Engaged)
```

**Evidence:** In `resultAndAnalysis/rppg_predictions.csv`, all 1784 rows have identical probability values and `pred_engagement = 2`.

**Impact:** The 49.4% accuracy is purely coincidental — Class 2 happens to be 49.4% of the test set.

### How to Fix

**Option A — Install a real rPPG library:**

```bash
pip install face2ppg
```

If `face2ppg` is not a real public package, use an alternative like `pyVHR` or `rPPG-Toolbox`:

```bash
pip install pyvhr
```

Then update `rppg_block.py` to use it:

```python
from pyvhr.analysis.pipeline import Pipeline

class RPPGBlock:
    def __init__(self, seed=42):
        self.seed = seed

    def run(self, clip_path):
        pipe = Pipeline()
        bvps, timesES, bpmES = pipe.run_on_video(clip_path, roi_method='convexhull',
                                                   roi_approach='patches',
                                                   method='POS',  # Plane-Orthogonal-to-Skin
                                                   estimate='clustering')
        hr = float(np.median(bpmES))
        bvp = bvps.flatten()
        # ... rest of processing
```

**Option B — Implement basic rPPG from scratch (GREEN channel method):**

```python
def _extract_rppg_signal(self, clip_path):
    cap = cv2.VideoCapture(clip_path)
    green_means = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Detect face ROI
        roi = self._extract_face_roi(frame)
        # Extract mean green channel intensity
        green_means.append(np.mean(roi[:, :, 1]))  # Green channel
    cap.release()

    signal = np.array(green_means)
    # Bandpass filter (0.7 - 4 Hz for heart rate range 42-240 BPM)
    from scipy.signal import butter, filtfilt
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nyq = fps / 2.0
    low, high = 0.7 / nyq, 4.0 / nyq
    b, a = butter(3, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)

    # Estimate HR from peak frequency
    from scipy.fft import rfft, rfftfreq
    spectrum = np.abs(rfft(filtered))
    freqs = rfftfreq(len(filtered), d=1.0/fps)
    cardiac_mask = (freqs >= 0.7) & (freqs <= 4.0)
    peak_freq = freqs[cardiac_mask][np.argmax(spectrum[cardiac_mask])]
    hr = peak_freq * 60.0

    return filtered, hr
```

**Also:** Add `face2ppg` (or the alternative) to `requirements-mac.txt`, `requirements-windows.txt`, and `requirements-cloud.txt`.

---

## Problem 2: Video File Paths Not Accessible — Macro Block Fails Silently

### What's Wrong

The `test.csv` file contains video paths like:

```
/mnt/c/Users/puneet/Downloads/DAiSEE/DataSet/Test/500044/5000441001/5000441001.avi
```

These are **Windows WSL (Windows Subsystem for Linux) paths** that only work on the specific machine where the dataset was originally set up. When the evaluation runs on any other machine (Mac, cloud, different Windows):

1. `macro_block.py` line 209: `os.path.isfile(clip_path)` returns `False`
2. The block returns `{"status": "File not found: ..."}` with **no predictions**
3. The evaluator receives no engagement probabilities → sets them to `[0, 0, 0, 0]`
4. The fallback logic `np.argmax([0,0,0,0])` returns index `0` → predicts Class 0 for every clip
5. Since only 4 out of 1784 test clips are actually Class 0 → **0.22% accuracy**

**Evidence:** In `resultAndAnalysis/macro_predictions.csv`, all probabilities are `0.0` for every row.

### How to Fix

**Step 1 — Set the `DAISEE_ROOT` environment variable** to point to where DAiSEE is on your machine:

```bash
# Mac
export DAISEE_ROOT="/Users/jeetashwar/Downloads/DAiSEE"

# Windows WSL
export DAISEE_ROOT="/mnt/c/Users/yourname/Downloads/DAiSEE"

# Windows PowerShell
$env:DAISEE_ROOT = "C:\Users\yourname\Downloads\DAiSEE"
```

**Step 2 — Update `configs/datasets.yaml`** with the correct root path:

```yaml
datasets:
  daisee:
    root_dir: "/Users/jeetashwar/Downloads/DAiSEE"   # <-- your actual path
    train_csv: "../train.csv"
    val_csv: "../val.csv"
    test_csv: "../test.csv"
```

**Step 3 — Fix the CSV paths to be relative** instead of absolute. Run this script:

```python
import pandas as pd

for split in ['train.csv', 'val.csv', 'test.csv']:
    df = pd.read_csv(split)
    # Convert absolute WSL paths to relative paths from DAiSEE root
    df['clip_path'] = df['clip_path'].str.replace(
        r'.*/DAiSEE/', '', regex=True
    )
    df.to_csv(split, index=False)
    print(f"Fixed {split}: {df['clip_path'].iloc[0]}")
```

After this fix, paths will look like:
```
DataSet/Test/500044/5000441001/5000441001.avi
```

And the evaluate.py path resolution logic (which already tries multiple root candidates) will find them.

**Step 4 — Add path validation before running evaluation:**

```bash
python -m data.validate_dataset --config configs/datasets.yaml --split test
```

This should report how many video files are actually found.

---

## Problem 3: rPPG & MER Use Hardcoded Heuristics Instead of Trained Classifiers

### What's Wrong

Neither the rPPG nor MER block has a **trained machine learning model** for engagement classification. Instead, they use hand-crafted mathematical formulas:

**rPPG** (`rppg_block.py`, lines 82–88):
```python
hr_norm = (hr - 72.0) / 10.0
p0 = 1.0 / (1.0 + np.exp(hr_norm + 1.8))
p1 = np.exp(-0.5 * (hr_norm + 0.8)**2)
p2 = np.exp(-0.5 * (hr_norm - 0.2)**2) * 1.5    # ← 1.5× bias toward class 2
p3 = 1.0 / (1.0 + np.exp(-hr_norm + 0.5)) * 1.5
```

**MER** (`mer_block.py`, lines 156–163):
```python
motion_metric = (flow_noise * 2.0 + flow_energy * 5.0)
p0 = np.exp(-0.5 * (motion_metric - 0.1)**2)
p1 = np.exp(-0.5 * (motion_metric - 0.3)**2)
p2 = np.exp(-0.5 * (motion_metric - 0.6)**2) * 1.5  # ← 1.5× bias toward class 2
p3 = np.exp(-0.5 * (motion_metric - 1.1)**2) * 1.5
```

These formulas:
- Have **no theoretical basis** linking HR or optical flow to DAiSEE engagement labels
- Are **biased toward Class 2** (the `* 1.5` multiplier)
- Cannot adapt to different datasets or distributions
- Were never validated against any ground truth labels

### How to Fix

**Replace the heuristics with trained lightweight classifiers.** Since rPPG and MER features are low-dimensional, even a simple SVM or small MLP will dramatically outperform the heuristics.

**For rPPG — Train an SVM on HR + quality features:**

```python
# In a new file: training/train_rppg_classifier.py
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# Extract features from training set
features = []  # Each row: [hr, snr, peak_prominence, hr_stability]
labels = []    # Engagement labels

for clip_path, label in train_data:
    result = rppg_block.run(clip_path)
    hr = result['predictions']['hr']
    q = result['quality_metrics']
    features.append([hr, q['snr'], q['peak_prominence'], q['hr_stability']])
    labels.append(label)

X = np.array(features)
y = np.array(labels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = SVC(kernel='rbf', class_weight='balanced', probability=True)
clf.fit(X_scaled, y)

joblib.dump({'scaler': scaler, 'model': clf}, 'weights/rppg/classifier.pkl')
```

Then update `rppg_block.py` to load and use this classifier instead of the heuristic.

**For MER — Train an MLP on Bi-WOOF features:**

```python
# In a new file: training/train_mer_classifier.py
from sklearn.neural_network import MLPClassifier

# Extract Bi-WOOF features from training set
features = []  # Each row: 8-dim Bi-WOOF descriptor + quality metrics
labels = []

for clip_path, label in train_data:
    result = mer_block.run(clip_path)
    feat = result['features']  # 8-dim Bi-WOOF
    q = result['quality_metrics']
    combined = np.concatenate([feat, [q['flow_noise'], q['blur_estimate']]])
    features.append(combined)
    labels.append(label)

clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, class_weight='balanced')
clf.fit(np.array(features), np.array(labels))

joblib.dump(clf, 'weights/mer/classifier.pkl')
```

---

## Problem 4: Early Fusion Uses Random Projection — Never Trained

### What's Wrong

The `EarlyFusion` class (`fusion/early_fusion.py`, lines 25–30) creates its projection weights using random numbers:

```python
rng = np.random.RandomState(42)
self.proj_weights = {
    mod: rng.randn(dim, 4) * 0.05 for mod, dim in self.modality_dims.items()
}
self.bias = np.zeros(4, dtype=np.float64)
```

Despite the `fusion.yaml` config specifying training hyperparameters (`learning_rate: 1e-3`, `epochs: 10`, `optimizer: adam`), there is **no training loop** anywhere in the codebase for early fusion. The config values are completely ignored.

The projection is effectively random noise → predictions are random → 0.22% accuracy (same as Macro, since both collapse to Class 0).

### How to Fix

**Implement an actual training loop for the early fusion head.** Create a new training script:

```python
# training/train_early_fusion.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TrainableEarlyFusion(nn.Module):
    def __init__(self, input_dim=1152, num_classes=4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# Training loop
model = TrainableEarlyFusion()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# Use class-weighted cross-entropy to handle imbalance
class_weights = torch.tensor([445.0, 21.2, 2.02, 2.19])  # inverse frequency
criterion = nn.CrossEntropyLoss(weight=class_weights)

for epoch in range(10):
    for features_concat, labels in train_loader:
        optimizer.zero_grad()
        logits = model(features_concat)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'weights/fusion/early_fusion.pt')
```

Then update `early_fusion.py` to load the trained weights instead of random projection.

**Also fix the feature dimension mismatch:** The current code expects rPPG=128, MER=256 but the actual features are BVP signal (variable length) and 8-dim Bi-WOOF. Either:
- Standardize feature extraction to output fixed-size vectors, or
- Use the pooled features (e.g., mean/std of BVP → fixed 4-dim vector for rPPG)

---

## Problem 5: Macro Checkpoint Loaded with strict=False — Mismatched Weights

### What's Wrong

In `macro_block.py` (lines 130–141), the trained checkpoint is loaded with `strict=False`:

```python
clean_sd = {k.replace("block.", "").replace("model.", ""): v for k, v in sd.items()}
self.load_state_dict(clean_sd, strict=False)
```

`strict=False` means PyTorch will **silently skip** any keys that don't match. The key-cleaning logic (`replace("block.", "")`) is a guess and may not correctly map the checkpoint's key names to the model's parameter names.

If the critical classification heads (`head_engagement`, `head_boredom`, etc.) don't receive their trained weights, they remain **randomly initialized** — producing near-random outputs even when videos are successfully processed.

There is also no logging of which keys matched and which were skipped, making this invisible.

### How to Fix

**Step 1 — Add diagnostic logging to verify checkpoint loading:**

```python
# Replace lines 136-138 in macro_block.py with:
clean_sd = {k.replace("block.", "").replace("model.", ""): v for k, v in sd.items()}

# Log matching information
model_keys = set(self.state_dict().keys())
ckpt_keys = set(clean_sd.keys())
matched = model_keys & ckpt_keys
missing_in_ckpt = model_keys - ckpt_keys
unexpected = ckpt_keys - model_keys

print(f"Checkpoint loading report:")
print(f"  Matched keys: {len(matched)}/{len(model_keys)}")
print(f"  Missing in checkpoint (using random init): {missing_in_ckpt}")
print(f"  Unexpected keys in checkpoint (ignored): {unexpected}")

# Check if classification heads loaded
critical_keys = ['head_engagement.weight', 'head_engagement.bias']
for k in critical_keys:
    if k not in matched:
        print(f"  ⚠️  CRITICAL: {k} was NOT loaded from checkpoint!")

self.load_state_dict(clean_sd, strict=False)
```

**Step 2 — Fix the key mapping by inspecting the checkpoint:**

```python
# Debug script to inspect what's in the checkpoint
import torch
ckpt = torch.load('weights/macro/best.ckpt', map_location='cpu')
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    ckpt = ckpt['state_dict']
for k in sorted(ckpt.keys()):
    print(k, ckpt[k].shape)
```

Then adjust the key-cleaning logic to match exactly.

**Step 3 — Use `strict=True` once the key mapping is correct** to catch any future mismatches immediately.

---

## Problem 6: Behavioral Features Are Meaningless — Same Number Repeated 30 Times

### What's Wrong

In `macro_block.py` (lines 181–206), the `_behavior_features()` method computes a single geometric ratio and then **duplicates it 30 times**:

```python
def _behavior_features(self, frame):
    # ... compute eye_distance and nose_mouth_dist ...
    ratio = eye_distance / (nose_mouth_dist + 1e-6)
    feats = np.full(30, ratio, dtype=np.float32)  # Same value × 30
    return feats
```

This 30-dimensional vector carries **exactly 1 dimension of information** (the ratio). The other 29 dimensions are redundant copies. The behavioral stream (which feeds into a 64-dim projection) adds virtually nothing to the model.

The README describes rich behavioral features: gaze direction, head pose, Action Units (AUs). None of these are actually computed.

### How to Fix

**Implement the behavioral features described in the README:**

```python
def _behavior_features(self, frame: np.ndarray) -> np.ndarray:
    """Extract meaningful behavioral features from facial landmarks."""
    if self.mp_face_mesh is None:
        return np.zeros(30, dtype=np.float32)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = self.mp_face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return np.zeros(30, dtype=np.float32)

    lm = results.multi_face_landmarks[0].landmark
    feats = []

    # 1. Eye Aspect Ratios (EAR) — detect blinks/drowsiness (2 features)
    # Left eye: landmarks 33, 160, 158, 133, 153, 144
    left_ear = self._eye_aspect_ratio(lm, [33, 160, 158, 133, 153, 144])
    # Right eye: landmarks 362, 385, 387, 263, 373, 380
    right_ear = self._eye_aspect_ratio(lm, [362, 385, 387, 263, 373, 380])
    feats.extend([left_ear, right_ear])

    # 2. Mouth Aspect Ratio (MAR) — detect yawning (1 feature)
    mouth_top = np.array([lm[13].x, lm[13].y])
    mouth_bottom = np.array([lm[14].x, lm[14].y])
    mouth_left = np.array([lm[78].x, lm[78].y])
    mouth_right = np.array([lm[308].x, lm[308].y])
    mar = np.linalg.norm(mouth_top - mouth_bottom) / (np.linalg.norm(mouth_left - mouth_right) + 1e-6)
    feats.append(mar)

    # 3. Head Pose — yaw, pitch, roll (3 features)
    nose_tip = np.array([lm[1].x, lm[1].y, lm[1].z])
    chin = np.array([lm[152].x, lm[152].y, lm[152].z])
    left_ear_pt = np.array([lm[234].x, lm[234].y, lm[234].z])
    right_ear_pt = np.array([lm[454].x, lm[454].y, lm[454].z])
    # Compute head pose angles
    yaw = np.arctan2(right_ear_pt[2] - left_ear_pt[2], right_ear_pt[0] - left_ear_pt[0])
    pitch = np.arctan2(chin[1] - nose_tip[1], chin[2] - nose_tip[2])
    roll = np.arctan2(right_ear_pt[1] - left_ear_pt[1], right_ear_pt[0] - left_ear_pt[0])
    feats.extend([yaw, pitch, roll])

    # 4. Eyebrow raise distance (2 features)
    left_brow = np.array([lm[70].x, lm[70].y])
    left_eye_top = np.array([lm[159].x, lm[159].y])
    right_brow = np.array([lm[300].x, lm[300].y])
    right_eye_top = np.array([lm[386].x, lm[386].y])
    feats.append(np.linalg.norm(left_brow - left_eye_top))
    feats.append(np.linalg.norm(right_brow - right_eye_top))

    # 5. Gaze direction proxy — iris position relative to eye corners (4 features)
    # Left iris center = landmark 468, Right iris center = landmark 473
    if len(lm) > 473:
        left_iris = np.array([lm[468].x, lm[468].y])
        right_iris = np.array([lm[473].x, lm[473].y])
        left_eye_center = (np.array([lm[33].x, lm[33].y]) + np.array([lm[133].x, lm[133].y])) / 2
        right_eye_center = (np.array([lm[362].x, lm[362].y]) + np.array([lm[263].x, lm[263].y])) / 2
        feats.extend([
            left_iris[0] - left_eye_center[0],   # horizontal gaze left eye
            left_iris[1] - left_eye_center[1],   # vertical gaze left eye
            right_iris[0] - right_eye_center[0],  # horizontal gaze right eye
            right_iris[1] - right_eye_center[1],  # vertical gaze right eye
        ])
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    # 6. Additional facial distances for expression analysis (16 features)
    key_landmarks = [1, 33, 61, 133, 159, 263, 291, 362, 386, 13, 14, 78, 308, 152, 10, 234]
    for i in range(len(key_landmarks)):
        for j in range(i+1, min(i+3, len(key_landmarks))):
            li = np.array([lm[key_landmarks[i]].x, lm[key_landmarks[i]].y])
            lj = np.array([lm[key_landmarks[j]].x, lm[key_landmarks[j]].y])
            feats.append(np.linalg.norm(li - lj))
            if len(feats) >= 30:
                break
        if len(feats) >= 30:
            break

    # Pad or truncate to exactly 30
    feats = feats[:30]
    while len(feats) < 30:
        feats.append(0.0)

    return np.array(feats, dtype=np.float32)

def _eye_aspect_ratio(self, landmarks, indices):
    """Compute Eye Aspect Ratio from 6 landmark indices."""
    pts = [np.array([landmarks[i].x, landmarks[i].y]) for i in indices]
    vertical1 = np.linalg.norm(pts[1] - pts[5])
    vertical2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    return (vertical1 + vertical2) / (2.0 * horizontal + 1e-6)
```

---

## Problem 7: Fusion Cascade Failure — Garbage In, Garbage Out

### What's Wrong

Since the individual modality blocks produce broken outputs, all fusion strategies inherit those failures:

| Fusion Method | What Happens | Result |
|---------------|-------------|--------|
| **Early Fusion** | Concatenates broken features + random projection | 0.22% (collapses to Class 0) |
| **Static Late Fusion** | Weighted average of `macro=[0,0,0,0]` + `rppg=[fixed]` + `mer=[noisy]` | 49.6% (follows rPPG) |
| **Agentic Late Fusion** | Macro weight → 0 (failed), rPPG weight → boosted (fake SNR=10.0 passes threshold), MER gets downweighted | 49.4% (becomes rPPG-only) |

The Agentic fusion is specifically designed to upweight reliable modalities, but since the rPPG dummy reports **fake high-quality metrics** (SNR=10.0, which exceeds the 5.0 threshold), the agentic system incorrectly trusts it the most. Meanwhile, Macro reports no quality metrics at all (because it failed), so it gets zero weight.

### How to Fix

**Fix the individual blocks first** (Problems 1–6 above). Then:

**Step 1 — Add validation in agentic fusion to detect degenerate inputs:**

```python
# In agentic_fusion.py, add after computing probabilities:
def _is_degenerate(self, prob):
    """Detect if a modality is outputting constant/degenerate predictions."""
    entropy = -np.sum(prob * np.log(prob + 1e-10))
    max_entropy = np.log(len(prob))
    # If entropy is very low (< 20% of max), the predictions are nearly constant
    if entropy < 0.2 * max_entropy:
        return True
    return False

# In run(), before fusion:
for mod in ["macro", "rppg", "mer"]:
    if mod in preds:
        prob = self._extract_prob(preds[mod])
        if self._is_degenerate(prob):
            weights[mod] = 0.0
            explanations.append(f"{mod} predictions degenerate → weight set to 0")
```

**Step 2 — Add quality metric validation** to detect dummy/placeholder values:

```python
# Reject quality metrics that are suspiciously perfect
if snr == 10.0 and quality.get("rppg", {}).get("peak_prominence") == 0.8:
    explanations.append("rPPG quality metrics appear to be placeholder values")
    weights["rppg"] = self.min_weight
```

**Step 3 — After all blocks are fixed, retune the fusion weights** on the validation set.

---

## Bonus: Severe Class Imbalance in Test Set

### What's Wrong

The test set class distribution is extremely imbalanced:

| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| 0 | Very Low Engagement | 4 | 0.22% |
| 1 | Low Engagement | 84 | 4.71% |
| 2 | Engaged | 882 | 49.44% |
| 3 | Highly Engaged | 814 | 45.63% |

Class 0 has only **4 samples** in 1784. Any model that ignores Class 0 and 1 can still get ~49% accuracy by predicting Class 2 for everything — which is exactly what's happening.

### How to Fix

**During training:**
1. Use **class-weighted cross-entropy loss** (already conceptually in training config but needs verification):
   ```python
   # Compute weights inversely proportional to class frequency
   class_counts = [count_0, count_1, count_2, count_3]
   total = sum(class_counts)
   weights = [total / (4 * c) for c in class_counts]
   criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))
   ```

2. Use **oversampling** for minority classes (SMOTE or random oversampling):
   ```python
   from imblearn.over_sampling import RandomOverSampler
   ros = RandomOverSampler(random_state=42)
   X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
   ```

**During evaluation:**
1. Use **Macro-F1** as the primary metric instead of accuracy (already computed but not prioritized)
2. Report **per-class F1** to catch classes being completely ignored
3. Consider using **balanced accuracy** which averages recall across classes

---

## Priority Summary

| Priority | Problem | Impact | Effort |
|----------|---------|--------|--------|
| 🔴 P0 | Fix video file paths (Problem 2) | Macro block completely non-functional | Low — update CSV paths and config |
| 🔴 P0 | Install real rPPG library (Problem 1) | rPPG outputs are identical for all videos | Medium — install `pyVHR` or implement GREEN method |
| 🔴 P0 | Train rPPG & MER classifiers (Problem 3) | Both use arbitrary heuristics, no learning | Medium — train SVM/MLP on extracted features |
| 🔴 P0 | Train Early Fusion head (Problem 4) | Early fusion uses random noise weights | Medium — implement training loop |
| 🟡 P1 | Fix checkpoint loading (Problem 5) | Macro classification heads may be random | Low — add logging, verify key mapping |
| 🟡 P1 | Implement real behavioral features (Problem 6) | 30-dim vector carries 1 dim of info | Medium — implement EAR, MAR, gaze, head pose |
| 🟡 P1 | Add degenerate input detection (Problem 7) | Fusion trusts broken modalities | Low — add entropy checks |
| 🟢 P2 | Handle class imbalance (Bonus) | Model ignores minority classes | Low — add class weights to loss |

> **Recommended fix order:** Problem 2 → Problem 1 → Problem 5 → Problem 3 → Problem 6 → Problem 4 → Problem 7 → Bonus
>
> Start with the data path fix (instant improvement), then the rPPG library, then verify the macro checkpoint is loading correctly. Once all blocks produce valid per-video outputs, train the classifiers and fusion head.
