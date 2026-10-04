# Project Information: Engagement-MAS

## 1. Project Overview

**Engagement-MAS** is a multi-modal perception and multi-agent fusion system designed to estimate human engagement levels (4 ordinal classes: 0 = Very Low, 1 = Low, 2 = High, 3 = Very High) from facial video clips, evaluated on the **DAiSEE** dataset.

### End-to-End Architecture
1. **Parallel Perception**:
   - **Macro Block (`blocks/macro/macro_block.py`)**: Facial appearance via frozen DINOv2 ViT-B/14 + geometric behavior features + temporal Transformer.
   - **rPPG Block (`blocks/rppg/rppg_block.py`)**: Remote photoplethysmography extracting blood volume pulse (BVP) and heart rate from subtle skin color variations.
   - **MER Block (`blocks/mer/mer_block.py`)**: Micro-expression recognition capturing rapid facial motion using Farneback optical flow and Bi-WOOF descriptors around apex frames.
2. **Thread-Safe Synchronization Gate (`gate/sync_gate.py`)**:
   - Manages asynchronous modality arrival times per clip.
   - Uses thread locks (`threading.Lock`) to safely handle parallel execution from `ThreadPoolExecutor`.
   - Releases bundles immediately when all modalities arrive or flushes partial bundles upon timeout (default 60s) or pipeline finalization.
3. **Fusion Decision Layer**:
   - **Early Fusion (`fusion/early_fusion.py`)**: Concatenates feature representations and projects to engagement logits.
   - **Static Late Fusion (`fusion/static_late_fusion.py`)**: Fixed weighted average of unimodal probability distributions (Macro: 0.5, rPPG: 0.3, MER: 0.2).
   - **Agentic Late Fusion (`fusion/agentic_fusion.py`)**: Dynamic weighting based on real-time quality metrics (Macro blur, rPPG SNR, MER flow noise) with temporal moving-average smoothing.
4. **Training & Evaluation**:
   - PyTorch Lightning training pipeline (`training/train_macro.py`) with embedding caching.
   - Full evaluation suite (`evaluation/evaluate.py`), robustness masking (`evaluation/robustness.py`), and statistical significance tests (`evaluation/significance.py`).

---

## 2. Standardized Modality Interface

Every perception block (`MacroBlock`, `RPPGBlock`, `MERBlock`) implements `run(clip_path: str) -> dict` returning a standardized dictionary:

| Key | Type | Description |
|---|---|---|
| `clip_id` | `str` | Deterministic SHA-256 hash (16 chars) derived from clip path. |
| `modality` | `str` | Name of modality: `"macro"`, `"rppg"`, or `"mer"`. |
| `features` | `np.ndarray` | Numerical feature representation (`832`-dim for Macro, BVP signal array for rPPG, `8`-dim Bi-WOOF for MER). |
| `predictions` | `dict` | Prediction dictionary (e.g. `{"engagement": [p0, p1, p2, p3]}` or `{"hr": 72.0}`). |
| `quality_metrics` | `dict` | Modality-specific quality indicators (e.g. `blur`, `face_detection_rate`, `snr`, `flow_noise`). |
| `timestamps` | `list` | Frame or time indices sampled during processing. |
| `status` | `str` | `"success"` or error message if clip processing failed. |

---

## 3. Detailed File-by-File Breakdown

### Perception Modules (`engagement-mas/blocks/`)

#### [`macro_block.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/blocks/macro/macro_block.py)
* **Purpose**: Macro-level facial appearance and behavioral movement extraction.
* **Mechanism**:
  - Uses MediaPipe FaceMesh to detect facial landmarks and crop face ROIs across sampled video frames.
  - Passes face ROIs through a frozen **DINOv2 ViT-B/14** backbone (`timm` or PyTorch Hub with fallback) to extract 768-dim appearance embeddings.
  - Extracts 30 geometric facial distances/angles (eye distance, nose-mouth ratio, eyebrow positions) projected to 64 dimensions.
  - Concatenates appearance and behavior (`832` dimensions) into a 2-layer temporal Transformer encoder.
  - Four multi-task linear classification heads output logits for `engagement`, `boredom`, `confusion`, and `frustration`.
  - Computes quality metrics: `face_detection_rate`, `blur` (Laplacian variance), `brightness`, and `motion_smoothness`.
* **Input**: Video file path (`.avi`, `.mp4`).
* **Output**: Standard dictionary with `832`-dim feature vector, 4-class probabilities for each affective state, and quality metrics.

#### [`rppg_block.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/blocks/rppg/rppg_block.py)
* **Purpose**: Remote physiological signal monitoring (heart rate from facial blood volume pulse).
* **Mechanism**:
  - Detects facial skin regions and tracks subtle green-channel intensity changes corresponding to cardiac pulses.
  - Wraps `Face2PPG` when installed; otherwise falls back gracefully to a deterministic synthetic pulse model based on clip hash.
  - Computes pulse quality metrics: signal-to-noise ratio (`snr`), `peak_prominence`, and `hr_stability`.
* **Input**: Video file path.
* **Output**: Standard dictionary with raw BVP signal (`features`), heart rate in bpm (`predictions: {"hr": float}`), and SNR quality metrics.

#### [`mer_block.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/blocks/mer/mer_block.py)
* **Purpose**: Micro-expression recognition detecting transient, involuntary facial movements.
* **Mechanism**:
  - Crops face bounding boxes across sampled frames.
  - Computes dense Farneback optical flow between successive frames.
  - Identifies the apex frame (peak optical flow magnitude) representing micro-expression onset/apex.
  - Extracts an 8-bin **Bi-WOOF** (Binarized Weighting of Oriented Optical Flow) descriptor capturing subtle directionality.
  - Computes motion quality metrics: `flow_noise`, `track_stability`, `framerate_factor`, and `blur_estimate`.
* **Input**: Video file path.
* **Output**: Standard dictionary with 8-dim Bi-WOOF descriptor (`features`), quality metrics, and status.

---

### Synchronization Gate (`engagement-mas/gate/`)

#### [`sync_gate.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/gate/sync_gate.py)
* **Purpose**: Aggregates asynchronous perception results per video clip and emits synchronized modality bundles.
* **Mechanism**:
  - **Thread-safe**: Protected with `threading.Lock` across all state mutations (`receive`, `check_and_release`, `finalize`, `check_timeout`).
  - **Early Release**: Automatically triggers bundle emission as soon as all 3 modalities (`macro`, `rppg`, `mer`) arrive for a clip.
  - **Timeout Protection**: Emits partial bundles if modalities exceed the timeout window (default 60 seconds from `configs/gate.yaml`).
  - **Deduplication & Late Arrival Handling**: Drops duplicate results for the same modality and logs late arrivals after a bundle has closed.
* **Input**: Modality result dictionaries fed via `gate.receive(result)`.
* **Output**: Synchronized bundle dictionary containing `clip_id`, `features`, `predictions`, `quality_metrics`, and `availability` mask.

---

### Fusion Strategies (`engagement-mas/fusion/`)

#### [`early_fusion.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/fusion/early_fusion.py)
* **Purpose**: Feature-level early fusion baseline.
* **Mechanism**: Concatenates available feature vectors (Macro 768/832, rPPG 128, MER 256) with zero-padding for missing modalities, applies a seeded linear projection matrix to 4 classes, and computes softmax probabilities.
* **Input**: Synchronized bundle from `SyncGate`.
* **Output**: Dictionary with key `"engagement"` containing 4-class probability distribution.

#### [`static_late_fusion.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/fusion/static_late_fusion.py)
* **Purpose**: Fixed-weight decision-level late fusion baseline.
* **Mechanism**: Takes class probability outputs from available modalities and computes a fixed weighted sum using configured weights (`macro: 0.5`, `rppg: 0.3`, `mer: 0.2` from `configs/fusion.yaml`), re-normalizing weights if a modality is absent.
* **Input**: Synchronized bundle from `SyncGate`.
* **Output**: Dictionary with key `"engagement"` containing fused 4-class probabilities.

#### [`agentic_fusion.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/fusion/agentic_fusion.py)
* **Purpose**: Dynamic multi-agent fusion adjusting modality trust based on signal quality.
* **Mechanism**:
  - **Quality Inspector**: Compares `rppg_snr`, `mer_flow_noise`, and `macro_blur` against thresholds in `configs/fusion.yaml`.
  - **Weight Reasoner**: Increases weight (by `+0.2`) for high-quality modalities and penalizes noisy/blurry modalities (by `-0.2`), bounded between `0.1` and `0.8`.
  - **Temporal Smoother**: Applies a moving average over a sliding window of recent predictions (window size = 5) for temporal stability.
* **Input**: Synchronized bundle from `SyncGate`.
* **Output**: Dictionary with key `"engagement"`, 4-class probabilities, and an audit trail `"explanation"` string.

---

### Training Pipeline (`engagement-mas/training/`)

#### [`train_macro.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/training/train_macro.py)
* **Purpose**: Fine-tunes the temporal Transformer and classification heads of `MacroBlock` using PyTorch Lightning.
* **Key Features**:
  - **Cross-Platform Device Selection**: Automatically detects NVIDIA CUDA, Apple MPS, or CPU fallback.
  - **Disk Caching**: Extracts appearance embeddings via frozen DINOv2 once on Epoch 0 and stores them in `data/cache/*.pt`, speeding up subsequent epochs by 10x+.
  - **Multi-task Loss**: Jointly optimizes engagement cross-entropy alongside boredom, confusion, and frustration auxiliary losses.
  - **Model Checkpointing**: Saves best checkpoint (`best.ckpt`) and final model (`last.pt`) under `weights/macro/`.
* **Input**: `--config configs/training.yaml`, `--dataset daisee`, `--split train`.
* **Output**: Checkpoints in `weights/macro/` and cached tensors in `data/cache/`.

---

### Evaluation Pipeline (`engagement-mas/evaluation/`)

#### [`evaluate.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/evaluation/evaluate.py)
* **Purpose**: Full end-to-end evaluation harness for the entire pipeline.
* **Mechanism**:
  - Loads dataset split (`test`, `val`, `train`, or `test_quick`).
  - Spawns parallel worker threads via `ThreadPoolExecutor` to process clips through perception blocks.
  - Streams results through `SyncGate`, applies the selected fusion model (`early`, `static`, `agentic`), and gathers engagement predictions.
  - Saves CSV results to `evaluation/results/results_<fusion>.csv` and prints Accuracy, Macro-F1, per-class metrics, and confusion matrix.

#### [`metrics.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/evaluation/metrics.py)
* **Purpose**: Classification metrics calculations using `scikit-learn` (Accuracy, Macro-F1, Precision, Recall, and Confusion Matrix).

#### [`robustness.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/evaluation/robustness.py)
* **Purpose**: Stress-tests pipeline reliability by randomly masking out modalities across iterations and logging drop schedules to JSON.

#### [`significance.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/evaluation/significance.py)
* **Purpose**: Runs evaluations across multiple random seeds and executes paired t-tests (`scipy.stats.ttest_rel`) to assess statistical significance between fusion approaches.

---

### Data Management & Verification (`engagement-mas/data/`)

#### [`convert_daisee.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/data/convert_daisee.py)
* Parses raw DAiSEE label text files and video folders into standard CSV files (`train.csv`, `val.csv`, `test.csv`).

#### [`validate_dataset.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/data/validate_dataset.py)
* Validates dataset integrity: checks column names, verifies that every video path in the CSV exists on disk, and prints label distributions.

#### Dataset Split Files (`data/daisee/*.csv` & Root `*.csv`)
* `train.csv` (5,358 clips), `val.csv` (1,429 clips), `test.csv` (1,784 clips), and `test_quick.csv` (10 clips for fast sanity checking).

---

### Configuration Files (`engagement-mas/configs/`)

| Configuration File | Role |
|---|---|
| [`datasets.yaml`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/configs/datasets.yaml) | Relative dataset paths for DAiSEE root directory and split CSVs, label column names, and number of classes (`4`). |
| [`fusion.yaml`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/configs/fusion.yaml) | Static weights (`0.5`, `0.3`, `0.2`), agentic quality thresholds (`rppg_snr: 5.0`, `mer_flow_noise: 0.2`, `macro_blur: 100.0`), and smoothing window. |
| [`gate.yaml`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/configs/gate.yaml) | Synchronization timeout duration (`60s`) and gate logger path. |
| [`training.yaml`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/configs/training.yaml) | Training hyperparameters (epochs: 20, batch size: 8, learning rate: 5e-4, loss weights, checkpoint dir, and device `"auto"`). |

---

### Test Suite (`engagement-mas/tests/`)

* [`test_gate.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/tests/test_gate.py): Validates early emission, timeout handling, duplicate dropping, and late arrival behavior in `SyncGate`.
* [`test_fusion.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/tests/test_fusion.py): Confirms deterministic outputs for Early, Static Late, and Agentic fusion given identical input bundles.
* [`test_determinism.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/tests/test_determinism.py): Ensures identical outputs across repeat runs on the perception blocks.

---

### Cross-Platform & Windows Execution Guides

* [`commands.md`](file:///Users/jeetashwar/Downloads/Baseline_3/commands.md): Quick-reference command cheat sheet for conda setup, CUDA wheel installation, training, and evaluation commands on Windows.
* [`requirements-windows.txt`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/requirements-windows.txt): Clean, platform-compatible pip dependency manifest for Windows (CUDA/CPU).
* [`cloud_run.md`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/cloud_run.md): Instructions for running on Google Colab, Kaggle, or cloud GPU instances.

---

## 4. How to Run & Train on Windows

### Step 1: Environment Setup
In Anaconda Prompt / PowerShell:
```powershell
conda create -n engagement-mas-windows python=3.10 -y
conda activate engagement-mas-windows
cd engagement-mas
pip install -r requirements-windows.txt
```
*(Optional for NVIDIA GPU)*:
```powershell
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Validate Data
```powershell
python -m data.validate_dataset --splits test_quick
```

### Step 3: Train the Macro Block
```powershell
python -m training.train_macro --config configs/training.yaml --dataset daisee --split train
```
The script will auto-detect CUDA GPU if present (otherwise CPU), cache extracted features in `data/cache/`, and save model weights to `weights/macro/best.ckpt`.

### Step 4: Evaluate Fusion
```powershell
# Quick test (10 clips)
python -m evaluation.evaluate --config configs/fusion.yaml --dataset daisee --split test_quick --fusion agentic --max-clips 10

# Full test set evaluation
python -m evaluation.evaluate --config configs/fusion.yaml --dataset daisee --split test --fusion agentic
```
Results and metrics will be printed to console and saved in `evaluation/results/results_agentic.csv`.
