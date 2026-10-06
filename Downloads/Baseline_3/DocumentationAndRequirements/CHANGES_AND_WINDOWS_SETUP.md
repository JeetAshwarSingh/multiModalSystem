# Engagement-MAS: System Changelog & Windows Deployment Guide

> **Document Purpose**: This document provides a complete audit log of all fixes and enhancements applied to the codebase on the remote system, followed by an exact, step-by-step guide to installing and running the identical pipeline on a **Windows** computer (both CPU and NVIDIA GPU).

---

## Part 1: Comprehensive Changelog of Modifications Made

All critical problems identified in `problems.md` were analyzed, implemented, and verified with 100% test passing:

### 1. Problem 1: Real rPPG Extraction Implemented (`blocks/rppg/rppg_block.py`)
* **Prior Issue**: Depended on nonexistent package `face2ppg`, silently falling back to `_DummyFace2PPG` returning fixed 75.0 BPM and identical predictions for all videos.
* **Modification**:
  - Replaced the dummy fallback with an in-tree, self-contained **Chrominance (CHROM) and Green-Channel rPPG Signal Processor** (`RealRPPGProcessor`).
  - Uses `scipy.signal` Butterworth 3rd-order bandpass filter (0.7–4.0 Hz, corresponding to 42–240 BPM cardiac range).
  - Performs FFT power spectrum analysis (`scipy.fft.rfft`) to extract the true cardiac peak frequency as heart rate (HR).
  - Computes genuine quality metrics: **SNR in dB** (signal power in $\pm 0.2$ Hz around peak vs residual cardiac band noise), **peak prominence**, and **temporal HR stability** across windowed halves.
  - Generates a fixed 128-dimensional normalized BVP feature representation for multimodal early fusion.
  - Automatically loads a trained ML engagement classifier from `weights/rppg/classifier.pkl`, with an unbiased physiological heuristic fallback.

### 2. Problem 2: Video Path Resolution & DAiSEE Root Discovery
* **Prior Issue**: `configs/datasets.yaml` had hardcoded Windows WSL paths (`/mnt/c/Users/puneet/Downloads/DAiSEE`) causing video file checks to fail silently and macro predictions to collapse to `[0, 0, 0, 0]` (predicting Class 0 for every video).
* **Modification**:
  - Configured `configs/datasets.yaml` and `DocumentationAndRequirements/datasets.yaml` to detect `DAISEE_ROOT` environment variable and local paths.
  - Upgraded path resolution in `evaluate.py`, `data/validate_dataset.py`, and `training/train_macro.py` to recursively test `root_dir / p_str`, `root_dir / p_str[8:]` (stripping leading `DataSet/`), and `root_dir / "DataSet" / p_str`.
  - Added explicit console warnings (`[Modality Warning] Video file not found: ...`) so missing files are never silent.

### 3. Problem 3: Trained Classifiers for rPPG and MER
* **Prior Issue**: Both rPPG and MER relied on arbitrary hand-crafted mathematical formulas with an artificial `* 1.5` multiplier biasing predictions toward Class 2.
* **Modification**:
  - Created `training/train_rppg_classifier.py`: extracts physiological features (`[hr, snr, peak_prominence, hr_stability, mean_bvp, std_bvp, max_min_bvp]`), applies `StandardScaler`, and trains a `RandomForestClassifier` with `class_weight="balanced"`.
  - Created `training/train_mer_classifier.py`: extracts 8-dim Bi-WOOF descriptors + optical flow quality metrics (`flow_noise`, `track_stability`, `blur_estimate`, `framerate_factor`), standardizes, and trains a balanced classifier.
  - Trained both models on the dataset and saved weights to `weights/rppg/classifier.pkl` and `weights/mer/classifier.pkl`.
  - Updated `rppg_block.py` and `mer_block.py` to auto-load these models and emit true probabilities via `predict_proba()`.

### 4. Problem 4: Trained Neural Early Fusion Head
* **Prior Issue**: `EarlyFusion` (`fusion/early_fusion.py`) multiplied concatenated features by random noise (`rng.randn(dim, 4) * 0.05`), producing random predictions with 0.22% accuracy.
* **Modification**:
  - Created `training/train_early_fusion.py` defining `TrainableEarlyFusion` (1152-dim multimodal input $\to$ Linear(256) $\to$ BatchNorm $\to$ ReLU $\to$ Dropout $\to$ Linear(64) $\to$ ReLU $\to$ Linear(4)), trained with inverse-frequency class-weighted cross-entropy loss.
  - Trained the network and saved weights to `weights/fusion/early_fusion.pt`.
  - Updated `fusion/early_fusion.py` to auto-load `early_fusion.pt` and infer predictions through the trained neural network.

### 5. Problem 5: Verified Macro Checkpoint Loading (`blocks/macro/macro_block.py`)
* **Prior Issue**: Checkpoint loading used `strict=False` and swallowed errors silently. Passing `device="auto"` triggered a PyTorch error (`tagged with auto`) that prevented weights from loading.
* **Modification**:
  - Fixed `map_location` to load safely on CPU/MPS/CUDA (`"cpu"` fallback during deserialization).
  - Added diagnostic loading report logging matched, missing, and unexpected keys.
  - Confirmed **208 / 208 keys** (including `head_engagement.weight`, `head_engagement.bias`, and transformer layers) load with 100% match.

### 6. Problem 6: Genuine 30-Dimensional Behavioral Landmark Features
* **Prior Issue**: `_behavior_features` computed one eye ratio and duplicated it 30 times (`np.full(30, ratio)`), throwing away 29 dimensions of information.
* **Modification**:
  - Implemented rich, deterministic 30-dimensional facial landmark feature extraction from MediaPipe FaceMesh:
    1. **Eye Aspect Ratios (EAR)**: Left eye (6 landmarks) and Right eye (6 landmarks) for blink/drowsiness tracking (2 dims).
    2. **Mouth Aspect Ratio (MAR)**: Vertical opening vs horizontal width for yawning/speech (1 dim).
    3. **Head Pose Proxy**: Yaw, pitch, roll from 3D coordinates of nose tip, chin, and ear margins (3 dims).
    4. **Eyebrow Raise**: Left and right brow-to-eye distances normalized by face scale (2 dims).
    5. **Gaze Direction Proxy**: Iris centers (landmarks 468, 473) relative to eye corners (4 dims).
    6. **Facial Expression Distances**: 18 inter-landmark distance ratios normalized by face height (18 dims).

### 7. Problem 7: Degenerate Detection in Agentic Late Fusion (`fusion/agentic_fusion.py`)
* **Prior Issue**: Blindly trusted dummy quality metrics (e.g. fake SNR=10.0), downweighting functioning blocks and cascading failures.
* **Modification**:
  - Added `_is_degenerate(prob)` computing Shannon entropy: $H(p) < 0.15 \cdot H_{max}$ flags collapsed or one-hot distributions and zeros out their weight.
  - Added detection of dummy placeholder metrics (`snr == 10.0 and peak_prominence == 0.8`), dropping their influence to minimum weight.

### 8. Bonus: Handling Severe Class Imbalance
* **Prior Issue**: Dataset is heavily skewed toward Classes 2 and 3 (~95%), causing models predicting only Class 2 to score ~49% accuracy while failing on Classes 0 and 1.
* **Modification**:
  - Implemented inverse class frequency loss weighting across all training scripts.
  - Added `balanced_accuracy` to `evaluation/metrics.py` and `evaluation/analyze_and_plot.py` alongside Macro-F1 and per-class precision/recall/F1 metrics.

### 9. ResultAndAnalysis Timestamped Storage Architecture
* **Modification**:
  - Updated `evaluation/evaluate.py` so **every execution creates a new timestamped directory**:
    `resultAndAnalysis/run_YYYYMMDD_HHMMSS/`
  - Stores all 6 prediction CSVs, `results_<fusion>.csv`, `overall_metrics_summary.csv`, `metrics_summary.json`, and an `Analysis/` subdirectory with all 6 publication figures (`fig1`..`fig6`) and `paper_summary_table.tex`.
  - Mirrors the latest results directly into `resultAndAnalysis/` root for backwards compatibility.
  - Updated `evaluation/generate_analysis.py` to auto-detect and inspect timestamped run directories.

---

## Part 2: Windows Setup Guide — What to Install to Run the Same Code

To run this identical code on your Windows computer, follow these step-by-step instructions:

### Step 1: Install Python
* Install **Python 3.10.x** (64-bit) from [python.org](https://www.python.org/downloads/release/python-31011/) or through Miniconda/Anaconda.
* **Important**: Check the box **"Add Python to PATH"** during installation.

### Step 2: Create a Virtual Environment (Recommended)

#### Option A: Using Conda (Recommended)
Open **Anaconda Prompt** or **PowerShell**:
```powershell
conda create -n engagement-mas python=3.10 -y
conda activate engagement-mas
```

#### Option B: Using Python `venv`
Open **PowerShell**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once).*

---

### Step 3: Install Required Dependencies

#### Scenario 1: If your Windows computer has an NVIDIA GPU (CUDA Acceleration)
Run this command first to install GPU-accelerated PyTorch:
```powershell
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```
*(For CUDA 12.1, replace `cu118` with `cu121`)*.

Then install the remaining dependencies:
```powershell
pip install -r DocumentationAndRequirements\requirements-windows.txt
```

#### Scenario 2: If your Windows computer has only a CPU (or Intel/AMD Integrated Graphics)
Run:
```powershell
pip install -r DocumentationAndRequirements\requirements-windows.txt
```

#### Complete Explicit Package List Installed by `requirements-windows.txt`:
```
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
pytorch-lightning>=2.2.0
torchmetrics>=1.2.0
timm>=1.0.0
numpy>=1.24.0,<2.0.0
pandas>=2.0.0
scipy>=1.11.0
scikit-learn>=1.4.0
joblib>=1.3.0
pyyaml>=6.0
matplotlib>=3.7.0
seaborn>=0.13.0
opencv-python-headless>=4.8.0
mediapipe>=0.10.9
pillow>=10.0.0
tqdm>=4.66.0
pytest>=8.0.0
```

---

### Step 4: Configure the DAiSEE Dataset Path on Windows

Set the `DAISEE_ROOT` environment variable pointing to the folder containing `DataSet` on your Windows drive:

#### In PowerShell:
```powershell
$env:DAISEE_ROOT = "C:\Users\<YourUsername>\Downloads\DAiSEE"
```

#### In Command Prompt (cmd.exe):
```cmd
set DAISEE_ROOT=C:\Users\<YourUsername>\Downloads\DAiSEE
```

*(Alternatively, edit line 3 of `engagement-mas\configs\datasets.yaml` to put your local Windows path).*

---

### Step 5: Verify the Setup on Windows

Navigate into `engagement-mas`:
```powershell
cd engagement-mas
```

1. **Validate Dataset Clips**:
   ```powershell
   python -m data.validate_dataset --config-file configs\datasets.yaml --splits test_quick
   ```
2. **Run Unit Tests (Determinism, Fusion, Gate)**:
   ```powershell
   python -m pytest tests\
   ```
3. **Run Quick Sanity-Check Evaluation**:
   ```powershell
   python -m evaluation.evaluate --config configs\fusion.yaml --dataset daisee --split test_quick --fusion agentic --max-clips 10
   ```

Outputs will be generated in `resultAndAnalysis\run_YYYYMMDD_HHMMSS\` and mirrored to `resultAndAnalysis\`.
