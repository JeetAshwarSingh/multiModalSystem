# Engagement-MAS: Windows Deployment & Changelog Summary

> **Detailed Guide**: Please refer to [DocumentationAndRequirements/CHANGES_AND_WINDOWS_SETUP.md](DocumentationAndRequirements/CHANGES_AND_WINDOWS_SETUP.md) for full architectural documentation.

---

## 🛠️ Summary of What Was Done on the Remote Computer

1. **Problem 1 (rPPG Dummy Fixed)**: Replaced dummy constant 75.0 BPM placeholder with an in-tree **CHROM & Green-channel rPPG signal extractor** in `blocks/rppg/rppg_block.py` with Butterworth bandpass filtering, FFT cardiac peak analysis, and real SNR/prominence/stability metrics.
2. **Problem 2 (Video Paths Fixed)**: Configured DAiSEE root directory detection across local paths and environment variable `DAISEE_ROOT` in `datasets.yaml` and `evaluate.py`.
3. **Problem 3 (rPPG & MER Classifiers Trained)**: Created `training/train_rppg_classifier.py` and `training/train_mer_classifier.py`. Trained balanced classifiers and saved them to `weights/rppg/classifier.pkl` and `weights/mer/classifier.pkl`.
4. **Problem 4 (Early Fusion Head Trained)**: Created `training/train_early_fusion.py`. Trained a deep multimodal projection neural network with class-weighted cross-entropy loss, saved weights to `weights/fusion/early_fusion.pt`, and connected it into `fusion/early_fusion.py`.
5. **Problem 5 (Macro Checkpoint Loading Verified)**: Fixed `torch.load` map_location device compatibility and verified all **208 / 208 checkpoint keys** load into `MacroBlock`.
6. **Problem 6 (Real 30-dim Behavioral Features)**: Implemented full 30-dim facial landmark features (Eye Aspect Ratios, Mouth Aspect Ratio, Head Pose yaw/pitch/roll, eyebrow raise, gaze direction, and expression ratios) in `macro_block.py`.
7. **Problem 7 (Fusion Cascade & Degenerate Detection)**: Implemented Shannon entropy-based degenerate output detection and placeholder quality filtering in `agentic_fusion.py`.
8. **Bonus (Class Imbalance Handled)**: Applied inverse frequency class weights to all training pipelines and added `balanced_accuracy` across metrics and evaluation scripts.
9. **ResultAndAnalysis System**: Every evaluation run now automatically creates a timestamped subdirectory `resultAndAnalysis/run_YYYYMMDD_HHMMSS/` containing all 6 prediction CSVs, summary metrics JSON/CSV, and an `Analysis/` subdirectory with all 6 publication figures and LaTeX tables, while also mirroring the latest results to the top-level directory.

---

## 💻 What to Install to Run the Same Code on Windows

### 1. Python Environment
Install **Python 3.10.x** (64-bit) from [python.org](https://www.python.org/downloads/release/python-31011/) (ensure "Add Python to PATH" is checked).

Create a conda environment:
```powershell
conda create -n engagement-mas python=3.10 -y
conda activate engagement-mas
```

### 2. Dependencies
If you have an **NVIDIA GPU (CUDA)**:
```powershell
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r DocumentationAndRequirements\requirements-windows.txt
```

If you have a **CPU-only machine**:
```powershell
pip install -r DocumentationAndRequirements\requirements-windows.txt
```

### 3. Set Dataset Root Path on Windows (PowerShell)
```powershell
$env:DAISEE_ROOT = "C:\Users\<YourUsername>\Downloads\DAiSEE"
```

### 4. Running the Code on Windows
Navigate into `engagement-mas`:
```powershell
cd engagement-mas

# 1. Verify Dataset
python -m data.validate_dataset --config-file configs\datasets.yaml --splits test_quick

# 2. Run Test Suite
python -m pytest tests\

# 3. Run Evaluation (Generates timestamped results in resultAndAnalysis\)
python -m evaluation.evaluate --config configs\fusion.yaml --dataset daisee --split test_quick --fusion agentic
```
