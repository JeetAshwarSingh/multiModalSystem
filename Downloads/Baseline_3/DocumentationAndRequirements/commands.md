# Training & Evaluation Commands for Engagement‑MAS (Windows & Cross-Platform)

## 1️⃣ Environment Setup (Once)

### Using Conda (Recommended on Windows)
```powershell
# In PowerShell or Anaconda Prompt:
conda create -n engagement-mas python=3.10 -y
conda activate engagement-mas
```

### Using Python venv
```powershell
# In PowerShell:
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

## 2️⃣ Install Dependencies

### For Windows with NVIDIA GPU (CUDA Acceleration)
```powershell
# Step 1: Install PyTorch with matching CUDA 11.8 build
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# Step 2: Install remaining required packages
pip install -r DocumentationAndRequirements\requirements-windows.txt
```
*(For CUDA 12.1, replace `cu118` with `cu121`)*.

### For Windows CPU-Only
```powershell
pip install -r DocumentationAndRequirements\requirements-windows.txt
```

---

## 3️⃣ Set DAiSEE Dataset Root Directory

Point `DAISEE_ROOT` to the directory on your drive that contains the `DataSet` folder:

### PowerShell
```powershell
$env:DAISEE_ROOT = "C:\Users\<YourUsername>\Downloads\DAiSEE"
```

### Windows Command Prompt (cmd.exe)
```cmd
set DAISEE_ROOT=C:\Users\<YourUsername>\Downloads\DAiSEE
```

*(Alternatively, edit `engagement-mas\configs\datasets.yaml` directly).*

---

## 4️⃣ Validate Dataset & Verify Test Suite

Always run dataset validation and unit tests to ensure all video clips resolve and modules are functional:

```powershell
cd engagement-mas

# Validate clips on test_quick split:
python -m data.validate_dataset --config-file configs\datasets.yaml --splits test_quick

# Run all 10 unit tests (Determinism, Fusion, SyncGate):
python -m pytest tests\
```

---

## 5️⃣ Training Commands

All training pipelines support balanced class weighting and cross-platform execution (CUDA / MPS / CPU).

### Train Macro Appearance & Behavioral Temporal Transformer
```powershell
python -m training.train_macro `
    --config configs\training.yaml `
    --dataset daisee `
    --split train
```

### Train rPPG Engagement Classifier (Random Forest + Balanced Weights)
```powershell
python -m training.train_rppg_classifier `
    --csv ..\train.csv `
    --root-dir $env:DAISEE_ROOT `
    --max-clips 100 `
    --output ..\weights\rppg\classifier.pkl
```

### Train MER Engagement Classifier (Bi-WOOF + Optical Flow Quality)
```powershell
python -m training.train_mer_classifier `
    --csv ..\train.csv `
    --root-dir $env:DAISEE_ROOT `
    --max-clips 100 `
    --output ..\weights\mer\classifier.pkl
```

### Train Early Fusion Neural Network (Multimodal Concatenation Head)
```powershell
python -m training.train_early_fusion `
    --csv ..\train.csv `
    --root-dir $env:DAISEE_ROOT `
    --max-clips 100 `
    --epochs 25 `
    --output ..\weights\fusion\early_fusion.pt
```

---

## 6️⃣ Evaluation Commands (All Fusions & Modalities)

A single execution of `evaluate.py` evaluates all 3 individual modalities (`macro`, `rppg`, `mer`) and all 3 fusion strategies (`early`, `static`, `agentic`) simultaneously on the specified split.

### Fast Sanity Check (10 Clips)
```powershell
python -m evaluation.evaluate `
    --config configs\fusion.yaml `
    --dataset daisee `
    --split test_quick `
    --fusion agentic `
    --max-clips 10
```

### Comprehensive Evaluation on Test Set (Default: Agentic Primary + All Fusions)
```powershell
python -m evaluation.evaluate `
    --config configs\fusion.yaml `
    --dataset daisee `
    --split test `
    --fusion agentic
```

### Run Static Late Fusion as Primary
```powershell
python -m evaluation.evaluate `
    --config configs\fusion.yaml `
    --dataset daisee `
    --split test `
    --fusion static
```

### Run Early Feature Fusion as Primary
```powershell
python -m evaluation.evaluate `
    --config configs\fusion.yaml `
    --dataset daisee `
    --split test `
    --fusion early
```

> **Ablation Studies**: Add `--disable-modality macro` (or `rppg` / `mer`) to simulate missing sensor streams.

---

## 7️⃣ Timestamped Results & Analysis Visualizations

Every evaluation run creates a **unique timestamped directory** in `resultAndAnalysis\run_YYYYMMDD_HHMMSS\`:

```
resultAndAnalysis\run_YYYYMMDD_HHMMSS\
├── macro_predictions.csv
├── rppg_predictions.csv
├── mer_predictions.csv
├── fusion_early_predictions.csv
├── fusion_static_predictions.csv
├── fusion_agentic_predictions.csv
├── results_<fusion>.csv
├── overall_metrics_summary.csv
├── metrics_summary.json
└── Analysis\
    ├── fig1_modality_vs_fusion_accuracy.png
    ├── fig2_fusion_methods_comparison.png
    ├── fig3_per_class_f1_comparison.png
    ├── fig4_confusion_matrices.png
    ├── fig5_agentic_modality_weights.png
    ├── fig6_radar_performance_profile.png
    └── paper_summary_table.tex
```

The latest run results are also mirrored directly into `resultAndAnalysis\` root for immediate inspection.

### Regenerating Figures from Saved Predictions
To re-render or customize publication figures for any run:
```powershell
# Analyze the latest run automatically:
python -m evaluation.generate_analysis --dir ..\resultAndAnalysis

# Or specify an exact timestamped run folder:
python -m evaluation.generate_analysis --dir ..\resultAndAnalysis\run_20261006_211523
```
