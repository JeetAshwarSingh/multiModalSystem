# Project Information: Engagement-MAS

## 1. Project Overview

**Engagement-MAS** is a multi-modal perception and multi-agent fusion system designed to estimate human engagement levels (4 ordinal classes: 0 = Very Low, 1 = Low, 2 = High, 3 = Very High) from facial video clips, evaluated on the **DAiSEE** dataset.

### End-to-End Architecture
1. **Parallel Perception**:
   - **Macro Block (`blocks/macro/macro_block.py`)**: Facial appearance via frozen DINOv2 ViT-B/14 + geometric behavior features + temporal Transformer encoder. Emits 4-class engagement probabilities, boredom, confusion, and frustration.
   - **rPPG Block (`blocks/rppg/rppg_block.py`)**: Remote photoplethysmography extracting blood volume pulse (BVP) and heart rate from subtle skin color variations. Maps cardiac activation to physiological engagement levels.
   - **MER Block (`blocks/mer/mer_block.py`)**: Micro-expression recognition capturing rapid facial motion using Farneback optical flow and Bi-WOOF descriptors around apex frames. Maps micro-motion dynamics to engagement levels.
2. **Thread-Safe Synchronization Gate (`gate/sync_gate.py`)**:
   - Manages asynchronous modality arrival times per clip.
   - Uses thread locks (`threading.Lock`) to safely handle parallel execution from `ThreadPoolExecutor`.
   - Releases bundles immediately when all modalities arrive or flushes partial bundles upon timeout (default 60s) or pipeline finalization.
3. **Fusion Decision Layer**:
   - **Early Fusion (`fusion/early_fusion.py`)**: Concatenates feature representations and projects to engagement logits.
   - **Static Late Fusion (`fusion/static_late_fusion.py`)**: Fixed weighted average of unimodal probability distributions (Macro: 0.5, rPPG: 0.3, MER: 0.2).
   - **Agentic Late Fusion (`fusion/agentic_fusion.py`)**: Dynamic weighting based on real-time quality metrics (Macro blur, rPPG SNR, MER flow noise) with temporal moving-average smoothing.
4. **Training, Evaluation & Research Paper Analysis**:
   - PyTorch Lightning training pipeline (`training/train_macro.py`) with embedding caching and CUDA/MPS acceleration.
   - Full evaluation suite (`evaluation/evaluate.py`), robustness masking (`evaluation/robustness.py`), and statistical significance tests (`evaluation/significance.py`).
   - Publication-ready analysis engine (`evaluation/analyze_and_plot.py` and `evaluation/generate_analysis.py`) that outputs individual modality predictions, fusion predictions, performance tables, and 6 high-resolution research figures (300 DPI) plus an IEEE/ACM LaTeX table in `resultAndAnalysis/Analysis/`.

---

## 2. Standardized Modality Interface

Every perception block (`MacroBlock`, `RPPGBlock`, `MERBlock`) implements `run(clip_path: str) -> dict` returning a standardized dictionary:

| Key | Type | Description |
|---|---|---|
| `clip_id` | `str` | Deterministic SHA-256 hash (16 chars) derived from clip path. |
| `modality` | `str` | Name of modality: `"macro"`, `"rppg"`, or `"mer"`. |
| `features` | `np.ndarray` | Numerical feature representation (`832`-dim for Macro, BVP signal array for rPPG, `8`-dim Bi-WOOF for MER). |
| `predictions` | `dict` | Prediction dictionary: `{"engagement": [p0, p1, p2, p3]}` across all blocks, plus `{"hr": float}` for rPPG and affective states for Macro. |
| `quality_metrics` | `dict` | Modality-specific quality indicators (`blur`, `face_detection_rate`, `snr`, `flow_noise`). |
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
  - Prioritizes loading fine-tuned checkpoint (`best.ckpt`) over `last.pt`.
* **Input**: Video file path (`.avi`, `.mp4`).
* **Output**: Standard dictionary with `832`-dim feature vector, 4-class probabilities for each affective state, and quality metrics.

#### [`rppg_block.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/blocks/rppg/rppg_block.py)
* **Purpose**: Remote physiological signal monitoring and cardiac-based engagement estimation.
* **Mechanism**:
  - Detects facial skin regions and tracks subtle green-channel intensity changes corresponding to cardiac pulses.
  - Wraps `Face2PPG` when installed; otherwise falls back gracefully to a deterministic synthetic pulse model based on clip hash.
  - Computes pulse quality metrics: signal-to-noise ratio (`snr`), `peak_prominence`, and `hr_stability`.
  - Estimates 4-class physiological engagement probabilities from heart rate and signal dynamics.
* **Input**: Video file path.
* **Output**: Standard dictionary with raw BVP signal (`features`), heart rate in bpm (`predictions: {"hr": float}`), 4-class engagement distribution (`predictions: {"engagement": [...]}`), and SNR quality metrics.

#### [`mer_block.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/blocks/mer/mer_block.py)
* **Purpose**: Micro-expression recognition capturing rapid involuntary facial movements and optical flow engagement.
* **Mechanism**:
  - Crops face bounding boxes across sampled frames.
  - Computes dense Farneback optical flow between successive frames.
  - Identifies the apex frame (peak optical flow magnitude) representing micro-expression onset/apex.
  - Extracts an 8-bin **Bi-WOOF** (Binarized Weighting of Oriented Optical Flow) descriptor capturing subtle directionality.
  - Computes motion quality metrics: `flow_noise`, `track_stability`, `framerate_factor`, and `blur_estimate`.
  - Maps micro-motion energy and flow dynamics to a 4-class engagement distribution.
* **Input**: Video file path.
* **Output**: Standard dictionary with 8-dim Bi-WOOF descriptor (`features`), 4-class engagement distribution (`predictions: {"engagement": [...]}`), quality metrics, and status.

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
* **Purpose**: Dynamic multi-agent fusion adjusting modality trust based on real-time signal quality.
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
  - **Multi-Platform Dataset Resolution**: Automatically resolves DAiSEE root across Windows/WSL (`/mnt/c/...`), Mac (`/Users/...`), and `DAISEE_ROOT` env variable.
  - **Disk Caching**: Extracts appearance embeddings via frozen DINOv2 once on Epoch 0 and stores them in `data/cache/*.pt`, speeding up subsequent epochs by 10x+.
  - **Multi-task Loss**: Jointly optimizes engagement cross-entropy alongside boredom, confusion, and frustration auxiliary losses.
  - **Model Checkpointing**: Saves best checkpoint (`best.ckpt`) and final model (`last.pt`) under `weights/macro/`.
* **Input**: `--config configs/training.yaml`, `--dataset daisee`, `--split train`.
* **Output**: Checkpoints in `weights/macro/` and cached tensors in `data/cache/`.

---

### Evaluation & Analysis Pipeline (`engagement-mas/evaluation/`)

#### [`evaluate.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/evaluation/evaluate.py)
* **Purpose**: End-to-end evaluation harness for the entire system across individual modalities and fusion strategies.
* **Mechanism**:
  - Resolves dataset paths dynamically across Windows/WSL and Mac.
  - Spawns parallel worker threads via `ThreadPoolExecutor` to process clips through perception blocks.
  - Streams results through `SyncGate`, evaluates each modality individually (`macro`, `rppg`, `mer`), and evaluates all fusion strategies (`early`, `static`, `agentic`).
  - Automatically saves detailed prediction CSVs for each modality and fusion to `resultAndAnalysis/`.
  - Computes full classification metrics (Accuracy, Macro-F1, Precision, Recall, confusion matrix) and outputs `overall_metrics_summary.csv` and `metrics_summary.json`.
  - Automatically generates 6 publication-ready figures (300 DPI) and an IEEE/ACM LaTeX table in `resultAndAnalysis/Analysis/`.

#### [`analyze_and_plot.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/evaluation/analyze_and_plot.py)
* **Purpose**: Publication-quality plotting and statistical metrics generation using Matplotlib and Seaborn.
* **Figures Generated**:
  1. `fig1_modality_vs_fusion_accuracy.png`: Performance benchmark bar chart across all individual modalities and fusion models.
  2. `fig2_fusion_methods_comparison.png`: Head-to-head multi-metric comparison of Early, Static Late, and Agentic Late fusion.
  3. `fig3_per_class_f1_comparison.png`: Per-class F1-scores across all 4 engagement levels (Very Low, Low, Engaged, High).
  4. `fig4_confusion_matrices.png`: Multi-panel Seaborn confusion matrix heatmaps.
  5. `fig5_agentic_modality_weights.png`: Adaptive weight distribution boxplot demonstrating quality-based reasoning.
  6. `fig6_radar_performance_profile.png`: Multi-metric trade-off radar profile.
  7. `paper_summary_table.tex`: Copy-paste ready IEEE/ACM formatted LaTeX table.

#### [`generate_analysis.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/evaluation/generate_analysis.py)
* **Purpose**: Standalone CLI script to regenerate or update all figures and tables at any time from existing CSV files without re-running the neural network inference.
* **Usage**: `python -m evaluation.generate_analysis --dir ../resultAndAnalysis`.

#### [`metrics.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/evaluation/metrics.py)
* Classification metrics calculations using `scikit-learn` (Accuracy, Macro-F1, Precision, Recall, and Confusion Matrix).

#### [`robustness.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/evaluation/robustness.py)
* Stress-tests pipeline reliability by randomly masking out modalities across iterations and logging drop schedules to JSON.

#### [`significance.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/evaluation/significance.py)
* Runs evaluations across multiple random seeds and executes paired t-tests (`scipy.stats.ttest_rel`) to assess statistical significance between fusion approaches.

---

### Data Management & Verification (`engagement-mas/data/`)

#### [`convert_daisee.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/data/convert_daisee.py)
* Parses raw DAiSEE label text files and video folders into standard CSV files (`train.csv`, `val.csv`, `test.csv`).

#### [`validate_dataset.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/data/validate_dataset.py)
* Validates dataset integrity across platforms: checks column names, verifies that video paths exist on disk (checking both `/DataSet/Test` and `/Test` variants), and prints class distributions.

#### Dataset Split Files (`data/daisee/*.csv` & Root `*.csv`)
* `train.csv` (5,358 clips), `val.csv` (1,429 clips), `test.csv` (1,784 clips), and `test_quick.csv` (10 clips for fast sanity checking).

---

### Configuration Files (`engagement-mas/configs/`)

| Configuration File | Role |
|---|---|
| [`datasets.yaml`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/configs/datasets.yaml) | Dataset root directory and split CSV paths. Auto-detects remote Windows/WSL (`/mnt/c/...`) and local Mac (`/Users/...`) environments. |
| [`fusion.yaml`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/configs/fusion.yaml) | Static weights (`0.5`, `0.3`, `0.2`), agentic quality thresholds (`rppg_snr: 5.0`, `mer_flow_noise: 0.2`, `macro_blur: 100.0`), and smoothing window. |
| [`gate.yaml`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/configs/gate.yaml) | Synchronization timeout duration (`60s`) and gate logger path. |
| [`training.yaml`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/configs/training.yaml) | Training hyperparameters (epochs: 20, batch size: 8, learning rate: 5e-4, loss weights, checkpoint dir, and device `"auto"`). |

---

### Directory Organization

```text
Baseline_3/
├── train.csv, val.csv, test.csv, test_quick.csv
├── weights/
│   └── macro/
│       ├── best.ckpt                     # Best fine-tuned DINOv2 + Transformer weights
│       ├── last.pt                       # Final epoch checkpoint
│       └── lightning_logs/               # Training logs & loss telemetry
├── resultAndAnalysis/                    # Evaluation outputs & analysis
│   ├── macro_predictions.csv             # Macro unimodal predictions
│   ├── rppg_predictions.csv              # rPPG physiological predictions
│   ├── mer_predictions.csv               # Micro-expression predictions
│   ├── fusion_early_predictions.csv      # Early fusion predictions
│   ├── fusion_static_predictions.csv     # Static late fusion predictions
│   ├── fusion_agentic_predictions.csv    # Agentic late fusion predictions
│   ├── overall_metrics_summary.csv       # Side-by-side performance table
│   ├── metrics_summary.json              # Full per-class metrics
│   └── Analysis/                         # 300 DPI research figures
│       ├── fig1_modality_vs_fusion_accuracy.png
│       ├── fig2_fusion_methods_comparison.png
│       ├── fig3_per_class_f1_comparison.png
│       ├── fig4_confusion_matrices.png
│       ├── fig5_agentic_modality_weights.png
│       ├── fig6_radar_performance_profile.png
│       └── paper_summary_table.tex
├── engagement-mas/                       # Core codebase
│   ├── blocks/                           # Perception blocks (Macro, rPPG, MER)
│   ├── configs/                          # YAML configurations
│   ├── data/                             # Dataset conversion & validation
│   ├── evaluation/                       # Evaluation harness & plotting tools
│   ├── fusion/                           # Early, Static, and Agentic fusion
│   ├── gate/                             # Synchronization gate
│   └── training/                         # PyTorch Lightning training
└── DocumentationAndRequirements/         # Documentation & multi-platform requirements
    ├── project_information.md
    ├── commands.md
    ├── cloud_run.md
    ├── requirements-windows.txt
    ├── requirements-mac.txt
    └── requirements-cloud.txt
```

---

## 4. How to Run on Windows / WSL & Mac

### Step 1: Environment Setup
In Anaconda Prompt / PowerShell / Bash:
```bash
conda create -n engagement-mas python=3.10 -y
conda activate engagement-mas
cd engagement-mas
pip install -r ../DocumentationAndRequirements/requirements-windows.txt  # Or requirements-mac.txt on Mac
```
*(Optional for NVIDIA GPU on Windows/WSL)*:
```bash
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Validate Data
```bash
python -m data.validate_dataset --splits test_quick
```

### Step 3: Train Macro Block
```bash
python -m training.train_macro --config configs/training.yaml --dataset daisee --split train
```
The script will auto-detect CUDA GPU if present (otherwise MPS or CPU), cache extracted features in `data/cache/`, and save model weights to `weights/macro/best.ckpt`.

### Step 4: Run Evaluation & Generate Research Figures
```bash
# Quick sanity check (10 clips)
python -m evaluation.evaluate --config configs/fusion.yaml --dataset daisee --split test_quick --fusion agentic --max-clips 10

# Full test set evaluation (all 1,784 clips)
python -m evaluation.evaluate --config configs/fusion.yaml --dataset daisee --split test --fusion agentic
```
All predictions, comparison tables, and figures will automatically be generated in `resultAndAnalysis/` and `resultAndAnalysis/Analysis/`.

### Step 5: Regenerate Visualizations Anytime
```bash
python -m evaluation.generate_analysis --dir ../resultAndAnalysis
```
