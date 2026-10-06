# Project Information: Engagement-MAS

## 1. Project Overview

**Engagement-MAS** is a multi-modal perception and multi-agent fusion system designed to estimate human engagement levels (4 ordinal classes: 0 = Very Low, 1 = Low, 2 = High, 3 = Very High) from facial video clips, evaluated on the **DAiSEE** dataset.

### End-to-End Architecture
1. **Parallel Perception**:
   - **Macro Block (`blocks/macro/macro_block.py`)**: Facial appearance via frozen DINOv2 ViT-B/14 + genuine 30-dimensional geometric landmark behavior features (EAR, MAR, head pose, eyebrow raise, gaze direction, expression ratios) + temporal Transformer encoder. Emits 4-class engagement probabilities, boredom, confusion, and frustration.
   - **rPPG Block (`blocks/rppg/rppg_block.py`)**: Remote photoplethysmography extracting blood volume pulse (BVP) and heart rate from skin color variations using a self-contained CHROM and Green-channel bandpass filtered pipeline. Maps cardiac dynamics to physiological engagement levels via a trained balanced machine learning classifier (`weights/rppg/classifier.pkl`).
   - **MER Block (`blocks/mer/mer_block.py`)**: Micro-expression recognition capturing rapid facial motion using Farneback optical flow and 8-bin Bi-WOOF descriptors around apex frames. Predicts engagement using a trained balanced classifier (`weights/mer/classifier.pkl`).
2. **Thread-Safe Synchronization Gate (`gate/sync_gate.py`)**:
   - Manages asynchronous modality arrival times per clip.
   - Uses thread locks (`threading.Lock`) to safely handle parallel execution from `ThreadPoolExecutor`.
   - Releases bundles immediately when all modalities arrive or flushes partial bundles upon timeout (default 60s) or pipeline finalization.
3. **Fusion Decision Layer**:
   - **Early Fusion (`fusion/early_fusion.py`)**: Concatenates feature representations (Macro 768/832, rPPG 128, MER 256) and infers 4-class engagement logits using a trained deep neural network projection head (`weights/fusion/early_fusion.pt`).
   - **Static Late Fusion (`fusion/static_late_fusion.py`)**: Fixed weighted average of unimodal probability distributions (Macro: 0.5, rPPG: 0.3, MER: 0.2).
   - **Agentic Late Fusion (`fusion/agentic_fusion.py`)**: Dynamic weighting based on real-time quality metrics (Macro blur, rPPG SNR, MER flow noise) with temporal moving-average smoothing, dummy placeholder quality validation, and Shannon entropy degenerate output detection.
4. **Training, Evaluation & Research Paper Analysis**:
   - PyTorch Lightning training pipeline (`training/train_macro.py`) with embedding caching and CUDA/MPS/CPU acceleration.
   - Modality classifier training pipelines (`training/train_rppg_classifier.py`, `training/train_mer_classifier.py`, `training/train_early_fusion.py`) with balanced class weighting.
   - Full evaluation suite (`evaluation/evaluate.py`), robustness masking (`evaluation/robustness.py`), and statistical significance tests (`evaluation/significance.py`).
   - Publication-ready analysis engine (`evaluation/analyze_and_plot.py` and `evaluation/generate_analysis.py`) that stores every execution in a timestamped run folder (`resultAndAnalysis/run_YYYYMMDD_HHMMSS/`) with prediction CSVs, performance tables, and 6 high-resolution research figures (300 DPI) plus an IEEE/ACM LaTeX table.

---

## 2. Standardized Modality Interface

Every perception block (`MacroBlock`, `RPPGBlock`, `MERBlock`) implements `run(clip_path: str) -> dict` returning a standardized dictionary:

| Key | Type | Description |
|---|---|---|
| `clip_id` | `str` | Deterministic SHA-256 hash (16 chars) derived from clip path. |
| `modality` | `str` | Name of modality: `"macro"`, `"rppg"`, or `"mer"`. |
| `features` | `np.ndarray` | Numerical feature representation (`832`-dim for Macro, normalized `128`-dim BVP for rPPG, `8`-dim Bi-WOOF for MER). |
| `predictions` | `dict` | Prediction dictionary: `{"engagement": [p0, p1, p2, p3]}` across all blocks, plus `{"hr": float}` for rPPG and affective states for Macro. |
| `quality_metrics` | `dict` | Modality-specific quality indicators (`blur`, `face_detection_rate`, `snr`, `flow_noise`, `hr_stability`, `track_stability`). |
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
  - Extracts **30 rich geometric facial distances/angles**: Eye Aspect Ratios (EAR) for left/right eyes, Mouth Aspect Ratio (MAR), Head Pose (yaw, pitch, roll), eyebrow raise distance, iris gaze direction proxies, and inter-landmark expression distance ratios.
  - Projects behavior to 64 dimensions, concatenates with appearance (`832` dimensions), and inputs to a 2-layer temporal Transformer encoder.
  - Four multi-task linear classification heads output logits for `engagement`, `boredom`, `confusion`, and `frustration`.
  - Computes quality metrics: `face_detection_rate`, `blur` (Laplacian variance), `brightness`, and `motion_smoothness`.
  - Auto-loads fine-tuned checkpoint (`weights/macro/best.ckpt`) with 208/208 key diagnostic verification across CPU, MPS, and CUDA.
* **Input**: Video file path (`.avi`, `.mp4`).
* **Output**: Standard dictionary with `832`-dim feature vector, 4-class probabilities for each affective state, and quality metrics.

#### [`rppg_block.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/blocks/rppg/rppg_block.py)
* **Purpose**: Remote physiological signal monitoring and cardiac-based engagement estimation.
* **Mechanism**:
  - Detects facial skin regions and tracks Green-channel and Chrominance (CHROM) intensity variations.
  - Applies 3rd-order Butterworth bandpass filtering (0.7–4.0 Hz $\to$ 42–240 BPM).
  - Computes FFT power spectral density to determine genuine heart rate (`hr`), signal-to-noise ratio (`snr` in dB), `peak_prominence`, and temporal `hr_stability`.
  - Standardizes filtered BVP signal to a normalized 128-dimensional feature vector.
  - Automatically loads trained engagement classifier from `weights/rppg/classifier.pkl` (with an unbiased physiological heuristic fallback).
* **Input**: Video file path.
* **Output**: Standard dictionary with 128-dim BVP signal (`features`), heart rate in bpm (`predictions: {"hr": float}`), 4-class engagement distribution (`predictions: {"engagement": [...]}`), and SNR quality metrics.

#### [`mer_block.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/blocks/mer/mer_block.py)
* **Purpose**: Micro-expression recognition capturing rapid involuntary facial movements and optical flow engagement.
* **Mechanism**:
  - Crops face bounding boxes across sampled frames.
  - Computes dense Farneback optical flow between successive frames.
  - Identifies the apex frame (peak optical flow magnitude) representing micro-expression onset/apex.
  - Extracts an 8-bin **Bi-WOOF** (Binarized Weighting of Oriented Optical Flow) descriptor capturing subtle directionality.
  - Computes motion quality metrics: `flow_noise`, `track_stability`, `framerate_factor`, and `blur_estimate`.
  - Automatically loads trained engagement classifier from `weights/mer/classifier.pkl` (with an unbiased Gaussian kernel heuristic fallback).
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
* **Purpose**: Multimodal feature-level concatenation and deep neural projection.
* **Mechanism**: Concatenates available feature vectors (Macro 768, rPPG 128, MER 256 = 1152 dims), loads trained deep neural network weights (`weights/fusion/early_fusion.pt`), and outputs softmax probabilities over the 4 engagement classes.
* **Input**: Synchronized bundle from `SyncGate`.
* **Output**: Dictionary with key `"engagement"` containing 4-class probability distribution.

#### [`static_late_fusion.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/fusion/static_late_fusion.py)
* **Purpose**: Fixed-weight decision-level late fusion baseline.
* **Mechanism**: Computes fixed weighted sum using configured weights (`macro: 0.5`, `rppg: 0.3`, `mer: 0.2` from `configs/fusion.yaml`), dynamically re-normalizing weights if a modality is absent.
* **Input**: Synchronized bundle from `SyncGate`.
* **Output**: Dictionary with key `"engagement"` containing fused 4-class probabilities.

#### [`agentic_fusion.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/fusion/agentic_fusion.py)
* **Purpose**: Dynamic multi-agent fusion adjusting modality trust based on real-time signal quality and confidence sanity.
* **Mechanism**:
  - **Degenerate Detector**: Computes Shannon entropy $H(p)$ of per-modality prediction distributions; zeroes out weight if entropy indicates a collapsed or constant prediction.
  - **Quality Inspector**: Validates SNR, flow noise, and blur metrics against thresholds, penalizing placeholder or degraded signals.
  - **Weight Reasoner**: Increases weight (by `+0.2`) for high-quality modalities and penalizes noisy/blurry modalities (by `-0.2`), bounded between `0.1` and `0.8`.
  - **Temporal Smoother**: Applies a moving average over a sliding window of recent predictions (window size = 5) for temporal stability.
* **Input**: Synchronized bundle from `SyncGate`.
* **Output**: Dictionary with key `"engagement"`, 4-class probabilities, and an audit trail `"explanation"` string.

---

### Training Pipeline (`engagement-mas/training/`)

#### [`train_macro.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/training/train_macro.py)
* **Purpose**: Fine-tunes the temporal Transformer and classification heads of `MacroBlock` using PyTorch Lightning.
* **Key Features**: Cross-platform device selection (CUDA / MPS / CPU), embedding caching (`data/cache/*.pt`), multi-task loss, and checkpoint saving (`weights/macro/best.ckpt`).

#### [`train_rppg_classifier.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/training/train_rppg_classifier.py)
* **Purpose**: Extracts physiological features and trains a balanced Random Forest classifier for cardiac engagement prediction.
* **Output**: Saved model in `weights/rppg/classifier.pkl`.

#### [`train_mer_classifier.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/training/train_mer_classifier.py)
* **Purpose**: Extracts Bi-WOOF motion descriptors and optical flow quality indicators to train a balanced engagement classifier.
* **Output**: Saved model in `weights/mer/classifier.pkl`.

#### [`train_early_fusion.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/training/train_early_fusion.py)
* **Purpose**: Trains a 3-layer multimodal neural network projection head on concatenated feature vectors with inverse class frequency loss.
* **Output**: Saved PyTorch model in `weights/fusion/early_fusion.pt`.

---

### Evaluation & Analysis Pipeline (`engagement-mas/evaluation/`)

#### [`evaluate.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/evaluation/evaluate.py)
* **Purpose**: End-to-end evaluation harness for the entire system across individual modalities and fusion strategies.
* **Mechanism**:
  - Resolves dataset paths dynamically across Windows, WSL, and Mac via `DAISEE_ROOT`.
  - Processes clips through parallel perception blocks using `ThreadPoolExecutor`.
  - Synchronizes streams through `SyncGate`, evaluating unimodal baselines and all three fusion heads (`early`, `static`, `agentic`).
  - **Timestamped Run Storage**: Automatically generates a dedicated folder for each run:
    `resultAndAnalysis/run_YYYYMMDD_HHMMSS/`
    containing all 6 prediction CSVs, `results_<fusion>.csv`, `overall_metrics_summary.csv`, `metrics_summary.json`, and an `Analysis/` folder containing all 6 research figures and `paper_summary_table.tex`.
  - Also mirrors latest results into `resultAndAnalysis/` root for immediate inspection.

#### [`analyze_and_plot.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/evaluation/analyze_and_plot.py)
* **Purpose**: Publication-quality plotting and statistical metrics generation using Matplotlib and Seaborn.
* **Figures Generated**:
  1. `fig1_modality_vs_fusion_accuracy.png`: Performance benchmark bar chart across all individual modalities and fusion models.
  2. `fig2_fusion_methods_comparison.png`: Head-to-head multi-metric comparison of Early, Static Late, and Agentic Late fusion.
  3. `fig3_per_class_f1_comparison.png`: Per-class F1-scores across all 4 engagement levels.
  4. `fig4_confusion_matrices.png`: Multi-panel Seaborn confusion matrix heatmaps.
  5. `fig5_agentic_modality_weights.png`: Adaptive weight distribution boxplot demonstrating quality-based reasoning.
  6. `fig6_radar_performance_profile.png`: Multi-metric trade-off radar profile.
  7. `paper_summary_table.tex`: Copy-paste ready IEEE/ACM formatted LaTeX table.

#### [`generate_analysis.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/evaluation/generate_analysis.py)
* **Purpose**: Standalone CLI script to regenerate or update all figures and tables at any time from existing CSV files without re-running model inference. Supports specifying a timestamped run folder or auto-detecting the latest run.

#### [`metrics.py`](file:///Users/jeetashwar/Downloads/Baseline_3/engagement-mas/evaluation/metrics.py)
* Classification metrics calculations (Accuracy, Balanced Accuracy, Macro-F1, Precision, Recall, Confusion Matrix).

---

### Directory Organization

```text
Baseline_3/
├── train.csv, val.csv, test.csv, test_quick.csv
├── WINDOWS_SETUP_AND_CHANGELOG.md        # Summary guide for Windows setup
├── weights/                              # Trained model weights
│   ├── macro/
│   │   ├── best.ckpt                     # 208/208 verified keys (DINOv2 + Transformer)
│   │   └── last.pt
│   ├── rppg/
│   │   └── classifier.pkl                # Trained rPPG engagement classifier
│   ├── mer/
│   │   └── classifier.pkl                # Trained MER engagement classifier
│   └── fusion/
│       └── early_fusion.pt               # Trained Early Fusion deep neural head
├── resultAndAnalysis/                    # Evaluation outputs & analysis
│   ├── run_YYYYMMDD_HHMMSS/              # Timestamped run folder (per execution)
│   │   ├── macro_predictions.csv
│   │   ├── rppg_predictions.csv
│   │   ├── mer_predictions.csv
│   │   ├── fusion_early_predictions.csv
│   │   ├── fusion_static_predictions.csv
│   │   ├── fusion_agentic_predictions.csv
│   │   ├── results_<fusion>.csv
│   │   ├── overall_metrics_summary.csv
│   │   ├── metrics_summary.json
│   │   └── Analysis/                     # 300 DPI research figures
│   │       ├── fig1_modality_vs_fusion_accuracy.png
│   │       ├── fig2_fusion_methods_comparison.png
│   │       ├── fig3_per_class_f1_comparison.png
│   │       ├── fig4_confusion_matrices.png
│   │       ├── fig5_agentic_modality_weights.png
│   │       ├── fig6_radar_performance_profile.png
│   │       └── paper_summary_table.tex
│   └── (Mirrored latest outputs in root resultAndAnalysis/)
├── engagement-mas/                       # Core codebase
│   ├── blocks/                           # Perception blocks (Macro, rPPG, MER)
│   ├── configs/                          # YAML configurations (datasets, fusion, gate, training)
│   ├── data/                             # Dataset conversion & validation
│   ├── evaluation/                       # Evaluation harness & plotting tools
│   ├── fusion/                           # Early, Static, and Agentic fusion
│   ├── gate/                             # Thread-safe synchronization gate
│   ├── tests/                            # PyTest unit test suite
│   └── training/                         # PyTorch & Scikit-Learn training pipelines
└── DocumentationAndRequirements/         # Documentation & multi-platform requirements
    ├── CHANGES_AND_WINDOWS_SETUP.md      # Comprehensive changelog & Windows guide
    ├── project_information.md            # Detailed architecture breakdown
    ├── commands.md                       # Complete training & evaluation commands
    ├── requirements-windows.txt          # Windows requirements
    ├── requirements-mac.txt              # macOS requirements
    └── requirements-cloud.txt            # Cloud/Linux CUDA requirements
```
