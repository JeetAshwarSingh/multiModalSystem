# 01_equal_weight_average: Equal-Weight Late Fusion

## 1. Strategy Overview
**Equal-Weight Late Fusion** (also known as Simple Soft Voting or Unweighted Probability Averaging) is the fundamental baseline late fusion strategy. It computes the arithmetic mean of class probability distributions emitted by all modalities without assigning preferential bias to any single modality:

$$P_{\text{fused}}(c \mid x) = \frac{1}{M} \sum_{m=1}^{M} P_m(c \mid x)$$

where $M = 3$ is the number of modalities (Macro, rPPG, MER), and $c \in \{0, 1, 2, 3\}$ denotes the engagement classes. The predicted class is chosen via argmax:

$$\hat{y} = \arg\max_{c} P_{\text{fused}}(c \mid x)$$

---

## 2. Why We Chose This Strategy
In the original baseline evaluation (`resultAndAnalysis/`), the authors selected an arbitrary heuristic weighting scheme for Static Late Fusion:
- Macro: $0.5$ (50%)
- rPPG: $0.3$ (30%)
- MER: $0.2$ (20%)

Assigning 50% weight to Macro was an untested human assumption. However, our deep analysis revealed that **Macro suffered from complete mode collapse**, predicting Class 3 for 100% of the clips (1,784/1,784) with virtually zero variance ($\sigma^2 = 2.90 \times 10^{-6}$). By weighting Macro at 0.5, the original static fusion allowed Macro's static ~0.54 probability for Class 3 to swamp all other modalities, causing the final fusion to predict Class 3 on 99.3% of the dataset.

We chose **Equal-Weight Average ($1/3, 1/3, 1/3$)** to:
1. **Eliminate arbitrary human bias**: Test performance under a neutral, unbiased prior where all sensor modalities are treated with equal initial trust.
2. **Provide an empirical control benchmark**: Any advanced late fusion strategy (rank-based, temperature-scaled, entropy-gated) must prove its worth against this fundamental baseline.
3. **Assess the natural democratic effect**: Equal weighting reduces Macro's weight from 0.50 down to 0.33, immediately granting rPPG and MER more voice.

---

## 3. Why You Should Apply It
1. **Sanity Check & Control Baseline**: In any scientific publication or benchmark, you should never report an advanced fusion method without comparing it to simple equal-weight averaging. Reviewers expect this control.
2. **Zero Hyperparameters**: Unlike grid-searched or learned fusion, equal weighting requires no training data, no validation split, and no hyperparameter tuning, making it immune to overfitting.
3. **Immediate Gains over Static Baseline**: Even simple equal weighting boosts Macro-F1 from **0.1603** (original static) to **0.1968** (+22.7% relative gain) simply by curbing Macro's dominant footprint.

---

## 4. How to Run
Execute the standalone fusion script:
```bash
python3 FusionStratStudy/01_equal_weight_average/run_fusion.py
```
Or from inside this directory:
```bash
cd FusionStratStudy/01_equal_weight_average
python3 run_fusion.py
```

The script will:
1. Load predictions from `resultAndAnalysis/macro_predictions.csv`, `rppg_predictions.csv`, and `mer_predictions.csv`.
2. Compute the fused probabilities and final predicted class.
3. Evaluate Accuracy, Macro-F1, Balanced Accuracy, Per-Class metrics, and Confusion Matrix.
4. Output results into `FusionStratStudy/result/01_equal_weight_average_predictions.csv` and `FusionStratStudy/result/01_equal_weight_average_metrics.json`.
