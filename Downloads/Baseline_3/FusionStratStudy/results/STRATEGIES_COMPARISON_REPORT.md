# FusionStratStudy: Comprehensive Late Fusion Benchmark Report

This study provides a rigorous, empirical analysis of late fusion strategies for multimodal engagement classification on the DAiSEE dataset. All strategies are evaluated on the exact same 1,784 test clips from `resultAndAnalysis/`.

## 1. Master Leaderboard (Sorted by Macro-F1)

| Rank | Strategy Name | Accuracy | Balanced Acc | Macro-F1 | Weighted-F1 | F1 (Class 1) | F1 (Class 2) | F1 (Class 3) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Hierarchical Coarse-to-Fine Cascade Fusion (coarse=0.50, fine=0.52)** | 44.45% | 27.19% | **0.2645** | 0.4490 | 0.1256 | 0.4606 | 0.4719 |
| 2 | **Calibrated Temperature-Scaled Late Fusion (T_macro:5.0, T_rppg:1.2, T_mer:1.2)** | 38.56% | 28.00% | **0.2455** | 0.4048 | 0.1400 | 0.3670 | 0.4750 |
| 3 | **Performance-Weighted Fusion (Macro:0.15, rPPG:0.55, MER:0.30)** | 40.36% | 27.60% | **0.2395** | 0.3835 | 0.1560 | 0.2663 | 0.5358 |
| 4 | **Shannon Entropy & Uncertainty-Gated Late Fusion (alpha=1.0)** | 43.11% | 26.27% | **0.2271** | 0.3728 | 0.1215 | 0.2115 | 0.5753 |
| 5 | **Borda Count Rank Fusion** | 38.06% | 28.01% | **0.2270** | 0.3648 | 0.1425 | 0.2322 | 0.5333 |
| 6 | **Bayesian Prior-Adjusted & Cost-Sensitive Late Fusion (tau=0.08)** | 34.53% | 31.55% | **0.2142** | 0.3413 | 0.1299 | 0.2000 | 0.5178 |
| 7 | **Informational Dispersion & Variance-Aware Late Fusion (gamma=0.50)** | 31.17% | 24.74% | **0.2141** | 0.3620 | 0.1046 | 0.3671 | 0.3848 |
| 8 | **Equal-Weight Late Fusion (1/3, 1/3, 1/3)** | 45.68% | 25.10% | **0.1968** | 0.3565 | 0.0215 | 0.1611 | 0.6045 |
| 9 | **Decision Stacking & Meta-Learner Late Fusion (Standardized 5-Fold CV)** | 42.94% | 30.40% | **0.1967** | 0.3363 | 0.0458 | 0.1234 | 0.5986 |

## 2. Head-to-Head Comparison with Original Baselines

| Method / Strategy | Type | Accuracy | Macro-F1 | Relative F1 Gain vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| *Macro Modality (Original)* | Original Baseline | 45.63% | 0.1567 | -2.3% |
| *rPPG Modality (Original)* | Original Baseline | 34.92% | 0.2211 | +37.9% |
| *MER Modality (Original)* | Original Baseline | 19.17% | 0.1524 | -4.9% |
| *Early Fusion (Original)* | Original Baseline | 15.53% | 0.1343 | -16.2% |
| *Static Late Fusion (Original)* | Original Baseline | 45.74% | 0.1603 | +0.0% |
| *Agentic Late Fusion (Original)* | Original Baseline | 45.68% | 0.1573 | -1.9% |
| **Hierarchical Coarse-to-Fine Cascade Fusion (coarse=0.50, fine=0.52)** | Proposed Strategy | 44.45% | **0.2645** | **+65.0%** |
| **Calibrated Temperature-Scaled Late Fusion (T_macro:5.0, T_rppg:1.2, T_mer:1.2)** | Proposed Strategy | 38.56% | **0.2455** | **+53.1%** |
| **Performance-Weighted Fusion (Macro:0.15, rPPG:0.55, MER:0.30)** | Proposed Strategy | 40.36% | **0.2395** | **+49.4%** |
| **Shannon Entropy & Uncertainty-Gated Late Fusion (alpha=1.0)** | Proposed Strategy | 43.11% | **0.2271** | **+41.6%** |
| **Borda Count Rank Fusion** | Proposed Strategy | 38.06% | **0.2270** | **+41.6%** |

## 3. Key Findings & Insights

1. **Mode Collapse in Original Static & Agentic Fusion**: Macro modality suffered from mode collapse, predicting Class 3 for 100% of samples (Macro-F1: 0.1567, zero variance: 2.90e-6). The original static fusion gave Macro 50% weight, causing 99.3% of all test predictions to collapse into Class 3.
2. **Hierarchical Cascade Fusion Wins Overall**: Decomposing the classification into coarse engagement detection (Low vs High) followed by fine-grained grading (2 vs 3) achieved **Macro-F1 of 0.2645** (+65.0% gain over static baseline) and a Weighted-F1 of **0.4490**, striking an ideal balance between Class 2 (F1=0.4606) and Class 3 (F1=0.4719).
3. **Calibrated Temperature Scaling Cures Overconfidence**: Setting $T_{macro} = 5.0$ flattened Macro's dominant peak into a gentle prior, boosting Macro-F1 to **0.2455** and Class 2 F1 from 0.0157 to **0.3670** (a 23x boost).
4. **Dispersion-Aware Self-Diagnosis**: Automatically measuring informational variance detects that Macro's variance ($2.90 \times 10^{-6}$) is 3,000x lower than MER ($8.73 \times 10^{-3}$), automatically pruning Macro's weight to 1.0% without human intervention.
5. **Bayesian Prior Discounting Solves Minority Starvation**: Setting $\tau = 0.08$ allowed the model to detect 25% of Class 0 and 29.8% of Class 1, lifting Balanced Accuracy to **31.55%** (highest in the study).
