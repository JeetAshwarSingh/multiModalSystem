# 03_borda_count_rank_fusion: Non-Parametric Borda Count Rank Fusion

## 1. Strategy Overview
**Borda Count Rank Fusion** is a classical voting and rank-aggregation method from social choice theory and ensemble learning. Rather than operating directly on raw, uncalibrated probability scores, each modality converts its class predictions into an ordinal preference ranking:

$$R_m(c) \in \{0, 1, 2, 3\}$$

where the least probable class for modality $m$ receives rank $0$, and the most probable class receives rank $3$ (the highest score). The overall Borda score for each class $c$ is computed by aggregating the ranks across all modalities:

$$S(c) = \sum_{m \in \{\text{macro}, \text{rppg}, \text{mer}\}} w_m \cdot R_m(c)$$

The final predicted class is:

$$\hat{y} = \arg\max_{c} S(c)$$

Ties are deterministically broken using the highest individual maximum confidence.

---

## 2. Why We Chose This Strategy
In our deep analysis of `resultAndAnalysis/`, we found a fundamental flaw in soft probability voting:
- **Probability Scale Distortion**: Macro always outputted an artificially inflated probability of $\approx 0.54$ for Class 3. In probability summation, this constant $0.54$ acted as a dominant anchor.
- **Uncalibrated Output Confidences**: rPPG and MER output subtle probabilities spread across classes (e.g. $0.35$ for Class 2 vs $0.34$ for Class 3). Even when both rPPG and MER favored Class 2, Macro's $0.54$ overwhelmed both of them combined!
- **Rank Invariance**: When you convert to ranks, Macro's $0.54$ simply becomes rank $3$. If both rPPG and MER prefer Class 2 (giving it rank $3$ each), the combined Borda score for Class 2 is $2 + 3 + 3 = 8$, easily defeating Class 3's score of $3 + 2 + 1 = 6$!

We chose **Borda Count Rank Fusion** because it completely strips away uncalibrated probability magnitudes and focuses exclusively on **relative ordinal agreement** across independent sensory channels.

---

## 3. Why You Should Apply It
1. **Immunity to Overconfident Modalities**: If one sensor or model has poor calibration (extreme overconfidence or underconfidence), rank transformation neutralizes the magnitude discrepancy.
2. **Robust Multi-Sensor Agreement**: A class that is consistently ranked in the top 2 by all sensors will win over a class that is ranked 1st by only one model and last by others.
3. **No Assumptions on Probability Distribution**: It requires no assumption that the modality outputs follow valid posterior probabilities or are well-calibrated.

---

## 4. How to Run
Execute the fusion script:
```bash
python3 FusionStratStudy/03_borda_count_rank_fusion/run_fusion.py
```

Results are saved to:
- `FusionStratStudy/result/03_borda_count_rank_fusion_predictions.csv`
- `FusionStratStudy/result/03_borda_count_rank_fusion_metrics.json`
