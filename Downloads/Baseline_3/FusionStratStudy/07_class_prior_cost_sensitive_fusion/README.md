# 07_class_prior_cost_sensitive_fusion: Bayesian Prior-Adjusted & Cost-Sensitive Late Fusion

## 1. Strategy Overview
**Bayesian Prior-Adjusted & Cost-Sensitive Late Fusion** addresses the extreme class imbalance inherent in engagement detection datasets (specifically DAiSEE).

In the DAiSEE test split, class distribution is drastically skewed:
- **Class 0 (Very Low)**: 4 samples ($0.22\%$)
- **Class 1 (Low)**: 84 samples ($4.71\%$)
- **Class 2 (Engaged)**: 882 samples ($49.44\%$)
- **Class 3 (Highly Engaged)**: 814 samples ($45.63\%$)

Standard neural networks optimized with cross-entropy implicitly absorb the prior probability $\pi_c = P(y=c)$:

$$P(y=c \mid x) \propto P(x \mid y=c) \cdot \pi_c$$

Because $\pi_0 = 0.0022$ and $\pi_1 = 0.0471$ are infinitesimal compared to $\pi_2 \approx 0.49$, minority classes are routinely suppressed even when strong disengagement symptoms (head drop, lack of eye contact, irregular pulse) are detected.

Prior-Adjusted Fusion applies Bayesian log-prior discounting:

$$\tilde{P}_{\text{fused}}(c \mid x) = \frac{P_{\text{fused}}(c \mid x)}{\pi_c^\tau + \epsilon}$$

where $\tau \in [0, 0.25]$ is the Bayesian prior discount exponent:
- $\tau = 0$: Standard unadjusted fusion (majority classes dominate).
- $\tau \approx 0.10$: Calibrated prior compensation (minority classes 0 & 1 receive sufficient boost to be recognized without destroying precision on 2 & 3).

Alternatively, in cost-sensitive mode, we minimize expected operational risk using a cost matrix $\mathbf{C}$:

$$\hat{y} = \arg\min_{k} \sum_{c=0}^{3} C_{k,c} \cdot P_{\text{fused}}(c \mid x)$$

where false positives on engagement (failing to detect a struggling/disengaged student) carry a higher penalty.

---

## 2. Why We Chose This Strategy
In `resultAndAnalysis/`, the baseline Static Late Fusion achieved:
- Class 0 F1: **0.0000** (0/4 detected)
- Class 1 F1: **0.0000** (0/84 detected)
- Class 2 F1: **0.0157** (7/882 detected)
- Class 3 F1: **0.6257** (809/814 detected)

The model was acting as a near-blind majority classifier because it was crushed under the sheer mass of the Class 2 and Class 3 training priors.

We chose **Prior-Adjusted Cost-Sensitive Fusion** because:
1. In educational technology, the primary goal of an engagement detector is to **flag disengaged, struggling, or bored students** so an intelligent tutoring system or teacher can intervene.
2. A model with 0% recall on Class 0 and 0% recall on Class 1 is virtually useless in a classroom setting, regardless of whether its overall accuracy is 45%.
3. By adjusting for the prior, we can tune the operational sensitivity directly.

---

## 3. Why You Should Apply It
1. **Critical for Minority Class Detection**: Boosts recall and F1 on minority Classes 0 and 1 without retraining any backbones.
2. **Tunable Operational Policy**: You can adjust $\tau$ based on the application:
   - High sensitivity to disengagement: set $\tau = 0.15$.
   - Balanced general accuracy: set $\tau = 0.08$.
3. **Mathematically Principled**: Rooted in Bayesian likelihood ratio inversion rather than ad-hoc heuristics.

---

## 4. How to Run
Execute the fusion script:
```bash
python3 FusionStratStudy/07_class_prior_cost_sensitive_fusion/run_fusion.py
```

To test different prior discount exponents $\tau$:
```bash
python3 FusionStratStudy/07_class_prior_cost_sensitive_fusion/run_fusion.py --tau 0.12
```

Results are saved to:
- `FusionStratStudy/result/07_class_prior_cost_sensitive_fusion_predictions.csv`
- `FusionStratStudy/result/07_class_prior_cost_sensitive_fusion_metrics.json`
