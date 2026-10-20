# 05_entropy_uncertainty_gated_fusion: Shannon Entropy & Uncertainty-Gated Late Fusion

## 1. Strategy Overview
**Entropy & Uncertainty-Gated Late Fusion** is an information-theoretic dynamic fusion strategy. Instead of relying on static modality weights, it computes the Shannon entropy of each modality's prediction distribution on a per-video basis:

$$H(P_m) = -\sum_{c=0}^{3} P_m(c \mid x) \ln\big(P_m(c \mid x) + \epsilon\big)$$

Entropy directly measures prediction uncertainty:
- **Low Entropy ($H \to 0$)**: The modality is sharp, decisive, and confident about its predicted engagement class.
- **High Entropy ($H \to \ln 4 \approx 1.386$)**: The modality is uncertain, diffuse, and approaching a uniform distribution.

The dynamic sample-level weight $w_m(x)$ is assigned inversely proportional to entropy:

$$w_m(x) = \exp\big(-\alpha \cdot H(P_m)\big)$$

where $\alpha > 0$ is an uncertainty sensitivity scaling factor. The weights are normalized across available modalities:

$$\tilde{w}_m(x) = \frac{w_m(x)}{\sum_{k} w_k(x)}$$

and the fused posterior probability is computed as:

$$P_{\text{fused}}(c \mid x) = \sum_{m} \tilde{w}_m(x) \cdot P_m(c \mid x)$$

---

## 2. Why We Chose This Strategy
In `resultAndAnalysis/`, the static late fusion treated every prediction with equal confidence weight regardless of whether the underlying model was sharp or guessing blindly.

Furthermore, the baseline `agentic_fusion.py` attempted to adjust weights using external heuristics (such as image blur or flow noise thresholds). However:
1. External heuristics frequently fail when metadata is missing or corrupted.
2. An image can have high sharpness (low blur) while the neural network inside remains totally uncertain.
3. Information theory gives us a self-contained measure of certainty directly from the model's posterior output $P(y \mid x)$.

We chose **Entropy-Gated Fusion** to provide an intrinsic, self-calibrating mechanism: when rPPG detects a clean, rhythmic pulse and yields an authoritative prediction for Class 2, its entropy drops and its weight automatically surges to dominate the decision. When MER optical flow is noisy or flat, its entropy spikes and its influence naturally recedes.

---

## 3. Why You Should Apply It
1. **Dynamic Sample-Level Adaptation**: Adapts instantly to per-video signal quality without requiring any hand-tuned camera or sensor heuristics.
2. **Self-Contained & Model-Agnostic**: Operates exclusively on output probability distributions; requires no access to internal feature representations or external quality metrics.
3. **Suppresses Uninformative Modalities**: Modalities that output diffuse distributions across all classes are naturally downweighted in favor of clear, peaked distributions.

---

## 4. How to Run
Execute the fusion script:
```bash
python3 FusionStratStudy/05_entropy_uncertainty_gated_fusion/run_fusion.py
```

To test different uncertainty scaling factors $\alpha$:
```bash
python3 FusionStratStudy/05_entropy_uncertainty_gated_fusion/run_fusion.py --alpha 1.5
```

Results are saved to:
- `FusionStratStudy/result/05_entropy_uncertainty_gated_fusion_predictions.csv`
- `FusionStratStudy/result/05_entropy_uncertainty_gated_fusion_metrics.json`
