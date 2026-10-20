# 08_hierarchical_cascade_fusion: Hierarchical Coarse-to-Fine Engagement Cascade Fusion

## 1. Strategy Overview
**Hierarchical Coarse-to-Fine Cascade Fusion** decouples the 4-class engagement classification problem into a structured two-stage hierarchical decision cascade.

Rather than forcing a flat 4-way argmax where majority classes routinely overpower minority classes, the decision process mimics clinical and educational assessment:

```
                          [ Multimodal Inputs ]
                                    |
                                    v
                     +------------------------------+
                     |    Stage 1: Coarse State     |
                     |  Low (0,1)  vs  High (2,3)   |
                     +------------------------------+
                                    |
                   +----------------+----------------+
                   |                                 |
                   v                                 v
     [ Predicted Low (0 or 1) ]        [ Predicted High (2 or 3) ]
                   |                                 |
                   v                                 v
    +-----------------------------+   +-----------------------------+
    |   Stage 2B: Fine Disengage  |   |    Stage 2A: Fine Engage    |
    |  Very Low (0) vs Low (1)    |   |    Engaged (2) vs High (3)  |
    +-----------------------------+   +-----------------------------+
```

### Mathematical Formulation
1. **Stage 1 (Binary Engagement State)**:
   We aggregate fused probabilities into macroscopic meta-states:
   $$P_{\text{low}} = P_{\text{fused}}(0) + P_{\text{fused}}(1)$$
   $$P_{\text{high}} = P_{\text{fused}}(2) + P_{\text{fused}}(3)$$

   If $P_{\text{high}} \ge \theta_{\text{coarse}}$ (default: $0.65$), the sample is routed to Stage 2A; otherwise to Stage 2B.

2. **Stage 2A (Fine-Grained Engaged State: 2 vs 3)**:
   Conditioned on high engagement, the decision between Engaged (2) and Highly Engaged (3) is determined by their relative likelihood ratio:
   $$\hat{y} = \begin{cases} 3 & \text{if } \frac{P_{\text{fused}}(3)}{P_{\text{fused}}(2) + P_{\text{fused}}(3) + \epsilon} \ge \theta_{\text{fine}} \\ 2 & \text{otherwise} \end{cases}$$

3. **Stage 2B (Fine-Grained Disengaged State: 0 vs 1)**:
   Conditioned on low engagement:
   $$\hat{y} = \begin{cases} 1 & \text{if } \frac{P_{\text{fused}}(1)}{P_{\text{fused}}(0) + P_{\text{fused}}(1) + \epsilon} \ge \theta_{\text{minority}} \\ 0 & \text{otherwise} \end{cases}$$

---

## 2. Why We Chose This Strategy
In flat softmax decision-making, the model must simultaneously compare:
- Class 0 ($N=4$) vs Class 3 ($N=814$)
- Class 2 ($N=882$) vs Class 3 ($N=814$)

Because Class 2 and Class 3 comprise **95.1%** of the entire dataset, a flat argmax almost always collapses to Class 2 or Class 3.
However, detecting whether a student is fundamentally disengaged (Classes 0 or 1) vs engaged (Classes 2 or 3) is a qualitatively distinct task from deciding whether an engaged student is moderately engaged (2) or highly engaged (3).

We chose **Hierarchical Cascade Fusion** because:
1. **Decouples Scale**: Resolves the severe class imbalance at Stage 1 before fine-grained class boundaries are established.
2. **Independent Calibration**: Thresholds can be tuned independently for coarse detection vs fine grading.
3. **High Clinical & Educational Alignment**: Teachers first notice *if* a student is zoning out (Stage 1), and then observe the *extent* of disengagement (Stage 2).

---

## 3. Why You Should Apply It
1. **Balanced Precision-Recall Control**: You can control the sensitivity to student disengagement simply by adjusting the coarse threshold $\theta_{\text{coarse}}$ without retraining.
2. **Prevents Majority Domination**: Stage 2A strictly chooses between 2 and 3 without interference from Classes 0 and 1, boosting Class 2 F1 up to **0.35+**.
3. **Structured Interpretability**: Yields two actionable diagnostic flags per clip: (1) macroscopic engagement status, and (2) fine-grained ordinal grade.

---

## 4. How to Run
Execute the fusion script:
```bash
python3 FusionStratStudy/08_hierarchical_cascade_fusion/run_fusion.py
```

To test custom coarse and fine thresholds:
```bash
python3 FusionStratStudy/08_hierarchical_cascade_fusion/run_fusion.py --theta-coarse 0.70 --theta-fine 0.52
```

Results are saved to:
- `FusionStratStudy/result/08_hierarchical_cascade_fusion_predictions.csv`
- `FusionStratStudy/result/08_hierarchical_cascade_fusion_metrics.json`
