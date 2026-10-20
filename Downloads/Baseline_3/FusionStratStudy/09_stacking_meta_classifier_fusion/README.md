# 09_stacking_meta_classifier_fusion: Decision Stacking & Meta-Learner Late Fusion

## 1. Strategy Overview
**Decision Stacking (Wolpert, 1992)** formulates late fusion as a second-level supervised learning task. Instead of assuming fixed linear combination rules or static heuristic thresholds, stacking trains a **meta-classifier** that learns how to optimally combine the output predictions of base modalities.

### Meta-Feature Representation
For every video $i$, we construct a 12-dimensional meta-feature vector concatenating the posterior class probabilities emitted by all $M=3$ individual blocks:

$$\mathbf{z}^{(i)} = \Big[ P_{\text{macro}}^{(i)}(0..3) \;\Vert\; P_{\text{rppg}}^{(i)}(0..3) \;\Vert\; P_{\text{mer}}^{(i)}(0..3) \Big] \in \mathbb{R}^{12}$$

### Meta-Learner Formulation
The meta-classifier is parameterized as a regularized linear multinomial model with parameters $\mathbf{W} \in \mathbb{R}^{4 \times 12}$ and $\mathbf{b} \in \mathbb{R}^{4}$:

$$\mathbf{s}^{(i)} = \mathbf{W} \mathbf{z}^{(i)} + \mathbf{b}$$

$$P_{\text{fused}}(c \mid \mathbf{z}^{(i)}) = \frac{\exp\big(s_c^{(i)}\big)}{\sum_{k=0}^{3} \exp\big(s_k^{(i)}\big)}$$

The parameters are optimized by minimizing L2-regularized weighted cross-entropy loss:

$$\mathcal{L}(\mathbf{W}, \mathbf{b}) = -\frac{1}{N} \sum_{i=1}^{N} \beta_{y^{(i)}} \ln P_{\text{fused}}\big(y^{(i)} \mid \mathbf{z}^{(i)}\big) + \lambda \|\mathbf{W}\|_F^2$$

where $\beta_c = \sqrt{\frac{N}{4 \cdot N_c}}$ provides class-balanced importance weighting to counteract the DAiSEE class imbalance.

To guarantee zero train-test data leakage, out-of-fold predictions can be generated via $K$-Fold Cross Validation ($K=5$).

---

## 2. Why We Chose This Strategy
All previous late fusion strategies (equal weighting, performance weighting, temperature scaling, Borda count) share a common assumption:
> *Each modality's prediction for class $c$ is evaluated independently, without considering cross-modal correlations.*

In reality, modalities exhibit complex joint dependencies:
- If rPPG predicts Class 2 (moderate heart rate) AND MER predicts Class 3 (high facial motion), the subject is actively engaged in discussion.
- If rPPG predicts Class 3 (elevated heart rate) but MER shows zero facial motion (Class 0), this may signal physical agitation or stress rather than high engagement.

A linear probability average cannot model these cross-modality interaction terms. We chose **Stacking** because it can learn cross-channel synergies directly from data.

---

## 3. Why You Should Apply It
1. **Learned Optimal Synergy**: Automatically learns the optimal relative importance of each class probability from each modality, including off-diagonal interactions.
2. **Built-in Class Balancing**: Uses inverse-root frequency weighting $\beta_c$ during training to ensure minority classes are not starved.
3. **Rigorous Cross-Validation**: Implements stratified $K$-fold cross-validation to ensure reliable, leak-free performance estimates.

---

## 4. How to Run
Execute the fusion script:
```bash
python3 FusionStratStudy/09_stacking_meta_classifier_fusion/run_fusion.py
```

To adjust the number of cross-validation folds:
```bash
python3 FusionStratStudy/09_stacking_meta_classifier_fusion/run_fusion.py --k-folds 5 --epochs 100
```

Results are saved to:
- `FusionStratStudy/result/09_stacking_meta_classifier_fusion_predictions.csv`
- `FusionStratStudy/result/09_stacking_meta_classifier_fusion_metrics.json`
