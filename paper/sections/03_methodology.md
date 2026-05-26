# 3. Methodology

## 3.1 Problem Formulation

Let D = {(x_t, y_t)}_{t=1}^T denote a temporal stream of credit transactions,
where x_t in R^d is the feature vector at time t and y_t in {0,1} is the
binary outcome label. We partition D into K non-overlapping evaluation windows
W_1, W_2, ..., W_K of fixed size w with stride s, preserving temporal order so
that no future observations enter any window's training set.

At each window k, a gradient boosting model M_k is trained on all data preceding
W_k and evaluated on W_k. Two regimes are compared:

- Static: M_k = M_1 for all k — trained once at initialisation, never updated.
- Adaptive: M_k is retrained whenever an online drift detector signals a
  statistically significant change in the model's error rate.

A concept drift event occurs at window k when P_k(x, y) differs significantly
from P_{k-1}(x, y) as determined by ADWIN (Bifet & Gavaldà, 2007). Real drift
— where the conditional P_k(y|x) shifts — degrades the static model
systematically, whereas the adaptive model recovers by retraining on recent data.

## 3.2 The SHAP Stability Index

At each window k, TreeSHAP (Lundberg et al., 2020) is applied to a stratified
subsample of up to 2,000 instances from W_k, stratified by class label to
preserve the window's positive rate. For ULB CC Fraud (0.17% positive rate)
this guarantees at least three fraud instances per SHAP evaluation, preventing
the background distribution from collapsing to an all-negative sample.
TreeSHAP computes exact Shapley values for tree ensembles in polynomial time,
making it tractable for high-dimensional credit datasets evaluated across many
windows.

When the adaptive model undergoes ADWIN-triggered retraining, TreeSHAP is
applied to the newly retrained model at each subsequent window. The SHAP
background distribution therefore reflects the current model's internal
structure rather than a fixed baseline, so SSI measures rank stability of the
evolving deployed model across consecutive retraining epochs.

The global feature importance vector phi_k in R^d is:

  phi_k^(j) = (1/|S_k|) * sum_{i in S_k} |phi_j(x_i)|,   j = 1,...,d

where S_k is the evaluation sample and phi_j(x_i) is the SHAP attribution for
feature j on instance x_i. R_k is the rank vector of the top d'=20 components
of phi_k in descending order.

**Definition (SHAP Stability Index):**
SSI(k) = (1/L) * sum_{l=1}^{L} rho_s(R_{k-l+1}, R_{k-l})

where rho_s is the Spearman rank correlation and L is the lookback window.

Spearman is preferred over Pearson because SHAP magnitudes vary in scale across
windows independently of rank structure; the ordinal measure captures genuine
reordering while being invariant to monotone rescaling. SSI=1 means identical
ordering to the previous window; SSI~0 means near-random rearrangement. SSI
requires only model outputs and unlabelled features — computable before
ground-truth labels are available.

**Lead-time measurement:**
- k_SSI: first window where SSI(k) < 0.80 (instability threshold)
- k_AUC: first window after drift where AUC drops > 0.02 below baseline
- Lead time lambda = k_AUC - k_SSI (in evaluation windows)

## 3.3 ADWIN-Triggered Adaptive Retraining

ADWIN (Bifet & Gavaldà, 2007) maintains a sliding window W over the per-window
error rate (1 - accuracy). It tests whether the mean of any sub-window W_0 of W
differs significantly from the mean of W_1 = W \ W_0:

  |mu_hat_{W_0} - mu_hat_{W_1}| >= epsilon_cut

where epsilon_cut is derived from the Hoeffding bound at confidence 1-delta.
The window is trimmed from its oldest end until no such partition satisfies the
inequality. When a cut is found, drift is declared and the adaptive model
retrains immediately. Implementation uses the River streaming ML library
(Montiel et al., 2021).

Algorithm: Temporal Streaming Evaluation with ADWIN and SSI Monitoring
  Input: W_1,...,W_K; model M; ADWIN(delta); lookback L; buffer size B
  1. Train M on W_1,...,W_init; initialise buffer B = {W_1,...,W_init}
  2. For k = init+1 to K:
     a. Compute AUC_k, error_k on W_k using M
     b. Feed error_k to ADWIN
     c. Compute phi_k via TreeSHAP on stratified sample of W_k
     d. Compute SSI(k)
     e. If ADWIN signals drift:
        - Retrain M on the B most recent windows in buffer
        - Log: k, AUC_k, SSI(k), lead time lambda
     f. Add W_k to buffer; drop oldest if |B| > B
  3. Return {AUC_k}, {SSI(k)}, drift event log

## 3.4 Hyperparameter Optimisation

XGBoost, LightGBM, and CatBoost are each tuned using Optuna (Akiba et al.,
2019) with the Tree-structured Parzen Estimator (TPE) sampler (Bergstra et al.,
2011). TPE models the distribution of high-performing trials as density
l(lambda) and low-performing trials as g(lambda), sampling candidates where
l/g is maximised.

The tuning objective is AUC on a held-out temporal validation window W_val
placed immediately after the training period, ensuring no future data leakage.
Each run executes 150 trials with a two-hour timeout. The Median pruner
terminates unpromising trials early. All optimised hyperparameter sets are
saved as JSON files. Full search spaces are listed in Section 4.
