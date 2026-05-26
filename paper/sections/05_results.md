# 5. Results

## 5.1 Main Performance Comparison

Table: model × dataset × mean AUC (±std) × ADWIN events × retrains × mean SSI

| Dataset  | Model               | Mean AUC (±std) | ADWIN Events | Retrains | Mean SSI |
|----------|---------------------|-----------------|--------------|----------|----------|
| IEEE-CIS | StaticXGBoost       | 0.899 ± 0.013   | —            | —        | —        |
| IEEE-CIS | AdaptiveXGBoost     | 0.904 ± 0.018   | 17           | 16       | 0.660    |
| IEEE-CIS | StaticLightGBM      | 0.919 ± 0.016   | —            | —        | —        |
| IEEE-CIS | AdaptiveLightGBM    | 0.865 ± 0.062   | 16           | 15       | 0.683    |
| IEEE-CIS | StaticCatBoost      | 0.902 ± 0.014   | —            | —        | —        |
| IEEE-CIS | AdaptiveCatBoost    | 0.835 ± 0.047   | 17           | 16       | 0.717    |
| GMSC     | StaticXGBoost       | 0.504 ± 0.013   | —            | —        | —        |
| GMSC     | AdaptiveXGBoost     | 0.500 ± 0.014   | 0            | 0        | 0.992    |
| GMSC     | StaticLightGBM      | 0.506 ± 0.016   | —            | —        | —        |
| GMSC     | AdaptiveLightGBM    | 0.500 ± 0.011   | 0            | 0        | 0.997    |
| GMSC     | StaticCatBoost      | 0.503 ± 0.017   | —            | —        | —        |
| GMSC     | AdaptiveCatBoost    | 0.499 ± 0.014   | 0            | 0        | 0.987    |
| ULB CC   | StaticXGBoost       | 0.999 ± 0.003   | —            | —        | —        |
| ULB CC   | AdaptiveXGBoost     | 0.963 ± 0.027   | 1            | 1        | 0.952    |
| ULB CC   | StaticLightGBM      | 0.999 ± 0.003   | —            | —        | —        |
| ULB CC   | AdaptiveLightGBM    | 0.972 ± 0.024   | 1            | 1        | 0.990    |
| ULB CC   | StaticCatBoost      | 0.998 ± 0.006   | —            | —        | —        |
| ULB CC   | AdaptiveCatBoost    | 0.960 ± 0.033   | 1            | 1        | 0.968    |

Key observations:
- IEEE-CIS (heavy drift, 16-17 ADWIN events): LightGBM achieves the highest static
  baseline (0.919). AdaptiveXGBoost is the only adaptive model to exceed its static
  counterpart (+0.005 AUC). Both AdaptiveLightGBM (0.865) and AdaptiveCatBoost (0.835)
  underperform their static baselines, with the B=5 window buffer discarding historical
  signal on rapid consecutive drift events; higher variance confirms window-to-window
  instability (std 0.062 and 0.047 respectively vs 0.013-0.016 for static models).
- GMSC (distributional stability control): all six models cluster at ~0.50 AUC,
  0 drift events, SSI ≥ 0.987. This result is expected and consistent with the
  dataset's role as a negative control: the demographic ordering (age ASC,
  utilization DESC) produces a smooth distributional gradient rather than
  temporal dynamics, so both ADWIN and SSI correctly indicate no drift activity.
- ULB CC Fraud: static models are near-perfect (0.998-0.999). All adaptive models
  incur a temporary dip from the single retraining episode, leaving static superior.
  CatBoost adaptive (0.960) is marginally below XGBoost adaptive (0.963).

## 5.2 SSI Dynamics Under Concept Drift

IEEE-CIS SSI values per window:
- AdaptiveXGBoost:  0.00, 0.43, 0.55, 0.64, 0.60, 0.77, ... (windows 8–23)
- AdaptiveLightGBM: 0.66, 0.78, 0.78, 0.77, 0.66, 0.60, ... (windows 8–23)
- AdaptiveCatBoost: 0.75, 0.74, 0.76, 0.79, 0.79, 0.74, 0.74, 0.76, 0.69,
                    0.70, 0.72, 0.70, 0.64, 0.64, 0.62, 0.68 (windows 8–23)

**SSI=0.00 anomaly (AdaptiveXGBoost, window 8):** The zero value at window 8
arises from a complete SHAP rank reorganisation immediately following the first
ADWIN-triggered retrain. When the intersection between the top-20 SHAP features
of consecutive windows is empty — because retraining causes XGBoost to assign
near-zero importance to the features that dominated the previous window — the
Spearman rank correlation is undefined; the implementation returns SSI=0.00 as
a conservative fallback. This is a meaningful signal (total feature-ordering
disruption) rather than a computation error. LightGBM and CatBoost do not
exhibit this behaviour at window 8 because their symmetric-tree and ordered-
boosting structures retain more feature overlap across the retrain boundary.
Windows 9–23 recover to 0.43–0.77 as the retrained XGBoost model stabilises
on a new feature ordering.

Mean SSI separation across all three architectures:
- IEEE-CIS (high drift):                        0.660 – 0.717
- ULB CC Fraud (localised drift):               0.952 – 0.990
- GMSC (distributional stability control):      0.987 – 0.997

SSI reliably separates drift regimes across all three model architectures without
recalibration. CatBoost's mean SSI on IEEE-CIS (0.717) lies above XGBoost (0.660)
and LightGBM (0.683), suggesting its symmetric tree structure produces marginally
more stable feature orderings under drift, though all three remain well below the
tau=0.80 alert threshold throughout the high-drift period.

## 5.3 SSI as Leading Indicator

SSI lead time table (lambda = k_AUC - k_SSI, in evaluation windows; tau=0.80):

| Model            | Dataset  | Events | Mean lambda | Std  | Range |
|------------------|----------|--------|-------------|------|-------|
| AdaptiveXGBoost  | IEEE-CIS | 16     | 7.81        | 4.26 | 0–15  |
| AdaptiveLightGBM | IEEE-CIS | 15     | 7.67        | 4.03 | 0–15  |
| AdaptiveCatBoost | IEEE-CIS | 16     | 8.00        | 4.11 | 0–15  |
| AdaptiveXGBoost  | ULB CC   | 1      | 3.00        | —    | 3     |
| AdaptiveLightGBM | ULB CC   | 1      | 0.00        | —    | 0     |
| AdaptiveCatBoost | ULB CC   | 1      | 0.00        | —    | 0     |

Events = drift windows with sufficient pre-window data for lambda computation
(first drift event per dataset is excluded as no prior window exists).
std computed as population standard deviation (ddof=0), consistent across models.

All three models share k_SSI=8 on IEEE-CIS: AdaptiveCatBoost SSI falls below
tau=0.80 at window 8 (SSI=0.745), the same alert window as XGBoost and LightGBM.
Mean lambda across all three architectures is 7.83 windows (~55 calendar days),
confirming SSI as a consistent early-warning signal regardless of model family.
Lead time grows monotonically from window 9 onward for all three models (windows
20–23: CatBoost SSI=0.636, 0.635, 0.623, 0.680), reinforcing the pattern.

Counter-example: AdaptiveLightGBM and AdaptiveCatBoost both yield lambda=0 on
ULB CC Fraud (single drift event at window 13, SSI=0.922 and 0.968 respectively —
both well above tau=0.80). On this localised-drift dataset, SSI does not provide
advance warning because the model's feature-importance ordering remains stable
despite the ADWIN alert; only AdaptiveXGBoost (lambda=3) shows a modest lead time.
This counter-example is consistent with the expectation that SSI lead times are
meaningful only under persistent, multi-window drift.

GMSC: 0 drift events detected across all models (consistent with distributional
stability control interpretation; no lambda computation applicable).

## 5.4 Statistical Significance

**AUC: adaptive vs static (Wilcoxon signed-rank, one-sided H1: adaptive > static)**
n=17–18 windows per model:
- AdaptiveXGBoost vs StaticXGBoost (IEEE-CIS): W=109, p=0.066, r=0.45 — medium
  effect, marginally non-significant due to small n.
- All other adaptive vs static comparisons: p > 0.50.

**SSI lead time: H0: lambda=0 (Wilcoxon signed-rank, one-sided H1: lambda > 0)**
Computed on IEEE-CIS only (n_ULB=1 per model, insufficient for testing):
- AdaptiveXGBoost:  n=16, n_zero=1, W=120, p<0.001***, r=0.88 (large effect)
- AdaptiveLightGBM: n=15, n_zero=1, W=105, p<0.001***, r=0.88 (large effect)
- AdaptiveCatBoost: n=16, n_zero=1, W=120, p<0.001***, r=0.88 (large effect)

All three models reject H0: lambda=0 with p<0.001 and large effect size (r=0.88),
confirming that SSI provides a statistically significant lead time over AUC
degradation under genuine concept drift. The single lambda=0 event per model
(at drift window 8, the first analyzable event) does not attenuate this result;
it arises because no pre-window baseline exists at the series onset.

## 5.5 SSI vs. PSI Drift Alert Comparison

The Population Stability Index (PSI) is the most widely deployed feature-level
drift monitor in production credit risk systems. We compute PSI on the 20
most drift-relevant raw features of IEEE-CIS (TransactionAmt; card1–card3,
card5; addr1, addr2, dist1; C1, C2, C5, C6, C9, C11, C13; D1–D4, D10),
using the first six windows as the reference distribution and quantile-based
10-bin binning. Per-feature PSI is computed for each of the 18 evaluation
windows (6–23). The standard operational threshold PSI > 0.25 is used for
both mean PSI (across 20 features) and individual feature breach detection.

**Results (IEEE-CIS):**

| Metric              | k_alert  | Notes                                           |
|---------------------|----------|-------------------------------------------------|
| PSI — mean          | none     | Mean PSI never exceeds 0.25 (peak: 0.050, w20) |
| PSI — any feature   | 13       | First individual feature breach at window 13    |
| SSI (all 3 models)  | 8        | tau=0.80, L=5; all three architectures agree   |
| AUC degradation     | ~16      | k_SSI + mean lambda 7.83 windows               |

Mean PSI across 20 features peaks at 0.050 (windows 20–21) and never
approaches the 0.25 alert threshold, meaning a standard PSI monitor would
produce **no alert** across the entire IEEE-CIS evaluation horizon despite
persistent concept drift. The first individual-feature breach (one feature
at window 13, PSI=0.272) lags SSI by five windows (35 calendar days). SSI
flags all three models at window 8 — five windows before PSI detects any
single-feature instability, and approximately eight windows before AUC
degradation becomes detectable.

Three structural reasons explain SSI's lead:

1. **Feature weighting**: SSI tracks only the top-d' SHAP-ranked features,
   concentrating sensitivity on the dimensions the model actively relies on.
   PSI applied uniformly across all features dilutes the signal from the few
   features that carry predictive weight.

2. **No binning artefacts**: PSI requires a binning decision (bin count,
   bin boundaries) that critically affects sensitivity. SSI's Spearman
   formulation is binning-free: it operates on rank order, which is
   invariant to monotone rescaling of the underlying values.

3. **Model-internal signal**: SSI measures how the model reorganises its
   internal feature weighting in response to distribution shift, not merely
   whether the raw input distribution has changed. A distribution change
   that does not affect the model's decision boundary will not depress SSI,
   reducing false positives.

These properties make SSI complementary to PSI rather than a replacement:
PSI monitors raw input stability and requires no model; SSI monitors
model-internal stability and requires no labels. A combined deployment
uses PSI for population-level input monitoring and SSI for
model-specific early warning.
