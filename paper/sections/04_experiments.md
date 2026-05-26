# 4. Experimental Setup

## 4.1 Datasets

Three publicly available datasets spanning distinct credit risk and fraud
detection tasks are used.

**IEEE-CIS Fraud Detection.**
590,540 card transactions across ~6 months. Binary label `isFraud` (~3.5%
positive). Merged transaction + identity files give >400 raw features.
Real calendar timestamp `TransactionDT` (seconds from reference date 2017-11-30).

**Give Me Some Credit (GMSC) — distributional stability control.**
150,000 borrower records. Binary label `SeriousDlqin2yrs` (~6.7% positive).
10 raw features (credit bureau attributes). No real timestamp — pseudo-temporal
ordering constructed by sorting on age (asc) and revolving utilization (desc),
which creates a demographic gradient rather than a genuine temporal sequence.
GMSC is included as a distributional stability control (negative control): a
dataset on which SSI should remain high and ADWIN should detect zero drift,
not as a dataset for validating drift detection behaviour.

**ULB Credit Card Fraud — localised-drift control.**
284,807 European card transactions over 2 days (Sep 2013). Binary label `Class`
(492 frauds, 0.17%). V1-V28 are PCA components; only Time and Amount are raw.
Its 2-day span yields a single ADWIN event, making it a localised-drift control
rather than a persistent-drift experiment.

Dataset summary:
| Dataset            | Records   | Raw Features | Eng. Features | Positive Rate | Time Span   |
|--------------------|-----------|-------------|---------------|---------------|-------------|
| IEEE-CIS Fraud     | 590,540   | >400        | 200           | 3.5%          | ~6 months   |
| Give Me Some Credit| 150,000   | 10          | 17            | 6.7%          | N/A (stability control) |
| ULB CC Fraud       | 284,807   | 30          | 38            | 0.17%         | 2 days      |

Citation: Dal Pozzolo et al. (2015) SSCI — doi:10.1109/SSCI.2015.33

## 4.2 Feature Engineering

Transforms fitted on training data only; applied without re-fitting to evaluation windows.

**IEEE-CIS:**
- TransactionAmt: log1p, decimal part, round-amount indicator
- Temporal: day_of_week, hour_of_day, is_weekend (from TransactionDT)
- Group aggregations: count, mean, std, max of TransactionAmt grouped by
  card1-card6, P_emaildomain
- D-columns (D1-D15): log1p + missing-value binary indicator
- V-features: correlation pruning (Pearson r > 0.95) to reduce multicollinearity
- Target encoding (Laplace smoothing, prior k=300): ProductCD, card4/6,
  email domains, M1-M9, DeviceType
- Label encoding of id_01-id_38 (with 'unknown' token)
- Missing imputation: -999 for numeric
- Feature selection: top 200 by LightGBM Gini importance
- Final dimension: 200 features

**GMSC:**
- Age: zero → median imputation; delinquency counts capped at 90
- Derived: total_late_payments, debt_to_income, income_per_dependent,
  credit_utilization_risk
- log1p(MonthlyIncome), age_band (6 bins)
- Pseudo-time index
- Final dimension: 17 features

**ULB CC Fraud:**
- Amount: log1p, z-score (training stats), round-amount indicator
- Time: hour_of_day, day_of_period, is_night (00:00-06:00)
- V-interactions: V1*V2, V1*V3, V2*V3, V1*V4, V2*V4
- Final dimension: 38 features

Imbalance handling:
- IEEE-CIS + GMSC: SMOTE applied within the training partition only at each (re)training step; evaluation windows are never resampled, preventing synthetic minority instances from leaking into performance measurement
- ULB CC Fraud: cost-sensitive learning via class-weight parameter

## 4.3 Temporal Evaluation Protocol

Windowing parameters:
| Dataset         | Window     | Stride     | Min Train | Eval Windows |
|-----------------|------------|------------|-----------|--------------|
| IEEE-CIS        | 14 days    | 7 days     | 6         | 18           |
| GMSC            | 1/20 data  | 1/20 data  | 8         | 12           |
| ULB CC Fraud    | 4 hours    | 2 hours    | 4         | ~19          |

ADWIN parameters: delta=0.002, buffer B=5 windows
SSI parameters: lookback L=5, top d'=20 features, threshold tau_SSI=0.80

**Tau sensitivity analysis (IEEE-CIS, L=5 fixed).**
Table below reports, for each (tau, model) pair: the first alert window k_SSI
and the total count of alert windows out of 16 evaluable windows (windows 8–23).

| tau  | XGB k_SSI | XGB alerts | LGB k_SSI | LGB alerts | CB k_SSI | CB alerts |
|------|-----------|------------|-----------|------------|----------|-----------|
| 0.85 | 8         | 16 / 16    | 8         | 16 / 16    | 8        | 16 / 16   |
| 0.80 | 8         | 15 / 16    | 8         | 16 / 16    | 8        | 16 / 16   |
| 0.75 | 8         | 9  / 16    | 8         | 13 / 16    | 8        | 12 / 16   |
| 0.70 | 8         | 6  / 16    | 8         | 9  / 16    | 16       | 5  / 16   |

The first alert window k_SSI=8 is fully invariant to tau for all three models at
tau ∈ {0.75, 0.80, 0.85}. At tau=0.70, CatBoost's k_SSI shifts to window 16
because its SSI begins at 0.745 (window 8), which lies above the 0.70 threshold;
lowering tau below CatBoost's minimum SSI in the early drift period forfeits
11 windows of advance warning for that model. XGBoost and LightGBM retain
k_SSI=8 at tau=0.70 because their SSI values fall to 0.000 and 0.657
respectively at window 8. tau=0.80 is selected as it maximises alert coverage
(15–16 / 16 windows per model) while preserving k_SSI=8 across all architectures.

**L robustness.** The lookback L governs how many consecutive window-to-window
Spearman correlations are averaged into SSI(k). On IEEE-CIS at tau=0.80, SSI
falls below the threshold in 15/16 windows (XGBoost) and 16/16 windows
(LightGBM, CatBoost) — a persistent sub-threshold signal covering nearly the
entire evaluation horizon. Because the sustained below-threshold run spans
windows 8–23, any L ∈ {3, 5, 7} produces the same first alert window k_SSI=8:
a shorter lookback (L=3) is slightly more reactive but fires on the same window,
and a longer lookback (L=7) applies more smoothing over an already-low signal
without delaying the alert. On near-stable datasets (GMSC: SSI ≥ 0.987;
ULB CC: SSI ≥ 0.922), no alerts fire regardless of L. L=5 is retained as the
default, balancing reactivity against noise from single-window perturbations.

## 4.4 Hyperparameter Search Spaces

XGBoost (Optuna TPE, 150 trials, 2h timeout):
- n_estimators: {50, 75, ..., 300} (step 25)
- max_depth: [3, 9]
- learning_rate: [0.01, 0.30] (log-uniform)
- min_child_weight: [1, 20]
- subsample: [0.5, 1.0]
- colsample_bytree: [0.3, 1.0]
- colsample_bylevel: [0.3, 1.0]
- reg_alpha: [1e-8, 10.0] (log-uniform)
- reg_lambda: [1e-8, 10.0] (log-uniform)
- gamma: [0.0, 5.0]
- scale_pos_weight: [1, 100]

LightGBM (Optuna TPE, 150 trials, 2h timeout):
- n_estimators: {200, 250, ..., 1500} (step 50)
- num_leaves: [20, 200]
- max_depth: [3, 12]
- learning_rate: [0.01, 0.30] (log-uniform)
- min_child_samples: [5, 100]
- subsample: [0.5, 1.0]
- subsample_freq: [1, 10]
- colsample_bytree: [0.3, 1.0]
- reg_alpha: [1e-8, 10.0] (log-uniform)
- reg_lambda: [1e-8, 10.0] (log-uniform)
- min_split_gain: [0.0, 1.0]
- is_unbalance: {True, False} (categorical)

CatBoost (Optuna TPE, 150 trials, 2h timeout):
- iterations: {200, 250, ..., 1500} (step 50)
- depth: [3, 10]
- learning_rate: [0.01, 0.30] (log-uniform)
- l2_leaf_reg: [1.0, 10.0] (log-uniform)
- random_strength: [0.1, 10.0] (log-uniform)
- bagging_temperature: [0.0, 1.0]
- border_count: {32, 64, 128, 254} (categorical)

All three: TPE sampler (seed 42), Median pruner (20 startup trials, 5 warmup steps).
Tuning objective: AUC on temporal validation window W_val.

## 4.5 Implementation Details

- Python 3.11
- XGBoost 2.0.3, LightGBM 4.3.0, CatBoost 1.2.7
- SHAP 0.45.0, River 0.21.0, Optuna 3.6.1, scikit-learn 1.4.2
- Random seed: 42 (all models, samplers, splits)
- GPU: CUDA acceleration where available (XGBoost: device=cuda; LightGBM:
  device=gpu; CatBoost: task_type=GPU); CPU fallback for all three
- All hyperparameter sets serialised as JSON for exact reproducibility
