# Abstract

Machine learning models for credit risk and fraud detection are trained on
historical snapshots and deployed into dynamic, non-stationary environments.
When the statistical properties of incoming data shift over time (a phenomenon
known as concept drift), predictive performance can erode silently with no
explicit alert. Existing monitoring frameworks focus on feature distributions and
aggregate error rates but overlook a complementary signal: whether the model's
feature-level explanations are themselves becoming unstable.

This paper introduces the SHAP Stability Index (SSI), a metric that tracks
temporal consistency in SHAP feature importance rankings across successive
evaluation windows. We evaluate XGBoost, LightGBM, and CatBoost in a streaming
temporal setting with ADWIN-triggered adaptive retraining on three public credit
datasets: IEEE-CIS Fraud Detection (590,540 transactions), Give Me Some Credit
(150,000 records), and ULB Credit Card Fraud (284,807 transactions).

Across 17 ADWIN-detected drift events on the IEEE-CIS dataset, SSI declined an
average of 7.8 evaluation windows (~55 calendar days) before AUC degradation
became detectable, consistently across all three model architectures (Wilcoxon
signed-rank: p<0.001, r=0.88). As a distributional stability control, Give Me
Some Credit — which carries no real timestamp and is pseudo-temporally ordered
by demographic attributes — maintained high mean SSI (0.987–0.997), confirming
that SSI does not produce spurious alerts on demographically-sorted data. We
propose SSI as a practical, interpretability-grounded complement to conventional
monitoring metrics for deployed credit risk systems.

**Keywords:** concept drift, credit risk, SHAP, model monitoring,
gradient boosting, feature importance stability
