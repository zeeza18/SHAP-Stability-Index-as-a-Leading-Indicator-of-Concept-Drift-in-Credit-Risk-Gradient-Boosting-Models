# 1. Introduction

Credit risk and fraud detection models occupy a central role in modern
financial services. Supervisory frameworks, including the Federal Reserve's
Model Risk Management guidance (SR 11-7, 2011), require institutions to
assess, validate, and continuously monitor the quantitative models used for
determining default probability, creditworthiness, and fraud likelihood.
Gradient boosting methods, specifically XGBoost (Chen & Guestrin, 2016),
LightGBM (Ke et al., 2017), and CatBoost (Prokhorenkova et al., 2018), have
become the standard approach for tabular credit risk tasks, consistently
outperforming logistic regression and neural network baselines on structured
financial data.

A fundamental challenge emerges once these models are deployed. They are
trained on a historical snapshot of the world and released into an environment
that does not stay fixed. Customer repayment behaviour shifts with
macroeconomic conditions, fraud tactics adapt to circumvent detection
systems, and regulatory or product changes reshape how future training labels
are generated. These forces cause the joint distribution P(x, y) to change
over time, a phenomenon formally defined as concept drift (Gama et al., 2014).
Under drift, a model with strong training-time performance degrades quietly in
production: no exception is raised, no alert is fired, and the degradation
accumulates silently until it has already affected lending or fraud decisions.
Surveys of deployed machine learning systems confirm that temporal model decay
is among the most consequential unresolved problems in production financial ML
(Hinder, Vaquet & Hammer, 2024).

Several algorithms have been developed to detect drift in data streams. Among
the most principled is ADWIN (Adaptive WINdowing) (Bifet & Gavaldà, 2007),
which maintains a sliding window of model error estimates whose length adjusts
automatically. When the mean error within any two sub-windows diverges beyond
a statistical threshold, drift is declared and adaptive retraining can be
triggered. Methods of this class monitor prediction outcomes: error rates,
AUC computed over labelled evaluation windows, or population stability indices
applied to raw feature distributions. They share a structural limitation:
they can only respond to drift once it has already surfaced in aggregate
performance. By then, the model has served deteriorated predictions for a
number of evaluation periods.

Explainable AI offers a different window into model behaviour. SHAP
(SHapley Additive exPlanations) (Lundberg & Lee, 2017) provides feature-level
attributions grounded in cooperative game theory and has become a standard
tool for satisfying regulatory expectations around the interpretability of
individual credit decisions (Guidotti et al., 2018). Recent work has begun to
quantify SHAP attribution stability as a property in its own right,
demonstrating that feature importance rankings can vary significantly under
class imbalance and resampling (Lin & Wang, 2025). What has not been studied
is whether SHAP feature importance rankings shift systematically over time as
concept drift accumulates, or whether such shifts carry information about
impending performance degradation before that degradation becomes visible in
outcome metrics.

This paper addresses both questions. We introduce the SHAP Stability Index
(SSI), a metric that tracks the Spearman rank correlation between global SHAP
feature importance vectors computed in consecutive evaluation windows. A
falling SSI indicates that the features the model relies on are reorganising,
even before this reorganisation appears in AUC or error rate. We evaluate SSI alongside XGBoost, LightGBM, and CatBoost under realistic
concept drift on three public credit datasets, using ADWIN-triggered adaptive
retraining and principled Optuna-based hyperparameter optimisation
(Akiba et al., 2019). This paper makes four contributions:

1. A streaming temporal evaluation framework covering three gradient boosting
   architectures (XGBoost, LightGBM, CatBoost), combining ADWIN drift detection
   with Optuna-tuned adaptive retraining across three large, structurally
   distinct public datasets.

2. The SHAP Stability Index (SSI): a metric that tracks the Spearman rank
   correlation of global SHAP feature importance vectors across successive
   evaluation windows, requiring no access to ground truth labels at detection
   time.

3. Empirical evidence that SSI functions as a leading indicator of model
   degradation: across 17 ADWIN-detected drift events on the IEEE-CIS Fraud
   Detection dataset (590,540 transactions), SSI declined an average of 7.8
   evaluation windows before AUC deterioration became detectable, consistent
   across all three model architectures.

4. Cross-dataset validation of SSI's sensitivity to genuine distributional
   change: SSI averaged between 0.66 and 0.72 on the drifting IEEE-CIS dataset
   and between 0.95 and 0.99 on the two lower-drift datasets across all three
   models, confirming that the metric tracks real rather than spurious variation.

The remainder of this paper is organised as follows. Section 2 reviews related
work on gradient boosting for credit risk, concept drift detection, and SHAP
in financial systems. Section 3 formalises the problem and presents the SSI
methodology. Section 4 describes the experimental setup and datasets. Section
5 presents the results. Section 6 discusses findings and limitations. Section 7 concludes.
