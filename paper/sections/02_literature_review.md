# 2. Related Work

## 2.1 Gradient Boosting for Credit Risk and Fraud Detection

Gradient boosting has become the dominant paradigm for tabular credit risk
modelling. Chen and Guestrin (2016) introduced XGBoost, a regularised tree
boosting system that achieves strong performance through column subsampling,
shrinkage, and cache-efficient split enumeration on sorted histograms. Ke et
al. (2017) introduced LightGBM, which replaces depth-wise tree growth with
leaf-wise growth and uses gradient-based one-side sampling to discard
low-gradient instances, yielding substantial speed advantages on large-scale
financial datasets. Both systems have become the de facto standard for
structured tabular data competitions and deployed credit risk applications.

Benchmarking studies confirm the superiority of ensemble methods over classical
alternatives in credit applications. Lessmann et al. (2015) evaluated 41
classifiers across eight public credit scoring datasets and found that gradient
boosting consistently ranked among the top performers, outperforming logistic
regression, neural networks, and support vector machines across both statistical
accuracy and expected-profit metrics. Bhattacharyya et al. (2011) reached
analogous conclusions for credit card fraud detection, where severe class
imbalance introduces an additional modelling challenge typically addressed
through synthetic oversampling (SMOTE; Chawla et al. 2002) or cost-sensitive
learning. Bahnsen et al. (2016) further demonstrated that temporal and velocity
features computed over transaction history are a primary driver of fraud
detection performance, beyond point-in-time transaction attributes.

Prokhorenkova et al. (2018) introduced CatBoost, a gradient boosting library
that addresses the target leakage problem inherent in greedy encoding of
categorical features through ordered boosting: each tree is trained on a
permutation of the data so that the target statistic for each sample is
computed only from preceding observations. CatBoost further employs symmetric
(oblivious) trees that evaluate the same split condition at every node of a
given depth level, enabling efficient GPU inference and reducing overfitting via
structural regularisation. These properties make CatBoost particularly relevant
for high-cardinality categorical features common in financial transaction data,
such as merchant codes, device identifiers, and email domains.

The comparative advantage of gradient boosting over deep learning on tabular
financial data has been examined on broad benchmarks. Shwartz-Ziv and Armon
(2022) evaluated 48 tabular datasets and found that tree-based ensembles matched
or exceeded neural network performance on the large majority, with the greatest
advantage on datasets combining continuous and categorical features. This feature
profile is characteristic of credit risk data and reinforces the choice of
XGBoost, LightGBM, and CatBoost as the three model architectures examined in
the present study.

All of the benchmarks above use static train-test splits. None evaluates
gradient boosting models in a streaming setting where the data distribution
changes after deployment. This gap motivates the temporal streaming evaluation
framework developed in Section 4.

## 2.2 Concept Drift Detection in Data Streams

Concept drift refers to changes in the joint distribution P(x, y) over time
that reduce the predictive validity of a trained model. Gama et al. (2014)
provided a comprehensive taxonomy distinguishing gradual, abrupt, incremental,
and recurring drift, together with a unified review of detection and adaptation
strategies. Webb et al. (2016) refined this taxonomy by separating virtual
drift, in which only the marginal P(x) shifts, from real drift, in which the
posterior P(y|x) changes. The distinction has practical consequences: virtual
drift may not require retraining, whereas real drift requires model adaptation
to maintain predictive validity.

Detection algorithms operate either on raw data distributions or on model
performance estimates. ADWIN (Adaptive WINdowing; Bifet and Gavaldà 2007) is
the most widely adopted performance-based detector. It maintains a sliding
window over a scalar performance measure and identifies the longest sub-window
whose mean departs significantly from the remainder, providing formal guarantees
under a binomial error model. Street and Kim (2001) proposed ensemble-based
streaming classifiers that retire stale component models and train new ones at
block boundaries, a strategy that implicitly handles gradual drift. Losing et
al. (2018) reviewed incremental online learning methods and found that
detector-triggered full retraining outperforms passive forgetting under abrupt
drift conditions, precisely the regime that ADWIN is designed to identify.

The 2024 survey by Hinder et al. (2024) identifies the temporal gap between
drift onset and detector response as a central open problem: all current methods
measure performance deterioration that has already occurred. Lu et al. (2019)
reviewed learning-under-drift approaches across domains and noted that
interpretability of drift causes remains largely absent from the literature. In
financial machine learning, monitoring practice relies on population stability
indices applied to raw feature distributions and on rolling error-rate windows.
These methods detect covariate shift reliably but do not reveal whether the
model's internal weighting of features has reorganised. No prior work has
examined whether changes in SHAP feature importance rankings carry advance
information about impending performance loss, or whether such changes can serve
as a leading indicator of drift.

## 2.3 Explainable AI in Credit Risk Governance

Regulatory and supervisory requirements have made model interpretability a
formal obligation in consumer credit. The Federal Reserve's Model Risk
Management guidance (SR 11-7, 2011) requires institutions to validate models on
an ongoing basis, monitor for performance deterioration, and produce outputs
that are interpretable to supervisors and internal auditors. The Basel
Committee's operational risk principles (2011) impose parallel transparency
requirements on models used for capital adequacy. These obligations have driven
adoption of post-hoc explanation methods across credit scoring, fraud detection,
and automated underwriting.

Two post-hoc explanation frameworks dominate deployed financial applications.
LIME (Ribeiro et al. 2016) approximates the local decision boundary around a
target instance with a locally faithful linear surrogate model. SHAP (Lundberg
and Lee 2017), grounded in Shapley values from cooperative game theory, assigns
each feature an attribution satisfying efficiency, symmetry, dummy, and
linearity axioms; the TreeSHAP variant computes exact attributions for tree
ensembles in polynomial time, making it the standard explanation method for
gradient boosting in regulated industries. Arrieta et al. (2020) surveyed the
full landscape of XAI methods and identified temporal consistency of
explanations as an open research problem: existing methods evaluate stability at
a single point in time or under controlled perturbations, not across a shifting
data distribution.

Guidotti et al. (2018) identified explanation stability as a key desideratum
alongside fidelity, observing that an explanation which changes arbitrarily
between consecutive decision windows provides no actionable governance signal.
Molnar (2022) discusses consistency of global feature importance rankings as a
practical property that analysts should assess but provides no quantitative
framework for monitoring it over time.

The most directly related prior work is Lin and Wang (2025), who examined SHAP
ranking stability in a static credit card default model across resampling
strategies and class imbalance conditions, finding significant rank variation
across experimental configurations. Their study does not examine whether SHAP
rankings shift systematically across evaluation windows as concept drift
accumulates, nor whether such shifts carry predictive information about upcoming
model performance degradation. These are precisely the questions addressed by
the SHAP Stability Index introduced in Section 3.
