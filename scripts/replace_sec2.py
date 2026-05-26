"""Replace Section 2 (Related Work) in paper.tex with a clean rewrite."""

NEW_SECTION2 = r"""%% =============================================================
%% SECTION 2 --- RELATED WORK
%% =============================================================
\section{Related Work}\label{sec:related}

\subsection{Gradient Boosting for Credit Risk and Fraud Detection}
\label{sec:related:gb}

Gradient boosting has become the dominant paradigm for tabular credit risk
modelling. Chen and Guestrin~\cite{chen2016xgboost} introduced XGBoost, a
regularised tree boosting system that achieves strong performance through column
subsampling, shrinkage, and cache-efficient split enumeration on sorted feature
histograms. Ke et al.~\cite{ke2017lightgbm} introduced LightGBM, which replaces
depth-wise tree growth with leaf-wise growth and applies gradient-based one-side
sampling to discard low-gradient instances, yielding substantial speed advantages
on large-scale financial datasets. Prokhorenkova et
al.~\cite{prokhorenkova2018catboost} introduced CatBoost, which handles
categorical features natively through ordered target statistics and symmetric
oblivious trees, eliminating the target-leakage risk inherent in conventional
preprocessing. Together, these three systems constitute the dominant gradient
boosting ecosystem for deployed credit risk applications, each occupying a
distinct trade-off between training speed, memory efficiency, and categorical
feature handling.

Benchmarking studies confirm the superiority of gradient boosting over classical
alternatives for credit scoring. Lessmann et al.~\cite{lessmann2015credit}
evaluated 41 classifiers across eight public credit scoring datasets and found
that gradient boosting consistently outperformed logistic regression, neural
networks, and support vector machines on both statistical accuracy and
expected-profit metrics. Dumitrescu et al.~\cite{dumitrescu2022credit}
demonstrated that incorporating gradient boosting non-linear effects into
logistic regression frameworks improves predictive accuracy while preserving
regulatory interpretability, showing that accuracy and transparency are
complementary rather than conflicting objectives in credit scoring. Shi et
al.~\cite{shi2022credit}, reviewing 76 studies on machine-learning-driven credit
risk, confirmed gradient boosting's leading position while identifying temporal
model decay and distributional shift as among the most consequential unresolved
problems in deployed credit systems.

Credit card fraud detection presents a closely related modelling challenge.
Bhattacharyya et al.~\cite{bhattacharyya2011fraud} demonstrated the superiority
of ensemble methods for fraud classification under severe class imbalance, a
challenge typically addressed through synthetic oversampling~\cite{chawla2002smote}
or cost-sensitive weighting. Bahnsen et al.~\cite{bahnsen2016feature} showed
that temporal and velocity features computed over transaction history are a
primary performance driver beyond point-in-time attributes. Lebichot et
al.~\cite{lebichot2021incremental} demonstrated that incremental learning
strategies can match batch retraining performance in production fraud detection
pipelines operating under concept drift, providing empirical evidence that
adaptive modelling is necessary in non-stationary transaction streams.

The comparative advantage of gradient boosting over deep learning on tabular
financial data has been established on broad benchmarks. Shwartz-Ziv and
Armon~\cite{shwartzziv2022tabular} evaluated 48 datasets and found that
tree-based ensembles matched or exceeded neural network performance on the
majority, with the greatest advantage on datasets combining continuous and
categorical features. Grinsztajn et al.~\cite{grinsztajn2022why}, across 45
datasets, identified irregular feature geometry as the structural condition
under which gradient boosting retains its advantage even when neural networks
benefit from extensive architecture search. Gorishniy et
al.~\cite{gorishniy2021revisiting} showed that purpose-built transformer
architectures for tabular data cannot consistently outperform well-tuned gradient
boosting on datasets with the mixed feature profile typical of credit risk.
McElfresh et al.~\cite{mcelfresh2023tabular}, in the largest tabular
benchmarking study to date across 176 datasets and 19 algorithms, confirmed that
GBDTs retain a decisive advantage specifically on data with heavy-tailed or
irregular feature distributions --- a condition characteristic of financial
transaction data.

Beyond accuracy, gradient boosting has attracted scrutiny on fairness and
transparency. Kozodoi et al.~\cite{kozodoi2022fairness} demonstrated that
fairness-aware models can achieve competitive AUC while measurably reducing
discriminatory outcomes, at the cost of a quantifiable profit trade-off.
B{\"u}cker et al.~\cite{bucker2022transparency} concluded that the explanatory
quality of deployed credit scoring models depends critically on whether
explanation consistency is monitored over time rather than assessed only at
initial validation --- a finding that directly motivates the SSI framework.

All benchmarks above use static train-test splits. None evaluates gradient
boosting in a streaming setting where the data distribution changes after
deployment. This gap motivates the temporal streaming evaluation framework
developed in Section~\ref{sec:experiments}.

\subsection{Concept Drift Detection in Data Streams}
\label{sec:related:drift}

Concept drift refers to changes in the joint distribution $P(\mathbf{x}, y)$
over time that reduce the predictive validity of a trained model. Gama et
al.~\cite{gama2014drift} provided the foundational taxonomy, distinguishing
gradual, abrupt, incremental, and recurring drift together with a unified review
of detection and adaptation strategies. Webb et al.~\cite{webb2016drift} refined
this taxonomy by separating virtual drift, in which only the marginal
$P(\mathbf{x})$ shifts, from real drift, in which the posterior
$P(y \mid \mathbf{x})$ changes. The distinction has practical consequences:
virtual drift may not require retraining, whereas real drift requires model
adaptation to maintain predictive validity.

\paragraph{Statistical drift detectors.}
Performance-based detectors monitor model error rates and signal retraining when
a statistically significant change is detected. ADWIN (Adaptive
WINdowing)~\cite{bifet2007adwin} maintains a self-adjusting sliding window over
a scalar error signal and identifies the longest sub-window whose mean diverges
from the remainder, providing formal guarantees under a binomial error model.
The Drift Detection Method (DDM)~\cite{gama2004ddm} monitors the online error
rate and fires when it exceeds its historical minimum by more than a
Hoeffding-bound threshold; it is computationally lightweight but produces
elevated false-positive rates under gradual drift. The Early Drift Detection
Method (EDDM)~\cite{baena2006eddm} improved DDM's sensitivity to gradual drift
by monitoring the inter-error interval rather than the error rate itself. For
feature-level monitoring, KSWIN~\cite{raab2020kswin} applies a
Kolmogorov-Smirnov test to a sliding window of observations on individual input
dimensions, flagging univariate shift without requiring labelled feedback.
HDDM~\cite{frias2015hddm} provides both a drift level and a warning level
through one-tailed and two-tailed Hoeffding tests, enabling graduated response
strategies that trigger proactive monitoring before full retraining is required.
The Page-Hinkley test~\cite{page1954phtest}, a sequential change-point detector
from statistical process control, accumulates a signed deviation statistic and
triggers when the cumulative sum exceeds a threshold, making it well-suited to
detecting sustained directional shifts in a scalar monitoring signal.

These detectors constitute a family of complementary strategies differing in
statistical power under different drift velocities, sensitivity to false alarms,
and dependence on labelled feedback. ADWIN is selected for this study because
its adaptive window mechanism and formal guarantees align best with the irregular,
burst-mode drift structure observed in financial transaction streams.
River~\cite{montiel2021river}, the open-source streaming machine learning
library used in the experiments reported here, provides production-ready
implementations of all these detectors in a unified API.

\paragraph{Adaptive learning and monitoring frameworks.}
Street and Kim~\cite{street2001streaming} proposed ensemble-based streaming
classifiers that retire stale component models and train new ones at block
boundaries, implicitly handling gradual drift. Losing et
al.~\cite{losing2018incremental} reviewed incremental online learning methods
and found that detector-triggered full retraining outperforms passive forgetting
under abrupt drift --- precisely the regime ADWIN is designed to identify.
Klaise et al.~\cite{klaise2020monitoring} formalised the case for integrating
drift detection, outlier detection, and explanation monitoring as complementary
components of a single production observability pipeline, noting that no single
signal is sufficient in isolation.

\paragraph{Surveys and the detection gap.}
Bayram et al.~\cite{bayram2022drift} reviewed performance-aware drift detectors
and confirmed that all current methods measure model degradation that has already
manifested in prediction errors rather than anticipating it from upstream
signals. The 2024 two-part survey by Hinder et al.\ addresses both the detection
question~\cite{hinder2024drift} and the drift localisation and explanation
question~\cite{hinder2024partb}, collectively identifying the temporal gap
between drift onset and detector response as the central open problem in the
field. Lu et al.~\cite{lu2018drift} reviewed learning-under-drift approaches
across domains and noted that interpretability of drift causes remains largely
absent from the literature.

\paragraph{Production monitoring in financial machine learning.}
In regulatory and industry practice, the Population Stability Index (PSI) is the
most widely used tool for monitoring distributional shift in credit risk
models~\cite{siddiqi2006scorecard}. PSI compares the distribution of a model
score or input feature between a reference and a monitoring population by
discretising values into $n$ bins:
\begin{equation}
  \mathrm{PSI} = \sum_{i=1}^{n}
    \bigl(A_i - E_i\bigr)\ln\!\Bigl(\tfrac{A_i}{E_i}\Bigr),
  \label{eq:psi}
\end{equation}
where $A_i$ and $E_i$ are the actual and expected proportions in bin $i$. PSI
values above 0.25 signal instability warranting model review under
SR~11-7~\cite{sr117_2011}. The Characteristic Stability Index (CSI) applies the
same formula independently to each input feature, providing a feature-level
decomposition of covariate shift. Despite widespread regulatory acceptance, PSI
and CSI share a fundamental limitation: they monitor marginal input or score
distributions but reveal nothing about whether the model's \emph{internal
decision logic} has reorganised. A model can maintain a stable output score
distribution while relying on entirely different features to produce that score
--- a regime in which PSI reports stability while the model's effective behaviour
has changed. This is the central motivating gap for SSI.

\paragraph{The label delay problem.}
In production credit systems, ground-truth outcome labels are unavailable for 30
to 90 days or more after the credit decision. Sculley et
al.~\cite{sculley2015debt} identified label delay and temporal model decay as
primary sources of technical debt in deployed machine learning systems, noting
that silent deterioration under distribution shift is among the most costly and
difficult-to-detect operational failures. This delay means that performance-based
detectors cannot provide a contemporaneous monitoring signal. Sethi and
Kantardzic~\cite{sethi2015margin} proposed Margin Density Drift Detection (MD3),
which monitors the fraction of instances near a classifier's decision boundary as
a label-free drift proxy, requiring labels only at retraining time. Grzenda et
al.~\cite{grzenda2020delayed} formalised the label delay problem for evolving
data streams and showed that the effective detection lag of error-based methods
grows proportionally with the delay window. The SSI metric introduced in this
paper addresses this problem directly: it operates on model outputs and
unlabelled feature vectors at inference time, producing a contemporaneous signal
without requiring resolved outcome feedback.

\subsection{Explainable AI in Credit Risk Governance}
\label{sec:related:xai}

Regulatory and supervisory requirements have made model interpretability a
formal obligation in consumer credit. The Federal Reserve's Model Risk
Management guidance (SR~11-7)~\cite{sr117_2011} requires institutions to
validate models on an ongoing basis, monitor for performance deterioration,
and produce outputs interpretable to supervisors and auditors. The Basel
Committee's operational risk principles~\cite{basel2011} impose parallel
transparency requirements on models used for capital adequacy. The European Union
Artificial Intelligence Act~\cite{eu_ai_act2024} classifies credit scoring as a
high-risk AI application subject to mandatory transparency, traceability, and
human oversight requirements throughout the entire operational lifetime of the
model. This creates a direct regulatory basis for ongoing explanation stability
monitoring: a model whose SHAP feature importance rankings shift substantially
over time may be non-compliant with the Act's traceability provisions even if
aggregate predictive performance remains acceptable.

Two post-hoc explanation frameworks dominate deployed financial applications.
LIME~\cite{ribeiro2016lime} approximates the local decision boundary with a
locally faithful linear surrogate. SHAP~\cite{lundberg2017shap}, grounded in
Shapley values from cooperative game theory, assigns each feature an attribution
satisfying efficiency, symmetry, dummy, and linearity axioms. Sundararajan and
Najmi~\cite{sundararajan2020shapley} characterised the full space of Shapley
value operationalisations for model explanation, showing that the choice of
background distribution materially affects attribution outputs and that no single
operationalisation is uniquely correct. Covert et al.~\cite{covert2021removing}
unified SHAP, LIME, and permutation importance under a single removal-based
framework, establishing a common theoretical foundation for comparing their
properties. The TreeSHAP variant~\cite{lundberg2020trees} computes exact
attributions for tree ensembles in polynomial time, making it the standard
explanation method for gradient boosting in regulated industries. Arrieta et
al.~\cite{arrieta2020xai} surveyed the full landscape of XAI methods and
identified temporal consistency of explanations as an open research problem:
existing methods evaluate stability at a single point in time or under controlled
perturbations, not across a shifting data distribution.

The stability and fragility of post-hoc explanations have been examined from
several angles with direct governance implications. Alvarez-Melis and
Jaakkola~\cite{alvarez2018robustness} defined a self-consistency criterion
requiring that similar inputs produce similar explanations and showed that neither
LIME nor SHAP satisfies it uniformly, establishing precedent for treating
explanation stability as a measurable governance property. Slack et
al.~\cite{slack2020fooling} demonstrated that both methods can be adversarially
manipulated to conceal discriminatory behaviour behind innocuous-looking
explanations, underscoring that static explanation audits are insufficient for
deployed systems under distribution shift. Fryer et al.~\cite{fryer2021shapley}
showed that Shapley values for feature selection can produce misleading importance
rankings under correlated features --- a common condition in credit data ---
indicating that attribution instability is a concern even before distributional
shift is introduced. Molnar et al.~\cite{molnar2022pitfalls} catalogued general
pitfalls of model-agnostic interpretation methods, including failure to account
for feature dependence and production of spurious attributions in high-dimensional
settings, concluding that one-time explanation audits provide insufficient
governance coverage for production systems.

In the credit domain specifically, Bussmann et al.~\cite{bussmann2021credit}
integrated SHAP attribution monitoring into an end-to-end credit risk management
framework, treating SHAP values as first-class governance artefacts alongside
AUC and Gini coefficient and demonstrating the practical utility of feature-level
explanation tracking in production. Guidotti et al.~\cite{guidotti2018survey}
identified explanation stability as a key desideratum alongside fidelity,
observing that explanations which change arbitrarily between consecutive decision
windows provide no actionable governance signal. Molnar~\cite{molnar2022iml}
discusses global feature importance consistency as a practical property analysts
should assess but provides no quantitative framework for monitoring it over time.

The most directly related prior work is Lin and Wang~\cite{lin2025shap}, who
examined SHAP ranking stability in a static credit card default model across
resampling strategies and class imbalance conditions, finding significant rank
variation across experimental configurations. Their study is restricted to a
single cross-sectional model under controlled perturbations and does not examine
whether SHAP rankings shift systematically across temporal evaluation windows as
concept drift accumulates, nor whether such shifts carry predictive information
about upcoming performance degradation before it surfaces in outcome metrics.
These are precisely the questions addressed by the SHAP Stability Index
introduced in Section~\ref{sec:method}.

"""

with open('paper/latex/paper.tex', encoding='utf-8') as f:
    content = f.read()

sec2_start = content.find('%% SECTION 2')
sec3_start = content.find('%% SECTION 3')

new_content = content[:sec2_start] + NEW_SECTION2 + content[sec3_start:]

with open('paper/latex/paper.tex', 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f'Done. New file: {len(new_content)} chars')
