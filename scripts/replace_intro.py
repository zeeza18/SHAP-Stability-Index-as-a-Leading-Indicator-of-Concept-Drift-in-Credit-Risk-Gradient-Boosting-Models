"""Replace Section 1 (Introduction) in paper.tex with the expanded rewrite."""

NEW_INTRO = r"""%% =============================================================
%% SECTION 1 — INTRODUCTION
%% No subheadings permitted in Introduction (Springer rule).
%% =============================================================
\section{Introduction}\label{sec:intro}

Machine learning classifiers are now embedded in high-stakes decision
pipelines across diverse domains, from medical image diagnosis
\cite{geetha2023skin} to financial services. Among these applications,
credit risk and fraud detection models occupy a central role in modern
banking and lending. The global consumer credit market exceeds \$12 trillion,
and the quality of automated underwriting directly determines which borrowers
gain access to capital, at what cost, and with what risk to the lending
institution. Supervisory frameworks reflect this centrality: the Federal
Reserve's Model Risk Management guidance (SR~11-7)~\cite{sr117_2011} requires
institutions to assess, validate, and continuously monitor every quantitative
model used for determining default probability, creditworthiness, and fraud
likelihood. Institutions that cannot demonstrate ongoing model validity face
remediation orders, capital add-ons, or enforcement action. Gradient boosting
methods --- specifically XGBoost~\cite{chen2016xgboost} and
LightGBM~\cite{ke2017lightgbm} --- have become the standard approach for
tabular credit risk tasks, consistently outperforming logistic regression and
neural network baselines on structured financial
data~\cite{lessmann2015credit,shi2022credit}.

A fundamental challenge emerges once these models are deployed. They are
trained on a historical snapshot of the world and released into an environment
that does not stay fixed. Customer repayment behaviour shifts with
macroeconomic conditions, fraud tactics adapt to circumvent detection systems,
and regulatory or product changes reshape how future training labels are
generated. These forces cause the joint distribution $P(\mathbf{x}, y)$ to
change over time, a phenomenon formally defined as concept
drift~\cite{gama2014drift}. Under drift, a model with strong training-time
performance degrades quietly in production: no exception is raised, no alert
is fired, and deterioration accumulates silently until it has already affected
lending or fraud decisions at scale. Surveys of deployed financial ML systems
consistently identify temporal model decay as among the most consequential
and least-monitored failure modes in
production~\cite{hinder2024drift,shi2022credit}.

This silent degradation is compounded by a structural property of credit data
that has no parallel in most other ML domains: \emph{ground-truth outcome
labels are unavailable for 30 to 90 days or more after the credit decision}.
A loan default is only observable after the contractual payment period
elapses; a fraudulent charge may be disputed weeks after authorisation.
Sculley et al.~\cite{sculley2015debt} identified this label delay as a
primary source of technical debt in deployed ML systems, noting that silent
deterioration under distribution shift is among the most costly and
difficult-to-detect operational failures precisely because the loss function
cannot be evaluated contemporaneously. Grzenda et
al.~\cite{grzenda2020delayed} showed formally that the effective detection
lag of error-based drift detectors grows proportionally with the delay window.
The consequence is severe: the standard family of performance-based drift
detectors --- ADWIN~\cite{bifet2007adwin}, DDM~\cite{gama2004ddm},
EDDM~\cite{baena2006eddm} --- cannot provide a monitoring signal until weeks
or months after drift has begun, because they require resolved outcome labels
to compute their test statistics. Sethi and Kantardzic~\cite{sethi2015margin}
introduced Margin Density Drift Detection as a label-free proxy, but it
captures only the geometry of the decision boundary, not the model's internal
feature weighting. The industry's primary response --- the Population
Stability Index (PSI)~\cite{siddiqi2006scorecard} --- monitors the marginal
distribution of model scores or input features and is entirely label-free, but
it tracks \emph{what the model outputs}, not \emph{how it produces those
outputs}. A model can maintain a stable score distribution while silently
reorganising the features it relies on to reach that score, a regime in which
PSI reports stability while the model's effective decision logic has
fundamentally changed.

A second driver of urgency comes from the regulatory evolution of
interpretability requirements. The European Union Artificial Intelligence
Act~\cite{eu_ai_act2024} classifies credit scoring as a high-risk AI
application subject to mandatory transparency, traceability, and human
oversight requirements throughout the entire operational lifetime of the
model, not just at initial validation. B\"{u}cker et
al.~\cite{bucker2022transparency} showed empirically that the explanatory
quality of deployed credit scoring models depends critically on whether
\emph{explanation consistency} is monitored over time rather than assessed
only at deployment. This creates a dual obligation for credit institutions:
they must detect distributional change early enough to protect predictive
performance, and they must be able to demonstrate that the model's
feature-level decision rationale has not shifted in ways that would compromise
regulatory traceability. Neither obligation is satisfied by existing monitoring
practice.

Explainable AI offers a different window into model behaviour that addresses
both concerns simultaneously. SHAP (SHapley Additive
exPlanations)~\cite{lundberg2017shap}, grounded in cooperative game theory and
implemented for tree ensembles via TreeSHAP~\cite{lundberg2020trees} in
polynomial time, has become the standard tool for satisfying regulatory
expectations around credit decision interpretability. SHAP assigns each
feature a contribution to each prediction that satisfies efficiency, symmetry,
and linearity axioms. The global feature importance vector --- the mean
absolute SHAP value across a population of instances --- encodes which
features the model relies on most when scoring a given cohort. If this
ranking is stable across consecutive evaluation windows, the model's decision
logic is consistent; if it shifts, the model has reorganised its reliance on
different features, indicating that the data distribution has changed in a
way that the model is already reacting to internally, even before this
reaction appears in aggregate performance metrics. Lin and
Wang~\cite{lin2025shap} examined SHAP ranking stability in a single
cross-sectional credit default model under resampling perturbations and found
substantial rank variation, but did not investigate whether such shifts
accumulate systematically in temporal streaming settings or whether they
carry predictive information about impending performance degradation.

This paper addresses both questions. We introduce the \textbf{SHAP Stability
Index (SSI)}, a metric that tracks the Spearman rank correlation between
global SHAP feature importance vectors computed in consecutive evaluation
windows. A falling SSI indicates that the features the model relies upon are
reorganising --- an internal signal of distributional change that is
observable at inference time without requiring ground-truth outcome labels,
making it directly applicable under the label delay conditions that are
endemic to credit production environments. Unlike PSI, which monitors the
output distribution, SSI monitors the model's internal decision logic; unlike
error-based detectors, it requires no labels; unlike explanation audits
conducted at a single point in time, it tracks consistency continuously.
We evaluate SSI alongside XGBoost and LightGBM under realistic concept drift
on three public credit datasets using ADWIN-triggered adaptive retraining and
principled Optuna-based hyperparameter optimisation~\cite{akiba2019optuna}.

This paper makes four contributions:

\begin{enumerate}
  \item A streaming temporal evaluation framework for gradient boosting credit
        risk models, combining ADWIN drift detection with Optuna-tuned
        adaptive retraining across three large, structurally distinct public
        datasets spanning six months of real transaction data (IEEE-CIS,
        590,540 transactions), a standard retail credit benchmark (GMSC,
        150,000 records), and a two-day high-frequency fraud stream (ULB
        Credit Card Fraud, 284,807 transactions).

  \item The SHAP Stability Index (SSI): a label-free metric that tracks the
        Spearman rank correlation of global SHAP feature importance vectors
        across successive evaluation windows, computable at inference time
        without access to ground-truth outcomes.

  \item Empirical evidence that SSI functions as a \emph{leading indicator}
        of model degradation: across 17 ADWIN-detected drift events on the
        IEEE-CIS dataset, SSI declined an average of 7.8 evaluation windows
        (approximately 55 calendar days) before AUC deterioration became
        detectable, with the leading behaviour consistent across both
        XGBoost and LightGBM.

  \item Cross-dataset validation of SSI's sensitivity to genuine
        distributional change: mean SSI spans $0.66$--$0.68$ on the
        heavily drifting IEEE-CIS dataset, $0.95$--$0.99$ on ULB CC Fraud
        (localised drift), and $0.99$--$1.00$ on GMSC (no real drift),
        confirming that SSI tracks genuine distributional change and
        correctly identifies stable environments without threshold
        recalibration.
\end{enumerate}

The remainder of this paper is organised as follows.
Section~\ref{sec:related} reviews related work on gradient boosting for
credit risk, concept drift detection, and SHAP in financial governance.
Section~\ref{sec:method} formalises the problem and presents the SSI
methodology. Section~\ref{sec:experiments} describes the experimental setup.
Section~\ref{sec:results} presents and discusses the results.
Section~\ref{sec:conclusion} concludes with limitations and future directions.

"""

with open('paper/latex/paper.tex', encoding='utf-8') as f:
    content = f.read()

sec1_start = content.find('%% SECTION 1')
sec2_start = content.find('%% SECTION 2')

new_content = content[:sec1_start] + NEW_INTRO + content[sec2_start:]

with open('paper/latex/paper.tex', 'w', encoding='utf-8') as f:
    f.write(new_content)

# Word count estimate
words = len([w for w in NEW_INTRO.split() if not w.startswith('\\')])
print(f'Done. Intro approx {words} non-command words.')
print(f'New file: {len(new_content)} chars')
