# 6. Discussion

## 6.1 SSI as a Model Governance Metric

The SHAP Stability Index offers a fundamentally new type of monitoring signal
for model risk management teams. Unlike error-rate monitors that are *reactive*
(they measure degradation after it has occurred), SSI is *proactive*: a
sustained drop in SSI indicates that the model's internal feature weighting is
shifting, even if AUC has not yet fallen below a performance threshold.

In practice, a model risk team could set an SSI alert threshold (e.g., SSI < 0.80
as evaluated across L=5 consecutive windows, consistent with §4.3) to trigger a
formal model review, rather than waiting for AUC to breach a performance floor.
The tau sensitivity analysis (§4.3) demonstrates that tau=0.80 maximises alert
coverage while keeping k_SSI=8 stable across all three model architectures;
lower thresholds (tau ≤ 0.70) reduce sensitivity and delay CatBoost alerts by
eight windows. This reframes explanation stability as a first-class operational
metric alongside accuracy-based KPIs.

It is important to note that on IEEE-CIS, SSI does not fire as a single point
alert at window 8 and then recover. Rather, SSI remains persistently below
tau=0.80 from window 8 through window 23 — spanning the entire 105-day
high-drift period. The 7.8-window lead time (~55 calendar days) characterises
when SSI first falls below the threshold ahead of AUC degradation; the
sustained below-threshold signal provides continuous governance coverage
throughout the drift episode, not a one-off alarm that could be dismissed as
noise.

## 6.2 Why Adaptive Retraining Works — and When It Does Not

Adaptive retraining is most effective when drift is gradual rather than abrupt.
When ADWIN fires after several windows of accumulating error, the window buffer
contains relevant recent data that reflects the new distribution. When drift is
sudden (a single-window step change), the buffer may contain pre-drift data that
hurts retraining. This is reflected in the variance of post-drift AUC recovery
across windows.

## 6.3 Limitations

**GMSC as distributional stability control, not drift validation**: GMSC contains
no native timestamp. The pseudo-temporal ordering (age ASC, utilization DESC)
creates a demographic gradient — not a temporal sequence — so the observed
SSI ≥ 0.987 and zero ADWIN events cannot be cited as cross-dataset validation
of drift detection behaviour. Throughout this paper GMSC serves exclusively as
a distributional stability control (negative control): it confirms that SSI does
not produce spurious low-SSI alerts when the data are smoothly ordered by
demographic attributes. Any claim that "SSI correctly identifies no-drift on
GMSC" would conflate distributional smoothness with temporal stability.

**TreeSHAP background distribution shifts post-retrain**: After each
ADWIN-triggered retraining episode, TreeSHAP is applied to the newly retrained
model. The internal SHAP background distribution therefore changes at retrain
boundaries, meaning consecutive SSI values on either side of a retrain reflect
rank stability of different model instances rather than a fixed reference model.
This is appropriate for monitoring the live deployed model but means SSI values
immediately post-retrain (e.g., window 8 for AdaptiveXGBoost) may reflect
reorganisation relative to a structurally different predecessor (see §5.2).

**CC Fraud PCA features**: V1-V28 are PCA-transformed components, which makes
individual feature names meaningless for domain interpretation. SSI rank
stability is technically valid, but the top features cannot be interpreted by
practitioners.

**Single hyperparameter set per model-dataset pair**: We tune one set of
hyperparameters per model and dataset via Optuna TPE. Ensemble approaches or
model stacking might yield higher absolute AUC but are outside our scope.

**CatBoost adaptive retraining cost**: CatBoost's ordered boosting procedure
is computationally heavier per iteration than XGBoost or LightGBM, which makes
its post-drift recovery slower under the B=5 window buffer constraint. Studies
with larger buffers or reduced iteration counts during retrain may narrow this
gap.

## 6.4 Future Work

**Label delay in production credit risk.** Fraud and default labels are
typically unavailable for 30–90 days after transaction origination, due to
dispute resolution windows, chargeback processing, and reporting lag. During
this interval, error-rate monitors and loss-based drift detectors cannot
fire — they require labels. SSI is computed from model outputs on unlabelled
incoming data and is therefore fully operational throughout the label-
unavailable window. For a system with a 60-day labelling delay, SSI's ~55
calendar-day lead time means a governance signal fires before any label-based
monitor could even begin to compute — making it especially well-suited to
production credit risk deployment.

Online learning models (Hoeffding trees, EFDT) that update incrementally
without full retraining present an interesting alternative to the
retrain-from-scratch approach. Federated drift detection — where each bank
shares only drift signals rather than raw data — could extend SSI to privacy-
preserving settings. Multi-class credit risk (delinquency grade, LGD
estimation) would require extending SSI to regression SHAP values.
HistGradientBoosting (scikit-learn) and TabNet represent additional gradient
boosting variants whose SSI dynamics under drift remain unexplored.
