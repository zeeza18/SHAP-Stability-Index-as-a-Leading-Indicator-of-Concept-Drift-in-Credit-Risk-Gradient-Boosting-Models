# 7. Conclusion

This paper introduced the SHAP Stability Index (SSI), a novel metric that
tracks the Spearman rank correlation of global SHAP feature importance vectors
across consecutive temporal evaluation windows. SSI is computed at inference
time from model outputs and unlabelled features — no ground-truth labels required.

## Key Findings

1. **SSI leads AUC degradation by ~7.8 windows on IEEE-CIS (590,540 tx).**
   XGBoost: mean 7.81 ± 4.26 windows across 16 measurable ADWIN drift events.
   LightGBM: mean 7.67 ± 4.03 windows across 15 events.
   CatBoost: mean 8.00 ± 4.11 windows across 16 events.
   All three models: Wilcoxon H₀: λ=0 rejected (p<0.001, r=0.88).
   With 7-day stride → ~55 calendar days from first SSI alert to AUC
   degradation. Crucially, SSI remains below tau=0.80 persistently from
   window 8 through window 23 (the full 105-day high-drift period) —
   a sustained governance signal, not a one-off alarm.

2. **SSI cleanly separates three drift regimes across all three model architectures:**
   - IEEE-CIS (persistent drift): 0.66–0.72 (XGBoost 0.660, LightGBM 0.683, CatBoost 0.717)
   - ULB CC Fraud (localised drift): 0.95–0.99
   - GMSC (distributional stability control, no real temporal dynamics): 0.987–0.997
   The three-cluster structure is preserved regardless of model architecture.

3. **Adaptive retraining is conditionally beneficial:**
   +0.005 AUC on high-drift IEEE-CIS (XGBoost only) but counter-productive for
   LightGBM, CatBoost, and all models on near-stable datasets. SSI should gate
   retraining decisions.

4. **CatBoost is a competitive static baseline but underperforms adaptively:**
   StaticCatBoost (0.902 IEEE-CIS, 0.998 ULB CC) matches XGBoost and approaches
   LightGBM. AdaptiveCatBoost lags due to higher per-iteration training cost
   relative to the B=5 buffer constraint.

## Limitations
- GMSC functions as a distributional stability control only — its pseudo-temporal
  ordering by demographic attributes (age ASC, utilization DESC) precludes its
  use as a drift validation dataset
- n=17–18 evaluation windows limits statistical power for AUC adaptive-vs-static
  comparison (p=0.066, r=0.45); SSI lead-time tests are significant (p<0.001)
- τ_SSI=0.80 and L=5 are empirical; sensitivity analysis across τ ∈ {0.70–0.85}
  and L ∈ {3, 5, 7} is reported in §4.3

## Future Work
- Threshold calibration for τ_SSI as function of SSI variance / drift frequency
- Multivariate SSI for feature-group level monitoring
- Label-delay scenarios (SSI especially suited — no labels needed)
- Other high-stakes domains: medical image diagnosis (cite: geetha2023skin),
  clinical governance frameworks
- Extend to HistGradientBoosting; federated drift detection

## Final Statement
SSI provides a computationally lightweight, interpretability-grounded complement
to conventional monitoring. Built on TreeSHAP values, it deploys alongside any
existing gradient boosting model risk infrastructure without pipeline modification.
Evaluated across three model architectures and three public datasets, SSI
demonstrates robust leading-indicator behaviour that generalises beyond any
single algorithmic choice.
