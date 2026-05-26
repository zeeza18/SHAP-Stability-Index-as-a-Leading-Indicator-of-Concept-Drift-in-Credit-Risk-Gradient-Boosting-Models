# Peer Review Report
**Paper:** SHAP Stability Index (SSI) for Concept Drift Monitoring in Credit Risk Models
**Venue:** Springer Applied Intelligence
**Review date:** 2026-05-21
**Mode:** full (5-reviewer panel)
**Decision: MAJOR REVISION**

---

## Reviewer Panel

| # | Role | Persona | Focus |
|---|---|---|---|
| 1 | Editor-in-Chief | Prof. Halim Doğan | Journal fit, originality, significance |
| 2 | Methodology Reviewer | Dr. Elena Vasquez | Statistical validity, temporal CV, ablations |
| 3 | Domain Reviewer | Dr. Marcus Chen | Credit risk literature, regulatory framing, PSI baseline |
| 4 | XAI Perspective | Dr. Aiko Tanaka | SHAP theory, Spearman design, TreeSHAP assumptions |
| 5 | Devil's Advocate | Dr. Rohan Bhatt | Core claim scrutiny, GMSC validity, CatBoost gap |

---

## Overall Scores

| Reviewer | Overall Score | Recommendation |
|---|---|---|
| EIC (Doğan) | 72 / 100 | Minor-to-Major Revision |
| Methodology (Vasquez) | 62 / 100 | Major Revision |
| Domain (Chen) | 70 / 100 | Minor-to-Major Revision |
| XAI Perspective (Tanaka) | 68 / 100 | Minor Revision + required additions |
| Devil's Advocate (Bhatt) | 52 / 100 | Major Revision |
| **Panel average** | **64.8 / 100** | **MAJOR REVISION** |

---

## CRITICAL Issues (Devil's Advocate — block Accept)

### C1. GMSC pseudo-temporal ordering invalidates drift claims
GMSC has no real timestamp. Sorting by age (asc) + revolving utilization (desc) creates a demographic gradient, not a temporal sequence. The "0 drift events, SSI ≥ 0.987" finding on GMSC is trivially explained by the artificial ordering. GMSC cannot be used to claim cross-dataset validation of drift behaviour.

**Fix:** Reframe GMSC as a distributional stability control / negative control throughout the paper (Abstract, §4.1, §5.2, §6.3, §7).

### C2. tau=0.80 threshold is data-fitted without out-of-sample validation
tau=0.80 is chosen such that all IEEE-CIS SSI values fall below it and all GMSC/ULB values stay above it — using the same data the claim is evaluated on. No sensitivity analysis exists.

**Fix:** Add threshold sensitivity analysis for tau ∈ {0.70, 0.75, 0.80, 0.85} and L ∈ {3, 5, 7}.

---

## Required Revisions (P1–P2)

### P1 — Critical (must fix before resubmission)

| # | Issue | Location | Fix |
|---|---|---|---|
| P1.1 | CatBoost lambda values missing from lead-time table | §5.3 | Run analyze_results pipeline for CatBoost; fill table |
| P1.2 | No formal test of H₀: λ = 0 | §5.4 | Add Wilcoxon signed-rank test on lambda values; report count of λ=0 events |
| P1.3 | GMSC reframed throughout | Abstract, §4.1, §5.2, §6.3, §7 | Label as "distributional stability control," not drift validation |

### P2 — Required (blocks publication)

| # | Issue | Location | Fix |
|---|---|---|---|
| P2.1 | No SSI vs. PSI/KL baseline comparison | §5, new subsection | Compute PSI on top-20 features per window; compare k_PSI vs k_SSI vs k_AUC; OR add explicit theoretical discussion |
| P2.2 | Threshold sensitivity analysis missing | §4.3 | tau/L sensitivity table |
| P2.3 | Introduction roadmap error | §1 | "Section 6 concludes" → "Section 7 concludes" |
| P2.4 | §5.5 + §6 structural redundancy | §5, §6 | Remove §5.5 discussion block; keep §6 as dedicated Discussion; rename §5 "Results" |
| P2.5 | tau inconsistency: §4.3 uses 0.80, §6.1 proposes 0.70 | §6.1 | Unify to single threshold or explain difference |

---

## Strongly Recommended Revisions (P3)

| # | Issue | Location | Fix |
|---|---|---|---|
| P3.1 | TreeSHAP background distribution confound under adaptive retrain | §3.2, §6.3 | Acknowledge: AdaptiveX SSI comparisons span different SHAP backgrounds post-retrain |
| P3.2 | Alert hysteresis framing | §6.1, §7 | SSI on IEEE-CIS is below tau for entire high-drift period — not a discrete alert; revise "55-day warning" framing |
| P3.3 | SMOTE temporal leakage clarification | §4.2 | State explicitly that SMOTE is applied within training window only |
| P3.4 | Label-delay paragraph | §6.1 or §6.4 | Connect SSI's no-label advantage to 30–90 day fraud label delay in production |
| P3.5 | SSI=0.00 anomaly in window 8 (AdaptiveXGBoost) | §5.2 | Investigate and explain; likely numerical issue post-retrain |

---

## Minor Corrections (P4)

| # | Issue | Location | Fix |
|---|---|---|---|
| P4.1 | Keywords: >6 keywords, "CatBoost" is a model name | Abstract | Reduce to ≤6 concept-level keywords |
| P4.2 | Abstract doesn't state calendar-day equivalent | Abstract | Add "(~55 calendar days)" after "7.8 evaluation windows" |
| P4.3 | lambda=0 for LightGBM on ULB CC not discussed | §5.3 | Add sentence acknowledging counter-example |
| P4.4 | TreeSHAP 2,000-instance stratification criterion unspecified | §3.2 | State stratification method (especially for ULB CC at 0.17% fraud) |
| P4.5 | ULB CC Fraud described as drift experiment (2 days, 1 event) | §4.1 | Label as localised-drift control |

---

## What Survives Scrutiny (Keep These Strengths)

1. **Three-cluster SSI regime separation** (0.66/0.95/0.99) — robust result across all 3 architectures; will survive revision
2. **SSI formulation** — mathematically well-defined, computationally tractable, grounded in Spearman theory
3. **No-label-required property** — genuine practical advantage; will strengthen with label-delay framing (P3.4)
4. **Reproducibility infrastructure** — seeds, versions, JSON serialisation are above-average for applied ML
5. **Governance framing (§6.1)** — SSI as a proactive model governance metric is the paper's strongest practical contribution

---

## Decision Summary

> **MAJOR REVISION** — The paper introduces a genuinely novel and practical contribution to model monitoring. It is publishable in Applied Intelligence contingent on: computing CatBoost lambda (P1.1), adding the λ>0 hypothesis test (P1.2), reframing GMSC (P1.3), adding a PSI comparison or equivalent (P2.1), and fixing the structural/roadmap errors (P2.3–P2.5). The three-cluster SSI separation result is robust and will anchor the revised paper. Panel expects re-review after revisions.
