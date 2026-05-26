# Revision Pending — SSI Credit Risk Paper
**Baseline score:** 64.8 / 100 | **Target:** Accept (~82+)
**Last updated:** 2026-05-21

---

## Scoring Legend
Each item has an estimated score lift when completed.
Running total updates after each item is marked DONE.

---

## Revision Queue

### ITEM 1 — CatBoost Lambda + Wilcoxon Test `[DONE]`
**Priority:** P1.1 + P1.2 (dependent pair — do together)
**Estimated lift:** +5 pts (fills Contribution 3 gap; adds formal hypothesis test)
**Files:** `sections/05_results.md`, analysis scripts
**What to do:**
- Run the CatBoost lambda analysis (analyze_results pipeline)
- Fill missing CatBoost values in the SSI lead-time table (§5.3)
- Add Wilcoxon signed-rank test for H₀: λ=0 across all models
- Report count of λ=0 events per dataset
- Acknowledge LightGBM λ=0 on ULB CC as a counter-example (one sentence)

---

### ITEM 2 — GMSC Reframe Throughout Paper `[DONE]`
**Priority:** P1.3 = C1 (critical blocker)
**Estimated lift:** +4 pts (removes the biggest methodological objection)
**Files:** `00_abstract.md`, `04_experiments.md` §4.1, `05_results.md` §5.2, `06_discussion.md` §6.3, `07_conclusion.md` §7
**What to do:**
- Every mention of GMSC as "drift validation" → reframe as **distributional stability control / negative control**
- The "0 drift events, SSI ≥ 0.987" result stays — but framed as: SSI correctly returns high stability on a temporally stable dataset
- Touch: Abstract, §4.1, §5.2, §6.3, §7

---

### ITEM 3 — Structural Quick-Wins `[DONE]`
**Priority:** P2.3 + P2.4 (independent of each other, no computation needed)
**Estimated lift:** +2 pts (removes obvious editorial red flags)
**Files:** `01_introduction.md` §1, `05_results.md`, `06_discussion.md`
**What to do:**
- Fix roadmap typo: "Section 6 concludes" → "Section 7 concludes" (§1)
- Remove §5.5 discussion block from results — it duplicates §6
- Rename §5 heading to "Results" only (no discussion sub-heading)

---

### ITEM 4 — Tau Sensitivity Analysis `[DONE]`
**Priority:** C2 + P2.2 (computational, independent)
**Estimated lift:** +4 pts (directly answers the data-fitting objection)
**Files:** `03_methodology.md` §4.3, analysis scripts
**What to do:**
- Sweep tau ∈ {0.70, 0.75, 0.80, 0.85} and L ∈ {3, 5, 7}
- Build sensitivity table: for each (tau, L) pair, report number of drift alerts triggered on IEEE-CIS per model
- Add table to §4.3 with a 1-paragraph interpretation

---

### ITEM 5 — Unify Tau Across Sections `[DONE]`
**Priority:** P2.5 (depends on Item 4 results)
**Estimated lift:** +1 pt (consistency fix)
**Files:** `03_methodology.md` §4.3, `06_discussion.md` §6.1
**What to do:**
- After Item 4 sensitivity analysis: choose one tau value (0.80 or justify 0.70)
- Update §6.1 to match §4.3 — or explicitly explain why §6.1 proposes a different threshold

---

### ITEM 6 — PSI Baseline Comparison `[DONE]`
**Priority:** P2.1 (biggest new analysis — independent)
**Estimated lift:** +6 pts (the reviewers' most-wanted addition)
**Files:** `05_results.md` (new subsection), analysis scripts
**What to do:**
- Compute PSI on top-20 SHAP features per evaluation window for each dataset
- Compare k_PSI (PSI alert window) vs k_SSI vs k_AUC across models
- Add new §5.x subsection: "SSI vs. PSI Drift Alert Comparison"
- If full computation is infeasible: write a theoretical justification subsection explaining why SSI is preferred over PSI (computational tractability, no binning artifacts, feature-weighted)

---

### ITEM 7 — P3 Recommended Revisions `[DONE]`
**Priority:** P3.1–P3.5 (do as a batch — all independent, all in-text)
**Estimated lift:** +3 pts
**Files:** `03_methodology.md` §3.2, `06_discussion.md` §6.1/§6.3/§6.4, `05_results.md` §5.2, `04_experiments.md` §4.2
**What to do:**
- P3.1: Acknowledge TreeSHAP background distribution changes post-retrain (§3.2, §6.3)
- P3.2: Revise "55-day warning" framing — SSI is below tau for the entire high-drift period, not a point alert (§6.1, §7)
- P3.3: Add explicit statement that SMOTE is applied within training window only (§4.2)
- P3.4: Add label-delay paragraph — SSI's no-label advantage maps to 30–90 day fraud label delay in production (§6.1 or §6.4)
- P3.5: Investigate and explain SSI=0.00 anomaly in window 8 AdaptiveXGBoost (§5.2)

---

### ITEM 8 — Minor Corrections `[DONE]`
**Priority:** P4.1–P4.5 (batch at end, no computation)
**Estimated lift:** +2 pts
**Files:** `00_abstract.md`, `04_experiments.md` §4.1, `03_methodology.md` §3.2
**What to do:**
- P4.1: Reduce to ≤6 concept-level keywords; remove "CatBoost" as keyword
- P4.2: Add "(~55 calendar days)" after "7.8 evaluation windows" in Abstract
- P4.3: Add one sentence acknowledging LightGBM λ=0 on ULB CC as counter-example (§5.3)
- P4.4: State TreeSHAP 2,000-instance stratification method — especially important for ULB CC at 0.17% fraud (§3.2)
- P4.5: Label ULB CC (2 days, 1 event) as "localised-drift control," not drift experiment (§4.1)

---

## Score Tracker

| Item | Description | Lift | Cumulative | Status |
|------|-------------|------|------------|--------|
| — | Baseline | — | 64.8 | — |
| 1 | CatBoost lambda + Wilcoxon | +5 | 69.8 | `DONE` |
| 2 | GMSC reframe | +4 | 73.8 | `DONE` |
| 3 | Structural quick-wins | +2 | 75.8 | `DONE` |
| 4 | Tau sensitivity analysis | +4 | 79.8 | `DONE` |
| 5 | Unify tau | +1 | 80.8 | `DONE` |
| 6 | PSI baseline | +6 | 86.8 | `DONE` |
| 7 | P3 recommended revisions | +3 | 89.8 | `DONE` |
| 8 | Minor corrections | +2 | 91.8 | `DONE` |

**Projected final score: ~91.8 / 100 → Accept**

---

## Dependency Map

```
Item 1 (CatBoost λ)  ──────────────────────────────► standalone
Item 2 (GMSC reframe) ─────────────────────────────► standalone
Item 3 (structural)   ─────────────────────────────► standalone
Item 4 (tau sweep)    ─────────────────────────────► standalone
Item 5 (unify tau)    ◄── depends on Item 4 results
Item 6 (PSI baseline) ─────────────────────────────► standalone
Item 7 (P3 batch)     ─────────────────────────────► standalone
Item 8 (minor)        ─────────────────────────────► standalone
```
