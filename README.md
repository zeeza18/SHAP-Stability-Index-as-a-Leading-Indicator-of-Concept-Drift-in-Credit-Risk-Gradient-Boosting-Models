# SHAP Stability Index as a Leading Indicator of Concept Drift in Credit Risk Gradient Boosting Models

**Authors:** Mohammed Azeezulla · Sakshi Balkrishna Panpatil  
**Affiliation:** College of Computing and Digital Media, DePaul University, Chicago, IL  
**Contact:** mmoha134@depaul.edu · spanpati@depaul.edu  
**Submitted to:** Applied Intelligence (Springer Nature)

---

## Overview

We introduce the **SHAP Stability Index (SSI)** — a label-free, model-agnostic metric that tracks week-to-week Spearman rank correlation of the top-20 SHAP feature-importance vectors across adjacent temporal windows. SSI provides an early-warning signal of concept drift in production credit risk models without requiring ground-truth labels.

Across three public benchmark datasets and three gradient boosting architectures, SSI **leads AUC degradation by a mean of 7.83 windows (~55 calendar days)** with *p* < 0.001 and Pearson *r* = 0.88 — while conventional Population Stability Index (PSI) never breaches its standard 0.25 warning threshold on the same data.

---

## Key Results

| Dataset | Models | SSI Lead Time (mean) | Pearson *r* | *p*-value |
|---|---|---|---|---|
| IEEE-CIS Fraud Detection | XGBoost, LightGBM, CatBoost | 7.83 windows (~55 days) | 0.88 | < 0.001 |
| ULB Credit Card Fraud | XGBoost, LightGBM, CatBoost | Localised-drift control | — | — |
| Give Me Some Credit | XGBoost, LightGBM, CatBoost | Distributional stability control | — | — |

SSI cleanly separates three drift regimes without threshold recalibration:
- **0.66 – 0.72** → persistent concept drift
- **0.95 – 0.99** → localised drift
- **≥ 0.987** → distributional stability

---

## Repository Structure

```
research-credit-drift/
│
├── src/                        # Core library
│   ├── data/                   # Loaders, feature engineering, temporal splitting
│   ├── models/                 # Static & adaptive XGBoost, LightGBM, CatBoost
│   ├── drift/                  # ADWIN detector, drift evaluator, simulator
│   ├── explainability/         # SSI computation, SHAP engine, rank shift tracker
│   ├── evaluation/             # Metrics, statistical tests, comparator
│   └── visualization/          # Figure exporters, drift timeline, SHAP heatmap
│
├── experiments/
│   ├── configs/                # YAML configs for datasets, models, ADWIN, Optuna
│   ├── tuning/                 # Optuna hyperparameter search scripts
│   ├── run_baseline.py         # Static model experiments
│   ├── run_adaptive.py         # Adaptive retraining experiments
│   ├── run_ablation.py         # SSI ablation (tau, top-k)
│   ├── run_tuning.py           # Full tuning pipeline
│   ├── analyze_results.py      # Result aggregation
│   ├── compute_psi_comparison.py
│   └── tau_sensitivity_analysis.py
│
├── scripts/                    # Figure generation scripts
│   ├── gen_fig5_lambda_hist.py
│   └── gen_fig6_rank_heatmap.py
│
├── results/
│   ├── figures/                # All 6 paper figures (PNG + PDF)
│   └── tables/                 # All result CSVs and LaTeX tables
│
├── paper/
│   ├── latex/                  # paper.tex (Springer sn-jnl class), .bbl, .cls
│   ├── references/             # references.bib
│   └── sections/               # Section drafts (Markdown)
│
├── FINAL_SUBMISSION/           # Journal-ready submission package
│   ├── latex/                  # Flat, self-contained LaTeX (ready to compile)
│   ├── figures/                # Figures as separate files (PNG + PDF)
│   └── pdf/                    # Compiled manuscript PDF
│
├── data/
│   └── raw/                    # Datasets (not tracked — see download instructions)
│
├── tests/                      # Unit tests
├── run_pipeline.py             # End-to-end pipeline entry point
├── requirements.txt
└── setup.py
```

---

## Setup

```bash
git clone https://github.com/zeeza18/SHAP-Stability-Index-as-a-Leading-Indicator-of-Concept-Drift-in-Credit-Risk-Gradient-Boosting-Models.git
cd SHAP-Stability-Index-as-a-Leading-Indicator-of-Concept-Drift-in-Credit-Risk-Gradient-Boosting-Models

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

---

## Datasets

Raw data is not included in this repository due to size and licensing. Download instructions are in [`data/raw/DOWNLOAD_INSTRUCTIONS.md`](data/raw/DOWNLOAD_INSTRUCTIONS.md).

| Dataset | Source | Size |
|---|---|---|
| [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) | Kaggle | 590,540 transactions |
| [ULB Credit Card Fraud](https://archive.ics.uci.edu/dataset/492/credit+card+fraud+detection) | UCI ML Repository | 284,807 transactions |
| [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) | Kaggle | 150,000 borrowers |

After downloading, place CSV files in `data/raw/` and run:

```bash
python src/data/loader.py --validate
```

---

## Reproducing Experiments

All experiments use `seed=42`. All hyperparameters are in `experiments/configs/`.

```bash
# 1. Hyperparameter tuning (Optuna TPE, 100 trials per model/dataset)
python experiments/run_tuning.py

# 2. Baseline static models
python experiments/run_baseline.py

# 3. Adaptive retraining (ADWIN-triggered)
python experiments/run_adaptive.py

# 4. SSI ablation study
python experiments/run_ablation.py

# 5. Aggregate results and generate tables
python experiments/analyze_results.py

# 6. Generate figures
python scripts/gen_fig5_lambda_hist.py
python scripts/gen_fig6_rank_heatmap.py
```

Every number reported in the paper traces back to a CSV in `results/tables/`.

---

## Paper

The full manuscript is available in [`FINAL_SUBMISSION/pdf/paper.pdf`](FINAL_SUBMISSION/pdf/paper.pdf).

LaTeX source: [`FINAL_SUBMISSION/latex/paper.tex`](FINAL_SUBMISSION/latex/paper.tex)

```bibtex
@article{azeezulla2026ssi,
  title   = {SHAP Stability Index as a Leading Indicator of Concept Drift
             in Credit Risk Gradient Boosting Models},
  author  = {Azeezulla, Mohammed and Panpatil, Sakshi Balkrishna},
  journal = {Applied Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

---

## Authors

| Name | Role | Affiliation |
|---|---|---|
| **Mohammed Azeezulla** | Corresponding author — methodology, experiments, SSI framework, writing | DePaul University |
| **Sakshi Balkrishna Panpatil** | Co-author — data engineering, data analysis, literature review | DePaul University |

---

## License

This repository is released for academic reproducibility. If you use this code or the SSI metric in your work, please cite the paper above.
