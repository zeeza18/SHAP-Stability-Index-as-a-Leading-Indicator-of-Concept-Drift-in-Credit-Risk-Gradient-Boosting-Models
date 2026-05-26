"""Compute tau sensitivity table from existing per-window SSI data.

Outputs:
  results/tables/tau_sensitivity.csv
  results/tables/lead_time_percentiles.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"

# ── Load per-window SSI for both adaptive models on IEEE-CIS ────────────────
xgb = pd.read_csv(TABLES / "adaptive_AdaptiveXGBoost_ieee_cis.csv")
lgb = pd.read_csv(TABLES / "adaptive_AdaptiveLightGBM_ieee_cis.csv")

xgb_ssi = xgb.set_index("window_index")["ssi"].fillna(0)
lgb_ssi = lgb.set_index("window_index")["ssi"].fillna(0)

# ── Load k_AUC per drift event (from existing lead-time table) ──────────────
lt = pd.read_csv(TABLES / "ssi_lead_time.csv")
ieee_lt = lt[lt["dataset"].str.contains("IEEE")]

xgb_lt = ieee_lt[ieee_lt["model"] == "AdaptiveXGBoost"].copy()
lgb_lt  = ieee_lt[ieee_lt["model"] == "AdaptiveLightGBM"].copy()

windows = sorted(xgb_ssi.index)
n_windows = len(windows)

def compute_tau_stats(ssi_series, lt_df, tau):
    """Return (k_ssi, mean_lambda, pct_flagged) for a given tau."""
    flagged = ssi_series[ssi_series < tau]
    pct_flagged = 100 * len(flagged) / n_windows

    # k_SSI = first window where SSI < tau
    k_ssi = flagged.index.min() if len(flagged) > 0 else None

    # Recompute lead times: lambda = k_AUC - k_SSI for each drift event
    if k_ssi is None:
        mean_lam = float("nan")
    else:
        lams = []
        for _, row in lt_df.iterrows():
            k_auc = row["auc_drop_window"]
            lam = k_auc - k_ssi
            lams.append(lam)
        mean_lam = float(np.mean(lams))

    return k_ssi, mean_lam, pct_flagged

taus = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
rows = []
for tau in taus:
    x_k, x_lam, x_pct = compute_tau_stats(xgb_ssi, xgb_lt, tau)
    l_k, l_lam, l_pct = compute_tau_stats(lgb_ssi, lgb_lt, tau)
    rows.append({
        "tau": tau,
        "xgb_k_ssi": x_k,
        "xgb_mean_lambda": round(x_lam, 2) if not np.isnan(x_lam) else "—",
        "xgb_pct_flagged": round(x_pct, 0),
        "lgb_k_ssi": l_k,
        "lgb_mean_lambda": round(l_lam, 2) if not np.isnan(l_lam) else "—",
        "lgb_pct_flagged": round(l_pct, 0),
    })

tau_df = pd.DataFrame(rows)
tau_df.to_csv(TABLES / "tau_sensitivity.csv", index=False)
print("Tau sensitivity:")
print(tau_df.to_string(index=False))

# ── Augmented lead-time percentiles ─────────────────────────────────────────
ieee_xgb = xgb_lt["lead_time_windows"].values.astype(float)
ieee_lgb  = lgb_lt["lead_time_windows"].values.astype(float)

ccfraud_lt = lt[lt["dataset"].str.contains("ULB")]
ccfraud_xgb = ccfraud_lt[ccfraud_lt["model"] == "AdaptiveXGBoost"]["lead_time_windows"].values.astype(float)
ccfraud_lgb = ccfraud_lt[ccfraud_lt["model"] == "AdaptiveLightGBM"]["lead_time_windows"].values.astype(float)

def pct_stats(arr):
    if len(arr) == 0:
        return {}
    return {
        "n": len(arr),
        "min": int(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": round(float(arr.mean()), 2),
        "p75": float(np.percentile(arr, 75)),
        "max": int(arr.max()),
        "std": round(float(arr.std()), 2),
    }

pct_rows = []
for model, dataset, arr in [
    ("AdaptiveXGBoost",  "IEEE-CIS", ieee_xgb),
    ("AdaptiveLightGBM", "IEEE-CIS", ieee_lgb),
    ("AdaptiveXGBoost",  "ULB CC Fraud", ccfraud_xgb),
    ("AdaptiveLightGBM", "ULB CC Fraud", ccfraud_lgb),
]:
    row = {"model": model, "dataset": dataset}
    row.update(pct_stats(arr))
    pct_rows.append(row)

pct_df = pd.DataFrame(pct_rows)
pct_df.to_csv(TABLES / "lead_time_percentiles.csv", index=False)
print("\nLead-time percentiles:")
print(pct_df.to_string(index=False))
