"""PSI vs SSI drift alert comparison for IEEE-CIS (Item 6).

Computes PSI on 20 representative features per evaluation window,
then reports k_PSI vs k_SSI vs k_AUC.

Reference date for TransactionDT: 2017-11-30 (seconds from epoch).
Windowing: 14-day windows, 7-day stride, 6-window minimum training period.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"

TRANSACTION_CSV = ROOT / "data" / "raw" / "train_transaction.csv"

# 20 representative features (continuous and low-cardinality numeric)
# These approximate what SHAP top-20 returns for IEEE-CIS gradient boosters
FEATURES = [
    "TransactionAmt",
    "card1", "card2", "card3", "card5",
    "addr1", "addr2", "dist1",
    "C1", "C2", "C5", "C6", "C9", "C11", "C13",
    "D1", "D2", "D3", "D4", "D10",
]

WINDOW_DAYS = 14
STRIDE_DAYS = 7
MIN_TRAIN_WINDOWS = 6
SECONDS_PER_DAY = 86_400
PSI_THRESHOLD = 0.25
N_BINS = 10
EPSILON = 1e-6

# Known k_SSI and k_AUC from paper results (all models agree at tau=0.80)
K_SSI = 8   # first alert window for all models (XGBoost, LightGBM, CatBoost)
# k_AUC is derived from mean lambda: k_SSI + mean_lambda ~ 8 + 7.83 = 15.83 -> w16
# Using AdaptiveXGBoost reference: first AUC drop window from lead-time data
K_AUC_XGB = 8 + 7    # approx window 15 (k_SSI + mean_lambda rounded)


def compute_psi_for_feature(ref: pd.Series, test: pd.Series, n_bins: int) -> float:
    """PSI between reference and test distributions using quantile bins."""
    ref_clean = ref.dropna()
    test_clean = test.dropna()
    if len(ref_clean) == 0 or len(test_clean) == 0:
        return 0.0

    # Build quantile-based bins on reference
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(ref_clean, quantiles))
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    test_counts, _ = np.histogram(test_clean, bins=bin_edges)

    ref_pct = ref_counts / len(ref_clean)
    test_pct = test_counts / len(test_clean)

    # Clip to avoid log(0)
    ref_pct = np.clip(ref_pct, EPSILON, None)
    test_pct = np.clip(test_pct, EPSILON, None)

    psi = np.sum((test_pct - ref_pct) * np.log(test_pct / ref_pct))
    return float(psi)


def main():
    print("Loading IEEE-CIS transaction data...")
    usecols = ["TransactionDT"] + FEATURES
    df = pd.read_csv(TRANSACTION_CSV, usecols=usecols)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    dt = df["TransactionDT"]
    dt_min = dt.min()
    dt_max = dt.max()
    total_days = (dt_max - dt_min) / SECONDS_PER_DAY
    print(f"TransactionDT span: {total_days:.1f} days "
          f"({dt_min:.0f} to {dt_max:.0f})")

    # Build window boundaries (seconds)
    window_s = WINDOW_DAYS * SECONDS_PER_DAY
    stride_s = STRIDE_DAYS * SECONDS_PER_DAY

    window_starts = []
    start = dt_min
    while start + window_s <= dt_max:
        window_starts.append(start)
        start += stride_s

    print(f"Total windows: {len(window_starts)}")

    # Slice windows into DataFrames
    windows = []
    for i, ws in enumerate(window_starts):
        mask = (dt >= ws) & (dt < ws + window_s)
        windows.append({"idx": i, "df": df[mask][FEATURES]})

    print(f"Window sizes: {[len(w['df']) for w in windows[:6]]} ...")

    # Reference distribution = union of first MIN_TRAIN_WINDOWS windows
    ref_df = pd.concat([windows[i]["df"] for i in range(MIN_TRAIN_WINDOWS)],
                       ignore_index=True)
    print(f"Reference period: {len(ref_df):,} rows across windows 0-{MIN_TRAIN_WINDOWS-1}")

    # Compute PSI per evaluation window (windows MIN_TRAIN_WINDOWS onward)
    results = []
    for w in windows[MIN_TRAIN_WINDOWS:]:
        win_idx = w["idx"]
        test_df = w["df"]

        feature_psi = {}
        for feat in FEATURES:
            psi = compute_psi_for_feature(ref_df[feat], test_df[feat], N_BINS)
            feature_psi[feat] = psi

        mean_psi = float(np.mean(list(feature_psi.values())))
        max_psi = float(np.max(list(feature_psi.values())))
        n_flagged = sum(1 for v in feature_psi.values() if v > PSI_THRESHOLD)

        results.append({
            "window_index": win_idx,
            "mean_psi": round(mean_psi, 4),
            "max_psi": round(max_psi, 4),
            "n_features_flagged": n_flagged,
            "alert": int(mean_psi > PSI_THRESHOLD),
            **{f"psi_{k}": round(v, 4) for k, v in feature_psi.items()},
        })

    df_results = pd.DataFrame(results)

    # Find k_PSI
    alerted = df_results[df_results["alert"] == 1]
    k_psi = int(alerted["window_index"].min()) if len(alerted) > 0 else None

    print("\n=== PSI per evaluation window ===")
    print(df_results[["window_index", "mean_psi", "max_psi",
                       "n_features_flagged", "alert"]].to_string(index=False))

    print(f"\n=== DRIFT ALERT COMPARISON (tau_PSI={PSI_THRESHOLD}, tau_SSI=0.80) ===")
    print(f"  k_PSI  (mean PSI > {PSI_THRESHOLD}): window {k_psi}")
    print(f"  k_SSI  (all 3 models, tau=0.80) : window {K_SSI}")
    print(f"  k_AUC  (approx, XGB reference)  : window {K_SSI + 8} "
          f"(mean lambda ~7.83 windows)")

    if k_psi is not None and k_psi > K_SSI:
        print(f"\n  SSI leads PSI by {k_psi - K_SSI} windows")
    elif k_psi is not None and k_psi == K_SSI:
        print(f"\n  SSI and PSI alert at the same window ({K_SSI})")
    elif k_psi is not None:
        print(f"\n  PSI leads SSI by {K_SSI - k_psi} windows")
    else:
        print("\n  PSI never crosses threshold — no mean-PSI alert triggered")

    # Save
    out_path = TABLES / "psi_comparison_ieee_cis.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # Compact summary for paper table
    print("\n=== PAPER SUMMARY TABLE ===")
    print(f"{'Metric':15} {'k_alert':>10} {'Basis':40}")
    print("-" * 65)
    print(f"{'PSI (mean)':15} {str(k_psi) if k_psi else 'none':>10} "
          f"{'mean PSI>0.25 across 20 features':40}")
    print(f"{'SSI (all models)':15} {K_SSI:>10} "
          f"{'SSI<0.80 at tau=0.80, L=5 (XGB/LGB/CB)':40}")
    print(f"{'AUC (XGB ref)':15} {'~16':>10} "
          f"{'k_SSI + mean lambda 7.83 windows':40}")

    return df_results, k_psi


if __name__ == "__main__":
    main()
