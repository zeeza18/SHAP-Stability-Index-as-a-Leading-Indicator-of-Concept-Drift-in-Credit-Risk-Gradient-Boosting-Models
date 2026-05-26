"""Compute CatBoost SSI lead-time (lambda) values and Wilcoxon H0: lambda=0 test.

Uses tau=0.80 (paper's stated threshold) and same methodology as
compute_ssi_lead_times in analyze_results.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"

TAU = 0.80
AUC_DROP_DELTA = 0.02


def compute_lambda(df: pd.DataFrame, tau: float = TAU) -> list[dict]:
    rows = []
    drift_windows = df.loc[df["drift_detected"] == True, "window_index"].tolist()
    for dw in drift_windows:
        pre = df[df["window_index"] < dw]
        if pre.empty:
            continue
        # k_SSI: first window (in pre) where SSI < tau; default = dw
        ssi_pre = pre.dropna(subset=["ssi"])
        ssi_low = ssi_pre[ssi_pre["ssi"] < tau]["window_index"]
        ssi_drop_at = int(ssi_low.min()) if not ssi_low.empty else dw

        # k_AUC: first post window where AUC drops > AUC_DROP_DELTA below recent baseline
        baseline_auc = pre["roc_auc"].tail(3).mean()
        post = df[df["window_index"] >= dw]
        auc_low = post[post["roc_auc"] < baseline_auc - AUC_DROP_DELTA]["window_index"]
        auc_drop_at = int(auc_low.min()) if not auc_low.empty else dw

        rows.append({
            "drift_window": dw,
            "ssi_drop_window": ssi_drop_at,
            "auc_drop_window": auc_drop_at,
            "lambda": auc_drop_at - ssi_drop_at,
        })
    return rows


def wilcoxon_test_lambda(lambdas: list[int]) -> dict:
    arr = np.array(lambdas)
    nonzero = arr[arr != 0]
    n = len(nonzero)
    if n < 4:
        return {"n_nonzero": n, "W": None, "p": None, "r": None, "note": "insufficient n"}
    stat, p = wilcoxon(nonzero, alternative="greater")
    # effect size r = Z / sqrt(n)
    from scipy.stats import norm
    # wilcoxon returns W statistic; compute z-score
    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (stat - mu) / sigma
    r = abs(z) / np.sqrt(n)
    return {"n_nonzero": n, "W": stat, "p": p, "z": z, "r": r}


def main():
    print(f"\n{'='*60}")
    print("  CatBoost Lambda + Wilcoxon Analysis  (tau={TAU})")
    print(f"{'='*60}\n")

    datasets = {
        "IEEE-CIS": "adaptive_AdaptiveCatBoost_ieee_cis.csv",
        "ULB CC":   "adaptive_AdaptiveCatBoost_ccfraud.csv",
        "GMSC":     "adaptive_AdaptiveCatBoost_gmsc.csv",
    }

    all_lambdas = {}
    for label, fname in datasets.items():
        path = TABLES / fname
        if not path.exists():
            print(f"  [MISS] {fname}")
            continue
        df = pd.read_csv(path)
        rows = compute_lambda(df)
        lambdas = [r["lambda"] for r in rows]
        all_lambdas[label] = lambdas

        print(f"--- {label} ---")
        print(f"  Drift events (computed): {len(rows)}")
        print(f"  Lambda values: {lambdas}")
        if lambdas:
            arr = np.array(lambdas)
            print(f"  Mean: {arr.mean():.2f}")
            print(f"  Std:  {arr.std(ddof=1):.2f}")
            print(f"  Range: {arr.min()}–{arr.max()}")
            n_zero = int((arr == 0).sum())
            print(f"  lambda=0 events: {n_zero}")
        print()

    print("\n--- Wilcoxon H0: lambda=0 (one-sided, H1: lambda>0) ---")
    for label, lambdas in all_lambdas.items():
        if not lambdas:
            continue
        res = wilcoxon_test_lambda(lambdas)
        n_zero = sum(1 for l in lambdas if l == 0)
        print(f"\n  {label}:")
        print(f"    n_total={len(lambdas)}, lambda=0 count={n_zero}, n_nonzero={res['n_nonzero']}")
        if res["p"] is not None:
            sig = "***" if res["p"] < 0.001 else ("**" if res["p"] < 0.01 else ("*" if res["p"] < 0.05 else "ns"))
            print(f"    W={res['W']:.0f}, p={res['p']:.4f} {sig}, r={res['r']:.2f}")
        else:
            print(f"    {res['note']}")

    # Also compare all three models on IEEE-CIS for the paper's λ=0 section
    print("\n\n--- Cross-model comparison on IEEE-CIS ---")
    for model_file, label in [
        ("adaptive_AdaptiveXGBoost_ieee_cis.csv", "XGBoost"),
        ("adaptive_AdaptiveLightGBM_ieee_cis.csv", "LightGBM"),
        ("adaptive_AdaptiveCatBoost_ieee_cis.csv", "CatBoost"),
    ]:
        path = TABLES / model_file
        if not path.exists():
            print(f"  [MISS] {model_file}")
            continue
        df = pd.read_csv(path)
        rows = compute_lambda(df)
        lambdas = [r["lambda"] for r in rows]
        res = wilcoxon_test_lambda(lambdas)
        arr = np.array(lambdas)
        n_zero = int((arr == 0).sum())
        sig = "***" if res["p"] < 0.001 else "ns"
        print(f"  {label}: mean={arr.mean():.2f}, std={arr.std(ddof=1):.2f}, "
              f"n={len(lambdas)}, lam=0 count={n_zero}, "
              f"W={res['W']:.0f}, p={res['p']:.4f} {sig}, r={res['r']:.2f}")


if __name__ == "__main__":
    main()
