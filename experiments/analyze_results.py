"""Full analysis pipeline for the paper.

Generates all tables and figures from experiment CSVs. Safe to run while
experiments are still in progress — missing files are skipped with a warning.

Outputs
-------
results/tables/
    comparison_summary.csv       Table 1 raw numbers
    comparison_<dataset>.tex     Table 1 LaTeX per dataset
    statistical_tests.csv        Table 2 Wilcoxon p-values + effect sizes
    ablation_summary.csv         Table 3 ablation 2x2

results/figures/
    fig1_auc_over_time.pdf/png   AUC trajectories static vs adaptive
    fig2_drift_timeline.pdf/png  Drift events, AUC, error rate per window
    fig3_ssi_leading.pdf/png     SSI as early warning signal vs AUC
    fig4_ablation.pdf/png        2x2 ablation bar chart

Usage
-----
    python experiments/analyze_results.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["ieee_cis", "gmsc", "ccfraud"]
DATASET_LABELS = {"ieee_cis": "IEEE-CIS Fraud", "gmsc": "Give Me Some Credit", "ccfraud": "ULB CC Fraud"}

# ── Matplotlib setup ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
})

COLORS = {
    "StaticXGBoost":    "#2196F3",   # blue
    "StaticLightGBM":   "#03A9F4",   # light blue
    "AdaptiveXGBoost":  "#F44336",   # red
    "AdaptiveLightGBM": "#FF9800",   # orange
}
LINESTYLES = {
    "StaticXGBoost":    "--",
    "StaticLightGBM":   "-.",
    "AdaptiveXGBoost":  "-",
    "AdaptiveLightGBM": "-",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names across baseline and adaptive CSVs."""
    # baseline uses f1_at_05; adaptive uses f1 — unify to f1
    if "f1_at_05" in df.columns and "f1" not in df.columns:
        df = df.rename(columns={"f1_at_05": "f1"})
    return df


def load_results() -> dict[str, dict[str, pd.DataFrame]]:
    """Load all available result CSVs.

    Returns nested dict:  results[model_name][dataset] = DataFrame
    """
    patterns = {
        "StaticXGBoost":    "baseline_StaticXGBoost_{ds}.csv",
        "StaticLightGBM":   "baseline_StaticLightGBM_{ds}.csv",
        "AdaptiveXGBoost":  "adaptive_AdaptiveXGBoost_{ds}.csv",
        "AdaptiveLightGBM": "adaptive_AdaptiveLightGBM_{ds}.csv",
    }

    results: dict[str, dict[str, pd.DataFrame]] = {m: {} for m in patterns}
    for model, pat in patterns.items():
        for ds in DATASETS:
            path = TABLES_DIR / pat.format(ds=ds)
            if path.exists():
                results[model][ds] = _normalise(pd.read_csv(path))
            else:
                print(f"  [WARN] Missing: {path.name}")
    return results


def load_ablation() -> dict[str, pd.DataFrame]:
    """Load ablation CSVs per dataset."""
    out = {}
    for ds in DATASETS:
        path = TABLES_DIR / f"ablation_{ds}.csv"
        if path.exists():
            out[ds] = pd.read_csv(path)
        else:
            print(f"  [WARN] Missing ablation: {path.name}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TABLE 1 — MAIN RESULTS COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def build_summary_table(results: dict) -> pd.DataFrame:
    """Aggregate per-window metrics into mean ± std per model per dataset."""
    rows = []
    metrics = ["roc_auc", "average_precision", "f1"]

    for model, ds_dict in results.items():
        for ds, df in ds_dict.items():
            row: dict = {"model": model, "dataset": ds}
            for m in metrics:
                if m in df.columns:
                    row[f"mean_{m}"] = df[m].mean()
                    row[f"std_{m}"] = df[m].std()
            if "drift_detected" in df.columns:
                row["n_drift_events"] = df["drift_detected"].sum()
                row["n_retrains"] = df.get("model_retrained", pd.Series(dtype=bool)).sum()
            if "ssi" in df.columns:
                row["mean_ssi"] = df["ssi"].dropna().mean()
            rows.append(row)

    return pd.DataFrame(rows)


def summary_to_latex(summary: pd.DataFrame, dataset: str) -> str:
    """Generate a single-dataset LaTeX table with best values bolded."""
    df = summary[summary["dataset"] == dataset].copy()
    if df.empty:
        return ""

    metric_cols = ["mean_roc_auc", "mean_average_precision", "mean_f1"]
    col_labels = ["AUC-ROC", "Avg Prec", "F1"]
    available = [c for c in metric_cols if c in df.columns]
    labels = [col_labels[metric_cols.index(c)] for c in available]

    best = {c: df[c].max() for c in available}

    rows_tex = []
    for _, row in df.iterrows():
        cells = [row["model"].replace("_", "\\_")]
        for col in available:
            val = row[col]
            std_col = col.replace("mean_", "std_")
            std = row.get(std_col, np.nan)
            cell = f"{val:.4f}" + (f" $\\pm$ {std:.4f}" if not np.isnan(std) else "")
            if abs(val - best[col]) < 1e-6:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        rows_tex.append(" & ".join(cells) + r" \\")

    label_str = DATASET_LABELS.get(dataset, dataset)
    n_cols = 1 + len(available)
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        f"\\caption{{Performance on {label_str}}}\n"
        f"\\label{{tab:results_{dataset}}}\n"
        f"\\begin{{tabular}}{{l{'r' * len(available)}}}\n"
        "\\hline\n"
        "Model & " + " & ".join(labels) + r" \\" + "\n"
        "\\hline\n"
        + "\n".join(rows_tex) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TABLE 2 — STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_statistical_tests(results: dict, metric: str = "roc_auc") -> pd.DataFrame:
    """Wilcoxon signed-rank: each adaptive model vs each static model per dataset."""
    from src.evaluation.statistical_tests import wilcoxon_test

    rows = []
    adaptive_models = [m for m in results if m.startswith("Adaptive")]
    static_models   = [m for m in results if m.startswith("Static")]

    for ds in DATASETS:
        for adap in adaptive_models:
            for stat in static_models:
                if ds not in results[adap] or ds not in results[stat]:
                    continue
                df_a = results[adap][ds][["window_index", metric]].dropna()
                df_s = results[stat][ds][["window_index", metric]].dropna()
                merged = df_a.merge(df_s, on="window_index", suffixes=("_a", "_s"))
                if len(merged) < 5:
                    continue
                test = wilcoxon_test(
                    merged[f"{metric}_a"].values,
                    merged[f"{metric}_s"].values,
                    alternative="greater",
                )
                rows.append({
                    "dataset": ds,
                    "adaptive": adap,
                    "static": stat,
                    "metric": metric,
                    "mean_adaptive": merged[f"{metric}_a"].mean(),
                    "mean_static": merged[f"{metric}_s"].mean(),
                    "delta": merged[f"{metric}_a"].mean() - merged[f"{metric}_s"].mean(),
                    **test,
                })

    return pd.DataFrame(rows)


def tests_to_latex(tests_df: pd.DataFrame) -> str:
    """Format statistical tests as a compact LaTeX table."""
    if tests_df.empty:
        return ""
    rows_tex = []
    for _, r in tests_df.iterrows():
        sig = "**" if r["significant_at_001"] else ("*" if r["significant_at_005"] else "")
        delta_str = f"+{r['delta']:.4f}" if r["delta"] >= 0 else f"{r['delta']:.4f}"
        rows_tex.append(
            f"{DATASET_LABELS.get(r['dataset'], r['dataset'])} & "
            f"{r['adaptive']} vs {r['static']} & "
            f"{r['mean_adaptive']:.4f} & {r['mean_static']:.4f} & "
            f"{delta_str} & {r['p_value']:.4f}{sig} & {r['effect_size_r']:.3f}"
            r" \\"
        )
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Statistical significance of adaptive vs static (Wilcoxon signed-rank, AUC-ROC, one-sided $H_1$: adaptive $>$ static). * $p{<}0.05$, ** $p{<}0.01$.}\n"
        "\\label{tab:stat_tests}\n"
        "\\begin{tabular}{llrrrrr}\n"
        "\\hline\n"
        r"Dataset & Comparison & $\bar{\mu}_{adap}$ & $\bar{\mu}_{static}$ & $\Delta$ & $p$-value & $r$ \\" + "\n"
        "\\hline\n"
        + "\n".join(rows_tex) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TABLE 3 — ABLATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_ablation_table(ablation: dict) -> pd.DataFrame:
    """Summarise ablation conditions per dataset."""
    rows = []
    for ds, df in ablation.items():
        for cond, grp in df.groupby("condition"):
            rows.append({
                "dataset": DATASET_LABELS.get(ds, ds),
                "condition": cond,
                "mean_auc": grp["roc_auc"].mean(),
                "std_auc": grp["roc_auc"].std(),
                "mean_f1": grp["f1"].mean() if "f1" in grp.columns else np.nan,
                "n_retrains": grp.get("model_retrained", pd.Series(dtype=bool)).sum(),
            })
    return pd.DataFrame(rows)


def ablation_to_latex(abl_df: pd.DataFrame) -> str:
    """Format ablation as a LaTeX table."""
    if abl_df.empty:
        return ""
    rows_tex = []
    for ds, grp in abl_df.groupby("dataset"):
        best_auc = grp["mean_auc"].max()
        for _, r in grp.iterrows():
            auc_str = f"{r['mean_auc']:.4f} $\\pm$ {r['std_auc']:.4f}"
            if abs(r["mean_auc"] - best_auc) < 1e-6:
                auc_str = f"\\textbf{{{auc_str}}}"
            rows_tex.append(
                f"{r['dataset']} & {r['condition']} & {auc_str} & {r['mean_f1']:.4f}"
                r" \\"
            )
    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Ablation study: contribution of each component (mean AUC-ROC $\\pm$ std).}\n"
        "\\label{tab:ablation}\n"
        "\\begin{tabular}{llrr}\n"
        "\\hline\n"
        r"Dataset & Condition & AUC-ROC & F1 \\" + "\n"
        "\\hline\n"
        + "\n".join(rows_tex) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FIGURE 1 — AUC OVER TIME
# ═══════════════════════════════════════════════════════════════════════════════

def fig_auc_over_time(results: dict) -> None:
    """One column per dataset, static (dashed) vs adaptive (solid) AUC trajectories."""
    available_ds = [ds for ds in DATASETS if any(ds in v for v in results.values())]
    if not available_ds:
        print("  [SKIP] fig1: no results yet")
        return

    n_cols = len(available_ds)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 3.5), sharey=False)
    if n_cols == 1:
        axes = [axes]

    for ax, ds in zip(axes, available_ds):
        for model, ds_dict in results.items():
            if ds not in ds_dict:
                continue
            df = ds_dict[ds].sort_values("window_index")
            ax.plot(
                df["window_index"], df["roc_auc"],
                label=model,
                color=COLORS.get(model, "grey"),
                linestyle=LINESTYLES.get(model, "-"),
                marker="o", markersize=3,
            )
            # Shade drift/retrain events for adaptive models
            if "drift_detected" in df.columns:
                retrain_wins = df.loc[df["model_retrained"] == True, "window_index"]
                for w in retrain_wins:
                    ax.axvline(w, color=COLORS[model], alpha=0.12, linewidth=6)

        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlabel("Window Index")
        ax.set_ylabel("AUC-ROC")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.02))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("AUC-ROC Over Time: Static vs Adaptive Models", fontsize=12, y=1.01)
    fig.tight_layout()
    _save_fig(fig, "fig1_auc_over_time")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FIGURE 2 — DRIFT DETECTION TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════

def fig_drift_timeline(results: dict) -> None:
    """Per-dataset: AUC + error rate with drift/retrain events marked."""
    adaptive_models = [m for m in results if m.startswith("Adaptive")]
    available_ds = [
        ds for ds in DATASETS
        if any(ds in results[m] for m in adaptive_models)
    ]
    if not available_ds:
        print("  [SKIP] fig2: no adaptive results yet")
        return

    n_cols = len(available_ds)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 5), sharex=False)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, ds in enumerate(available_ds):
        ax_auc = axes[0, col]
        ax_err = axes[1, col]

        for model in adaptive_models:
            if ds not in results[model]:
                continue
            df = results[model][ds].sort_values("window_index")
            color = COLORS.get(model, "grey")

            ax_auc.plot(df["window_index"], df["roc_auc"],
                        label=model, color=color, marker="o", markersize=3)
            ax_err.plot(df["window_index"], df["error_rate"],
                        label=model, color=color, linestyle="--", marker="s", markersize=3)

            # Mark retrain windows
            if "model_retrained" in df.columns:
                retrain = df[df["model_retrained"] == True]
                ax_auc.scatter(retrain["window_index"], retrain["roc_auc"],
                               marker="v", color=color, s=60, zorder=5, label=f"{model} retrain")

        ax_auc.set_title(DATASET_LABELS.get(ds, ds))
        ax_auc.set_ylabel("AUC-ROC")
        ax_err.set_ylabel("Error Rate")
        ax_err.set_xlabel("Window Index")
        for ax in (ax_auc, ax_err):
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Deduplicate
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    fig.legend(*zip(*unique), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Drift Detection Timeline: AUC and Error Rate per Window", fontsize=12, y=1.01)
    fig.tight_layout()
    _save_fig(fig, "fig2_drift_timeline")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. FIGURE 3 — SSI AS LEADING INDICATOR
# ═══════════════════════════════════════════════════════════════════════════════

def fig_ssi_leading(results: dict) -> None:
    """SSI (right axis) vs AUC (left axis) to show SSI leads AUC drops."""
    adaptive_models = [m for m in results if m.startswith("Adaptive")]
    available = [
        (ds, model)
        for ds in DATASETS
        for model in adaptive_models
        if ds in results[model] and "ssi" in results[model][ds].columns
        and results[model][ds]["ssi"].notna().sum() > 2
    ]
    if not available:
        print("  [SKIP] fig3: no SSI data yet")
        return

    n = len(available)
    n_cols = min(n, 3)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
    axes_flat = np.array(axes).flatten()

    for idx, (ds, model) in enumerate(available):
        ax = axes_flat[idx]
        df = results[model][ds].sort_values("window_index")
        ssi_vals = df["ssi"].dropna()
        if ssi_vals.empty:
            continue

        color = COLORS.get(model, "grey")
        ax2 = ax.twinx()

        ax.plot(df["window_index"], df["roc_auc"], color=color,
                label="AUC-ROC", marker="o", markersize=3)
        ax2.plot(df["window_index"], df["ssi"], color="green", linestyle=":",
                 label="SSI", marker="D", markersize=3, alpha=0.8)
        ax2.axhline(0.7, color="green", linestyle="--", alpha=0.4, linewidth=1,
                    label="SSI threshold (0.7)")

        ax.set_title(f"{DATASET_LABELS.get(ds, ds)}\n{model}", fontsize=9)
        ax.set_xlabel("Window Index")
        ax.set_ylabel("AUC-ROC", color=color)
        ax2.set_ylabel("SSI", color="green")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc="lower left")

    for idx in range(len(available), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("SHAP Stability Index as Leading Indicator of Model Degradation",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    _save_fig(fig, "fig3_ssi_leading")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. FIGURE 4 — ABLATION BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════

def fig_ablation(ablation: dict) -> None:
    """Grouped bar chart: 4 conditions × 3 datasets."""
    if not ablation:
        print("  [SKIP] fig4: no ablation results yet")
        return

    abl_df = build_ablation_table(ablation)
    conditions = abl_df["condition"].unique().tolist()
    datasets   = abl_df["dataset"].unique().tolist()

    x = np.arange(len(conditions))
    width = 0.25
    offsets = np.linspace(-(len(datasets) - 1) * width / 2,
                          (len(datasets) - 1) * width / 2, len(datasets))

    palette = ["#4CAF50", "#2196F3", "#FF9800"]
    fig, ax = plt.subplots(figsize=(9, 4))

    for i, (ds, offset) in enumerate(zip(datasets, offsets)):
        sub = abl_df[abl_df["dataset"] == ds].set_index("condition")
        vals = [sub.loc[c, "mean_auc"] if c in sub.index else 0.0 for c in conditions]
        errs = [sub.loc[c, "std_auc"]  if c in sub.index else 0.0 for c in conditions]
        bars = ax.bar(x + offset, vals, width, label=ds, color=palette[i], alpha=0.85,
                      yerr=errs, capsize=3, error_kw={"linewidth": 1})

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.set_ylabel("Mean AUC-ROC")
    ax.set_title("Ablation Study: Component Contribution to Performance")
    ax.legend(title="Dataset")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    fig.tight_layout()
    _save_fig(fig, "fig4_ablation")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SSI LEAD-TIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ssi_lead_times(results: dict) -> pd.DataFrame:
    """How many windows does SSI drop before AUC drops at each drift event?"""
    from src.explainability.shap_stability_index import ShapStabilityIndex

    rows = []
    for model, ds_dict in results.items():
        if not model.startswith("Adaptive"):
            continue
        for ds, df in ds_dict.items():
            if "ssi" not in df.columns or "drift_detected" not in df.columns:
                continue
            drift_windows = df.loc[df["drift_detected"] == True, "window_index"].tolist()
            if not drift_windows:
                continue
            ssi_threshold = 0.7
            for dw in drift_windows:
                pre = df[df["window_index"] < dw]
                if pre.empty:
                    continue
                ssi_drop = pre[pre["ssi"] < ssi_threshold]["window_index"]
                ssi_drop_at = int(ssi_drop.min()) if not ssi_drop.empty else dw

                post = df[df["window_index"] >= dw]
                baseline_auc = pre["roc_auc"].tail(3).mean()
                auc_drop = post[post["roc_auc"] < baseline_auc - 0.02]["window_index"]
                auc_drop_at = int(auc_drop.min()) if not auc_drop.empty else dw

                rows.append({
                    "model": model,
                    "dataset": DATASET_LABELS.get(ds, ds),
                    "drift_window": dw,
                    "ssi_drop_window": ssi_drop_at,
                    "auc_drop_window": auc_drop_at,
                    "lead_time_windows": auc_drop_at - ssi_drop_at,
                })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _save_fig(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        path = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    print(f"  Saved {name}.pdf / .png")
    plt.close(fig)


def _print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("  ANALYSIS PIPELINE")
    print("=" * 60)

    _print_section("Loading results")
    results = load_results()
    ablation = load_ablation()

    n_loaded = sum(len(v) for v in results.values())
    print(f"  Loaded {n_loaded} model×dataset result files")
    print(f"  Ablation datasets: {list(ablation.keys())}")

    # ── Table 1: Summary ──────────────────────────────────────────────────────
    _print_section("Table 1 — Main results summary")
    summary = build_summary_table(results)
    summary.to_csv(TABLES_DIR / "comparison_summary.csv", index=False)
    print(summary.to_string(index=False))

    for ds in DATASETS:
        tex = summary_to_latex(summary, ds)
        if tex:
            path = TABLES_DIR / f"comparison_{ds}.tex"
            path.write_text(tex)
            print(f"  LaTeX → {path.name}")

    # ── Table 2: Statistical tests ────────────────────────────────────────────
    _print_section("Table 2 — Statistical significance tests")
    has_both = any(
        ds in results.get("StaticXGBoost", {}) and ds in results.get("AdaptiveXGBoost", {})
        for ds in DATASETS
    )
    if has_both:
        tests_df = run_statistical_tests(results)
        tests_df.to_csv(TABLES_DIR / "statistical_tests.csv", index=False)
        if not tests_df.empty:
            display_cols = ["dataset", "adaptive", "static", "mean_adaptive",
                            "mean_static", "delta", "p_value", "effect_size_r", "significant_at_005"]
            print(tests_df[display_cols].to_string(index=False))
            tex = tests_to_latex(tests_df)
            (TABLES_DIR / "statistical_tests.tex").write_text(tex)
            print("  LaTeX → statistical_tests.tex")
        else:
            print("  Not enough paired windows for Wilcoxon tests.")
    else:
        print("  [SKIP] Need both static and adaptive results for tests.")

    # ── Table 3: Ablation ─────────────────────────────────────────────────────
    _print_section("Table 3 — Ablation study")
    if ablation:
        abl_df = build_ablation_table(ablation)
        abl_df.to_csv(TABLES_DIR / "ablation_summary.csv", index=False)
        print(abl_df.to_string(index=False))
        tex = ablation_to_latex(abl_df)
        (TABLES_DIR / "ablation.tex").write_text(tex)
        print("  LaTeX → ablation.tex")
    else:
        print("  [SKIP] No ablation results yet.")

    # ── SSI lead-time ─────────────────────────────────────────────────────────
    _print_section("SSI Lead-Time Analysis")
    lead_df = compute_ssi_lead_times(results)
    if not lead_df.empty:
        lead_df.to_csv(TABLES_DIR / "ssi_lead_time.csv", index=False)
        print(lead_df.to_string(index=False))
        mean_lead = lead_df["lead_time_windows"].mean()
        print(f"\n  Mean SSI lead time: {mean_lead:.1f} windows")
    else:
        print("  [SKIP] Insufficient SSI + drift data.")

    # ── Figures ───────────────────────────────────────────────────────────────
    _print_section("Generating figures")
    fig_auc_over_time(results)
    fig_drift_timeline(results)
    fig_ssi_leading(results)
    fig_ablation(ablation)

    _print_section("Done")
    print(f"  Tables  → {TABLES_DIR}")
    print(f"  Figures → {FIGURES_DIR}")
    print()


if __name__ == "__main__":
    main()
