"""Model comparison table generator with LaTeX output.

Generates the main results table (Table 1 in the paper):
    Rows: Static XGB, Static LGBM, Adaptive XGB, Adaptive LGBM
    Columns: Mean AUC, Std AUC, Mean F1, Mean KS, Mean SSI, Post-drift recovery
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)

TABLES_DIR = Path(__file__).resolve().parents[3] / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


class ModelComparator:
    """Aggregates per-window metrics and generates comparison tables."""

    METRIC_COLS = ["roc_auc", "f1_at_05", "ks_statistic", "average_precision"]
    DISPLAY_NAMES = {
        "roc_auc": "AUC-ROC",
        "f1_at_05": "F1 (0.5)",
        "ks_statistic": "KS",
        "average_precision": "Avg Precision",
    }

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self._results: dict[str, pd.DataFrame] = {}  # model_name -> metrics DataFrame
        self._ssi_results: dict[str, pd.DataFrame] = {}

    def add_model_results(
        self, model_name: str, metrics_df: pd.DataFrame
    ) -> None:
        """Register per-window metrics for a model."""
        self._results[model_name] = metrics_df

    def add_ssi_results(self, model_name: str, ssi_df: pd.DataFrame) -> None:
        """Register SSI results for a model."""
        self._ssi_results[model_name] = ssi_df

    def summary_table(self) -> pd.DataFrame:
        """Return aggregated summary across all registered models."""
        rows = []
        for model_name, df in self._results.items():
            row = {"model": model_name}
            for col in self.METRIC_COLS:
                if col in df.columns:
                    row[f"mean_{col}"] = df[col].mean()
                    row[f"std_{col}"] = df[col].std()
            # Add mean SSI if available
            if model_name in self._ssi_results:
                ssi_df = self._ssi_results[model_name]
                row["mean_ssi"] = ssi_df["ssi_value"].dropna().mean()
            rows.append(row)
        return pd.DataFrame(rows)

    def to_latex(self, output_path: Optional[Path] = None) -> str:
        """Generate LaTeX table with best values bolded per column."""
        df = self.summary_table()
        if df.empty:
            return ""

        metric_cols = [c for c in df.columns if c.startswith("mean_") and not c.startswith("std_")]
        latex_rows = []

        # Find best (max) per metric column
        best = {col: df[col].max() for col in metric_cols}

        for _, row in df.iterrows():
            cells = [row["model"]]
            for col in metric_cols:
                val = row[col]
                std_col = col.replace("mean_", "std_")
                std_val = row.get(std_col, np.nan)
                cell = f"{val:.4f}"
                if not np.isnan(std_val):
                    cell += f" ± {std_val:.4f}"
                if abs(val - best[col]) < 1e-6:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
            latex_rows.append(" & ".join(cells) + " \\\\")

        header_names = ["Model"] + [
            self.DISPLAY_NAMES.get(c.replace("mean_", ""), c) for c in metric_cols
        ]
        header = " & ".join(header_names) + " \\\\"
        latex = (
            "\\begin{table}[ht]\n"
            "\\centering\n"
            f"\\caption{{Results on {self.dataset_name}}}\n"
            "\\begin{tabular}{l" + "r" * len(metric_cols) + "}\n"
            "\\hline\n"
            + header + "\n"
            "\\hline\n"
            + "\n".join(latex_rows) + "\n"
            "\\hline\n"
            "\\end{tabular}\n"
            "\\end{table}"
        )

        if output_path:
            output_path.write_text(latex)
        return latex

    def save_csv(self) -> Path:
        path = TABLES_DIR / f"comparison_{self.dataset_name}.csv"
        self.summary_table().to_csv(path, index=False)
        print(f"Comparison table saved to {path}")
        return path
