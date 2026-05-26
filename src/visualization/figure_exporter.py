"""Utility to export all paper figures in batch."""

from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGURES_DIR = Path(__file__).resolve().parents[3] / "results" / "figures"


def save_figure(fig: plt.Figure, name: str, output_dir: Optional[Path] = None) -> None:
    """Save a matplotlib figure as both PNG (300dpi) and PDF."""
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / name}.png / .pdf")


def list_generated_figures() -> list[Path]:
    """List all PNG figures in the results/figures/ directory."""
    return sorted(FIGURES_DIR.glob("**/*.png"))
