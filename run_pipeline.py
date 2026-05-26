"""Master pipeline runner for the credit risk drift research project.

Runs all steps in order, skipping anything already done. Every step is
checkpointed so you can kill it at any time and resume without losing work.

Usage
-----
  # Run everything (skips completed steps automatically)
  python run_pipeline.py

  # Run one step only
  python run_pipeline.py --step preprocess
  python run_pipeline.py --step tune
  python run_pipeline.py --step baseline
  python run_pipeline.py --step adaptive
  python run_pipeline.py --step ablation

  # One dataset only (any step)
  python run_pipeline.py --dataset ieee_cis
  python run_pipeline.py --step adaptive --dataset gmsc

  # Ignore ALL checkpoints and redo everything from scratch
  python run_pipeline.py --fresh

  # Force CPU (disables GPU for this run)
  python run_pipeline.py --no-gpu

Steps execute in order: preprocess → tune → baseline → adaptive → ablation
Each step uses the cached output of the previous one.
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATASETS = ["ieee_cis", "gmsc", "ccfraud"]
TUNING_RESULTS = ROOT / "experiments" / "tuning" / "tuning_results"

ALL_STEPS = ["preprocess", "tune", "baseline", "adaptive", "ablation"]


# ─────────────────────────────────────────────────────────────────────────────
# Step implementations
# ─────────────────────────────────────────────────────────────────────────────

def step_preprocess(datasets: list[str], fresh: bool) -> None:
    """Build and cache temporal windows for each dataset."""
    from src.data.window_cache import load_or_build
    for ds in datasets:
        print(f"\n[Preprocess] {ds}")
        load_or_build(ds, force=fresh)


def step_tune(datasets: list[str], n_trials: int, fresh_cache: bool) -> None:
    """Optuna tuning (resumes from SQLite if interrupted).

    Tuning is run once on the primary dataset (ieee_cis by default).
    The resulting params are reused across all datasets.
    """
    # Use first dataset in list as primary tuning dataset
    tuning_dataset = datasets[0] if datasets else "ieee_cis"
    from experiments.run_tuning import run_all_tuning
    run_all_tuning(dataset=tuning_dataset, n_trials=n_trials, fresh_cache=fresh_cache)


def step_baseline(datasets: list[str], fresh: bool) -> None:
    from experiments.run_baseline import run_baseline
    tuned_params_dir = TUNING_RESULTS if TUNING_RESULTS.exists() else None
    for ds in datasets:
        run_baseline(ds, tuned_params_dir=tuned_params_dir, fresh=fresh)


def step_adaptive(datasets: list[str], fresh: bool) -> None:
    from experiments.run_adaptive import run_adaptive
    tuned_params_dir = TUNING_RESULTS if TUNING_RESULTS.exists() else None
    for ds in datasets:
        run_adaptive(ds, tuned_params_dir=tuned_params_dir, fresh=fresh)


def step_ablation(datasets: list[str], fresh: bool) -> None:
    from experiments.run_ablation import run_ablation
    for ds in datasets:
        run_ablation(ds, fresh=fresh)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Credit Risk Drift Pipeline — GPU-accelerated, fully checkpointed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step",
        choices=ALL_STEPS + ["all"],
        default="all",
        help="Which step to run (default: all)",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS + ["all"],
        default="all",
        help="Which dataset(s) to process (default: all)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=150,
        help="Optuna trials for tuning step (default: 150)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore all checkpoints and re-run from scratch",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (force CPU)",
    )
    args = parser.parse_args()

    # GPU override
    if args.no_gpu:
        os.environ["FORCE_CPU"] = "1"
        print("[Pipeline] GPU disabled via --no-gpu (FORCE_CPU=1)")

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    steps = ALL_STEPS if args.step == "all" else [args.step]

    if args.fresh:
        from src.utils.checkpoint import clear_all_checkpoints
        clear_all_checkpoints()

    # Print GPU status once at the top
    from src.utils.gpu import print_gpu_status
    print_gpu_status()

    t_start = time.time()
    print(f"\nPipeline: {steps}")
    print(f"Datasets: {datasets}")
    print(f"Fresh:    {args.fresh}")
    print("=" * 60)

    sep = "-" * 60
    for step in steps:
        t_step = time.time()
        print(f"\n{sep}")
        print(f"  STEP: {step.upper()}")
        print(sep)

        if step == "preprocess":
            step_preprocess(datasets, fresh=args.fresh)
        elif step == "tune":
            step_tune(datasets, n_trials=args.n_trials, fresh_cache=args.fresh)
        elif step == "baseline":
            step_baseline(datasets, fresh=args.fresh)
        elif step == "adaptive":
            step_adaptive(datasets, fresh=args.fresh)
        elif step == "ablation":
            step_ablation(datasets, fresh=args.fresh)

        elapsed = time.time() - t_step
        print(f"\n  [{step}] Done in {elapsed/60:.1f} min")

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Pipeline complete. Total time: {total/60:.1f} min")
    print(f"Results -> {ROOT / 'results'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
