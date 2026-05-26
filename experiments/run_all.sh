#!/bin/bash
# Master runner: executes all experiment steps in sequence.
# Run from the project root: bash experiments/run_all.sh

set -e

TUNED_DIR="experiments/tuning/tuning_results"

echo "============================================"
echo "Step 1: Download data (if not already done)"
echo "============================================"
if [ ! -f "data/raw/train_transaction.csv" ]; then
    bash data/raw/download_all.sh
else
    echo "Data already present. Skipping download."
fi

echo ""
echo "============================================"
echo "Step 2: Validate data"
echo "============================================"
python src/data/loader.py --validate

echo ""
echo "============================================"
echo "Step 3: Hyperparameter tuning (Optuna)"
echo "============================================"
python experiments/run_tuning.py --dataset ieee_cis --n-trials 150

echo ""
echo "============================================"
echo "Step 4: Run baseline models (all datasets)"
echo "============================================"
python experiments/run_baseline.py --dataset all --tuned-params $TUNED_DIR

echo ""
echo "============================================"
echo "Step 5: Run adaptive models (all datasets)"
echo "============================================"
python experiments/run_adaptive.py --dataset all --tuned-params $TUNED_DIR

echo ""
echo "============================================"
echo "Step 6: Run ablation study (all datasets)"
echo "============================================"
python experiments/run_ablation.py --dataset all

echo ""
echo "============================================"
echo "Step 7: Generate all figures"
echo "============================================"
python notebooks/10_paper_figures.py 2>/dev/null || echo "Run notebook 10 manually."

echo ""
echo "============================================"
echo "ALL DONE. Check results/tables/ and results/figures/"
echo "============================================"
