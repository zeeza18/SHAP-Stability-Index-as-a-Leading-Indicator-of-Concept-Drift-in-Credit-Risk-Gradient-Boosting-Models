# Experiments

## Execution Order

```
Step 1 (optional):  python experiments/run_tuning.py --dataset ieee_cis
Step 2:             python experiments/run_baseline.py --dataset all
Step 3:             python experiments/run_adaptive.py --dataset all
Step 4:             python experiments/run_ablation.py --dataset all
```

Or run everything at once:
```
bash experiments/run_all.sh
```

## Configs
All tunable parameters are in `configs/`. Never hardcode numbers in scripts.

## Tuning Results
Best Optuna parameters are saved to `tuning/tuning_results/*.json`.
Optuna study objects (for visualization) are saved as `.pkl`.
