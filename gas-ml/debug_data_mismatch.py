import pandas as pd
import numpy as np
from pathlib import Path

# Paths
xgb_path = Path("models/xgboost_notebook/test_predictions.parquet")
bench_path = Path("models/benchmarks/gbr/y_test.npy")

print("--- XGBoost Data ---")
if xgb_path.exists():
    df_xgb = pd.read_parquet(xgb_path)
    y_true_xgb = df_xgb['y_true'].values
    print(f"Shape: {y_true_xgb.shape}")
    print(f"First 5: {y_true_xgb[:5]}")
    print(f"Mean: {y_true_xgb.mean()}")
    print(f"Max: {y_true_xgb.max()}")
else:
    print("XGBoost predictions not found.")

print("\n--- Benchmark Data ---")
if bench_path.exists():
    y_true_bench = np.load(bench_path)
    print(f"Shape: {y_true_bench.shape}")
    print(f"First 5: {y_true_bench[:5]}")
    print(f"Mean: {y_true_bench.mean()}")
    print(f"Max: {y_true_bench.max()}")

print("\n--- Comparison ---")
if xgb_path.exists() and bench_path.exists():
    if y_true_xgb.shape == y_true_bench.shape:
        mse = np.mean((y_true_xgb - y_true_bench)**2)
        if mse == 0:
            print("Status: IDENTICAL data.")
        else:
            print(f"Status: DIFFERENT data (MSE: {mse})")
    else:
        print("Status: DIFFERENT SHAPES (Different splits used)")
