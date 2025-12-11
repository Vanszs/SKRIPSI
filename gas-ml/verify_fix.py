import numpy as np
from pathlib import Path

path = Path("models/benchmarks/gbr/y_test.npy")
if path.exists():
    data = np.load(path)
    print(f"Benchmark y_test shape: {data.shape}")
    if data.shape[0] == 1998:
        print("SUCCESS: Matches XGBoost test size (1998).")
    else:
        print(f"FAILURE: Mismatch (Expected 1998, got {data.shape[0]})")
else:
    print("File not found.")
