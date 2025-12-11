# Benchmark Reproduction Guide

This document outlines the steps to reproduce the "SINTA 3 Benchmark Comparison" (Notebook 04), comparing the proposed XGBoost model against standard baselines (Linear Regression, Random Forest, ARIMA, ETS, SVR).

## Prerequisites
- Python environment with requirements installed (`pip install -r requirements.txt`)
- Feature engineering must be completed (`data/features.parquet` must exist).

## 1. Train Baselines
Run the benchmark training script. This trains all 5 baseline models and saves their metrics/predictions.

```bash
python src/train_benchmarks.py
```
*Output location:* `models/benchmarks/`

## 2. Train Proposed Model (XGBoost)
Train the standalone XGBoost model using the standard training script.

```bash
python src/train_xgb.py --cfg cfg/exp.yaml --in data/features.parquet
```
*Output location:* `models/xgboost_robust/` (or similar, check `models/` for the latest timestamped output if standard path varies)

## 3. Run Comparison Notebook
Execute the comparison notebook to visualize the results.

```bash
jupyter nbconvert --to notebook --execute notebooks/04_Model_Comparison.ipynb --output notebooks/04_Model_Comparison_Run.ipynb
```
Or open `notebooks/04_Model_Comparison.ipynb` in Jupyter/VSCode and Run All Cells.

## Troubleshooting
- **Missing Data**: Ensure `python -m src.features --in data/blocks.csv --out data/features.parquet` has been run.
- **Path Errors**: The scripts assume they are run from the project root (`d:\SKRIPSI\gas-ml`).
