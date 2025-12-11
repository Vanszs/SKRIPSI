
import pandas as pd
import numpy as np
import os

# Load the data (assuming processed data)
data_path = 'd:/SKRIPSI/gas-ml/data/features.parquet'
if not os.path.exists(data_path):
    print(f"File not found: {data_path}")
    exit()

df = pd.read_parquet(data_path)

# Columns of interest - adjust names if they differ in your parquet
cols = ['baseFeePerGas', 'gasUsed', 'baseFee_next']

# Check if columns exist
missing = [c for c in cols if c not in df.columns]
if missing:
    print(f"Columns not found: {missing}")
    print(f"Available: {df.columns.tolist()}")
    exit()

# Create a copy for display
df_display = df[cols].copy()

# Convert Wei to Gwei
df_display['baseFeePerGas (Gwei)'] = df_display['baseFeePerGas'] / 1e9
df_display['baseFee_next (Gwei)'] = df_display['baseFee_next'] / 1e9
df_display['gasUsed (M)'] = df_display['gasUsed'] / 1e6

# Drop original columns
df_display = df_display.drop(columns=['baseFeePerGas', 'gasUsed', 'baseFee_next'])

# Calculate stats
stats = df_display.describe().T[['mean', 'std', 'min', '50%', 'max']]

# Format for nice output
pd.options.display.float_format = '{:,.2f}'.format
print("\n=== Prettier Descriptive Statistics (SINTA 3 Ready) ===\n")
print(stats)
print("\nNote: Descriptive Statistics should be raw data (unnormalized) to show real-world scale, but units should be readable.")
