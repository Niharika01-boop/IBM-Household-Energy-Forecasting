import pandas as pd

# Dataset path
DATA_PATH = "data/household_power_consumption.txt"

# Load a small sample first
df = pd.read_csv(
    DATA_PATH,
    sep=";",
    nrows=10000
)

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

print("\n========== COLUMN NAMES ==========\n")
print(df.columns.tolist())

print("\n========== DATA SHAPE ==========\n")
print(df.shape)

print("\n========== DATA TYPES ==========\n")
print(df.dtypes)

print("\n========== MISSING VALUES ==========\n")
print(df.isna().sum())

print("\n========== BASIC STATISTICS ==========\n")
print(df.describe())