import pandas as pd
import numpy as np
import os

# ============================================================
# STEP 6: FEATURE ENGINEERING
# ============================================================

INPUT_PATH = "data/household_power_hourly.csv"
OUTPUT_PATH = "data/household_power_features.csv"

# ------------------------------------------------------------
# 1. Load hourly dataset
# ------------------------------------------------------------

print("Loading hourly dataset...")

df = pd.read_csv(
    INPUT_PATH,
    parse_dates=["datetime"],
    index_col="datetime"
)

print("Original shape:", df.shape)

# ------------------------------------------------------------
# 2. Create time-related variables
# ------------------------------------------------------------

# Hour of day: 0 to 23
df["hour"] = df.index.hour

# Day of week: Monday=0, Sunday=6
df["day_of_week"] = df.index.dayofweek

# Month: 1 to 12
df["month"] = df.index.month

# ------------------------------------------------------------
# 3. Cyclic encoding of HOUR
# ------------------------------------------------------------

df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

# ------------------------------------------------------------
# 4. Cyclic encoding of DAY OF WEEK
# ------------------------------------------------------------

df["dow_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["dow_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)

# ------------------------------------------------------------
# 5. Cyclic encoding of MONTH
# ------------------------------------------------------------

df["month_sin"] = np.sin(
    2 * np.pi * (df["month"] - 1) / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * (df["month"] - 1) / 12
)

# ------------------------------------------------------------
# 6. Remove intermediate integer time columns
# ------------------------------------------------------------

df.drop(
    columns=["hour", "day_of_week", "month"],
    inplace=True
)

# ------------------------------------------------------------
# 7. Check missing values
# ------------------------------------------------------------

print("\n========== MISSING VALUES ==========")
print(df.isna().sum())

# ------------------------------------------------------------
# 8. Check final columns
# ------------------------------------------------------------

print("\n========== FINAL COLUMNS ==========")

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

# ------------------------------------------------------------
# 9. Display sample
# ------------------------------------------------------------

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

# ------------------------------------------------------------
# 10. Save engineered dataset
# ------------------------------------------------------------

df.to_csv(OUTPUT_PATH)

print("\n========== RESULTS ==========")
print("Final shape:", df.shape)

print("\nFeature-engineered dataset saved to:")
print(OUTPUT_PATH)

print("\nFeature engineering completed successfully!")