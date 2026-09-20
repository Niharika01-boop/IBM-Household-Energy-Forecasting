import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
import joblib

# ============================================================
# STEP 7
# TRAIN / VALIDATION / TEST SPLIT
# + LEAK-FREE MINMAX SCALING
# ============================================================

INPUT_PATH = "data/household_power_features.csv"

OUTPUT_DIR = "data/scaled"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. Load feature-engineered dataset
# ------------------------------------------------------------

print("Loading feature-engineered dataset...")

df = pd.read_csv(
    INPUT_PATH,
    parse_dates=["datetime"],
    index_col="datetime"
)

print("Complete dataset shape:", df.shape)

# ------------------------------------------------------------
# 2. Define target and input features
# ------------------------------------------------------------

TARGET = "Global_active_power"

FEATURES = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos"
]

# Check that all required columns exist
missing_columns = [
    col for col in FEATURES
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

# ------------------------------------------------------------
# 3. Chronological split
# ------------------------------------------------------------

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print("\n========== DATA SPLIT ==========")

print("Total rows:", n)

print(
    f"Train:      {len(train_df)} rows "
    f"({len(train_df) / n * 100:.1f}%)"
)

print(
    f"Validation: {len(val_df)} rows "
    f"({len(val_df) / n * 100:.1f}%)"
)

print(
    f"Test:       {len(test_df)} rows "
    f"({len(test_df) / n * 100:.1f}%)"
)

print("\nTrain period:")
print(train_df.index.min(), "→", train_df.index.max())

print("\nValidation period:")
print(val_df.index.min(), "→", val_df.index.max())

print("\nTest period:")
print(test_df.index.min(), "→", test_df.index.max())

# ------------------------------------------------------------
# 4. Separate X and y
# ------------------------------------------------------------

X_train = train_df[FEATURES].values
X_val = val_df[FEATURES].values
X_test = test_df[FEATURES].values

y_train = train_df[[TARGET]].values
y_val = val_df[[TARGET]].values
y_test = test_df[[TARGET]].values

# ------------------------------------------------------------
# 5. Create separate scalers
# ------------------------------------------------------------

feature_scaler = MinMaxScaler()

target_scaler = MinMaxScaler()

# ------------------------------------------------------------
# 6. FIT SCALERS ONLY ON TRAINING DATA
# ------------------------------------------------------------

print("\nFitting scalers ONLY on training data...")

feature_scaler.fit(X_train)

target_scaler.fit(y_train)

# ------------------------------------------------------------
# 7. Transform train / validation / test
# ------------------------------------------------------------

X_train_scaled = feature_scaler.transform(X_train)

X_val_scaled = feature_scaler.transform(X_val)

X_test_scaled = feature_scaler.transform(X_test)

y_train_scaled = target_scaler.transform(y_train)

y_val_scaled = target_scaler.transform(y_val)

y_test_scaled = target_scaler.transform(y_test)

# ------------------------------------------------------------
# 8. Verify scaling
# ------------------------------------------------------------

print("\n========== SCALING CHECK ==========")

print(
    "Scaled training feature minimum:",
    X_train_scaled.min()
)

print(
    "Scaled training feature maximum:",
    X_train_scaled.max()
)

print(
    "Scaled training target minimum:",
    y_train_scaled.min()
)

print(
    "Scaled training target maximum:",
    y_train_scaled.max()
)

# ------------------------------------------------------------
# 9. Save scaled arrays
# ------------------------------------------------------------

np.save(
    f"{OUTPUT_DIR}/X_train.npy",
    X_train_scaled
)

np.save(
    f"{OUTPUT_DIR}/X_val.npy",
    X_val_scaled
)

np.save(
    f"{OUTPUT_DIR}/X_test.npy",
    X_test_scaled
)

np.save(
    f"{OUTPUT_DIR}/y_train.npy",
    y_train_scaled
)

np.save(
    f"{OUTPUT_DIR}/y_val.npy",
    y_val_scaled
)

np.save(
    f"{OUTPUT_DIR}/y_test.npy",
    y_test_scaled
)

# ------------------------------------------------------------
# 10. Save scalers
# ------------------------------------------------------------

joblib.dump(
    feature_scaler,
    f"{OUTPUT_DIR}/feature_scaler.joblib"
)

joblib.dump(
    target_scaler,
    f"{OUTPUT_DIR}/target_scaler.joblib"
)

# ------------------------------------------------------------
# 11. Save feature names
# ------------------------------------------------------------

with open(
    f"{OUTPUT_DIR}/feature_names.txt",
    "w"
) as f:

    for feature in FEATURES:
        f.write(feature + "\n")

# ------------------------------------------------------------
# 12. Final message
# ------------------------------------------------------------

print("\n========================================")
print("SPLIT + SCALING COMPLETED SUCCESSFULLY!")
print("========================================")

print("\nSaved files:")
print("X_train.npy")
print("X_val.npy")
print("X_test.npy")
print("y_train.npy")
print("y_val.npy")
print("y_test.npy")
print("feature_scaler.joblib")
print("target_scaler.joblib")
print("feature_names.txt")