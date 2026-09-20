import pandas as pd

# ============================================================
# STEP 4: HOURLY RESAMPLING
# ============================================================

DATA_PATH = "data/household_power_consumption.txt"
OUTPUT_PATH = "data/household_power_hourly.csv"

# ------------------------------------------------------------
# 1. Load raw dataset
# ------------------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    sep=";",
    na_values=["?"]
)

print("Raw shape:", df.shape)

# ------------------------------------------------------------
# 2. Create datetime column
# ------------------------------------------------------------

df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True,
    errors="coerce"
)

# Remove original date/time columns
df.drop(columns=["Date", "Time"], inplace=True)

# ------------------------------------------------------------
# 3. Convert measurements to numeric
# ------------------------------------------------------------

measurement_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

for column in measurement_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# ------------------------------------------------------------
# 4. Set datetime as index
# ------------------------------------------------------------

df.set_index("datetime", inplace=True)

# Sort chronologically
df.sort_index(inplace=True)

# ------------------------------------------------------------
# 5. Handle missing values
# ------------------------------------------------------------

print("\nMissing values before interpolation:")
print(df.isna().sum())

df = df.interpolate(method="time")
df = df.ffill().bfill()

print("\nMissing values after interpolation:")
print(df.isna().sum())

# ------------------------------------------------------------
# 6. Resample to hourly frequency
# ------------------------------------------------------------

print("\nResampling minute data to hourly data...")

hourly_df = df.resample("h").mean()

# ------------------------------------------------------------
# 7. Remove any remaining missing rows
# ------------------------------------------------------------

hourly_df.dropna(inplace=True)

# ------------------------------------------------------------
# 8. Save hourly dataset
# ------------------------------------------------------------

hourly_df.to_csv(OUTPUT_PATH)

# ------------------------------------------------------------
# 9. Display results
# ------------------------------------------------------------

print("\n========== RESULTS ==========")

print("Original shape:")
print(df.shape)

print("\nHourly shape:")
print(hourly_df.shape)

print("\nHourly data:")
print(hourly_df.head())

print("\nHourly dataset saved to:")
print(OUTPUT_PATH)