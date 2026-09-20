import pandas as pd

# ============================================================
# STEP 3: LOAD AND CLEAN HOUSEHOLD ENERGY DATA
# ============================================================

DATA_PATH = "data/household_power_consumption.txt"

# ------------------------------------------------------------
# 1. Load the raw dataset
# ------------------------------------------------------------
df = pd.read_csv(
    DATA_PATH,
    sep=";",
    na_values=["?"]
)

print("Raw dataset loaded successfully!")
print("Shape:", df.shape)

# ------------------------------------------------------------
# 2. Combine Date and Time into a single datetime column
# ------------------------------------------------------------
df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True,
    errors="coerce"
)

# ------------------------------------------------------------
# 3. Remove the original Date and Time columns
# ------------------------------------------------------------
df.drop(columns=["Date", "Time"], inplace=True)

# ------------------------------------------------------------
# 4. Set datetime as the index
# ------------------------------------------------------------
df.set_index("datetime", inplace=True)

# ------------------------------------------------------------
# 5. Convert all measurement columns to numeric
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
    df[column] = pd.to_numeric(df[column], errors="coerce")

# ------------------------------------------------------------
# 6. Check missing values
# ------------------------------------------------------------
print("\nMissing values BEFORE interpolation:")
print(df.isna().sum())

# ------------------------------------------------------------
# 7. Interpolate missing values
# ------------------------------------------------------------
df = df.interpolate(method="time")

# Handle any remaining missing values at beginning/end
df = df.ffill().bfill()

# ------------------------------------------------------------
# 8. Check missing values again
# ------------------------------------------------------------
print("\nMissing values AFTER cleaning:")
print(df.isna().sum())

# ------------------------------------------------------------
# 9. Display cleaned data
# ------------------------------------------------------------
print("\nFirst 5 cleaned rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nCleaned dataset shape:")
print(df.shape)