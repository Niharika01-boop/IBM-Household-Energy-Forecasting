import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# STEP 5: EXPLORATORY DATA ANALYSIS
# ============================================================

DATA_PATH = "data/household_power_hourly.csv"
OUTPUT_DIR = "artifacts"

# Create artifacts folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. Load hourly dataset
# ------------------------------------------------------------

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["datetime"],
    index_col="datetime"
)

print("Dataset loaded successfully!")

# ------------------------------------------------------------
# 2. Basic information
# ------------------------------------------------------------

print("\n========== DATASET INFO ==========")
df.info()

print("\n========== SHAPE ==========")
print(df.shape)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== LAST 5 ROWS ==========")
print(df.tail())

# ------------------------------------------------------------
# 3. Missing values
# ------------------------------------------------------------

print("\n========== MISSING VALUES ==========")
print(df.isna().sum())

# ------------------------------------------------------------
# 4. Statistical summary
# ------------------------------------------------------------

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# ------------------------------------------------------------
# 5. Zero values in target
# ------------------------------------------------------------

target = "Global_active_power"

zero_count = (df[target] == 0).sum()

print("\n========== ZERO VALUES ==========")
print(f"Zero {target} values: {zero_count}")

# ============================================================
# GRAPH 1 — COMPLETE TIME SERIES
# ============================================================

print("\nCreating Graph 1...")

plt.figure(figsize=(15, 5))

plt.plot(df.index, df[target])

plt.title("Hourly Global Active Power")
plt.xlabel("Date")
plt.ylabel("Global Active Power (kW)")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "hourly_energy_consumption.png"),
    dpi=150
)

plt.close()

print("Graph 1 saved.")

# ============================================================
# GRAPH 2 — FIRST 7 DAYS
# ============================================================

print("Creating Graph 2...")

first_week = df.iloc[:24 * 7]

plt.figure(figsize=(15, 5))

plt.plot(first_week.index, first_week[target])

plt.title("Hourly Global Active Power - First Week")
plt.xlabel("Date")
plt.ylabel("Global Active Power (kW)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "first_week_energy.png"),
    dpi=150
)

plt.close()

print("Graph 2 saved.")

# ============================================================
# GRAPH 3 — DAILY AVERAGE
# ============================================================

print("Creating Graph 3...")

daily = df[target].resample("D").mean()

plt.figure(figsize=(15, 5))

plt.plot(daily.index, daily)

plt.title("Daily Average Global Active Power")
plt.xlabel("Date")
plt.ylabel("Average Global Active Power (kW)")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "daily_average_energy.png"),
    dpi=150
)

plt.close()

print("Graph 3 saved.")

# ============================================================
# GRAPH 4 — HOUR OF DAY
# ============================================================

print("Creating Graph 4...")

hourly_pattern = df.groupby(
    df.index.hour
)[target].mean()

plt.figure(figsize=(10, 5))

plt.plot(
    hourly_pattern.index,
    hourly_pattern.values,
    marker="o"
)

plt.title("Average Energy Consumption by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Global Active Power (kW)")

plt.xticks(range(24))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "hour_of_day_pattern.png"),
    dpi=150
)

plt.close()

print("Graph 4 saved.")

# ============================================================
# GRAPH 5 — DAY OF WEEK
# ============================================================

print("Creating Graph 5...")

weekday_pattern = df.groupby(
    df.index.dayofweek
)[target].mean()

days = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

plt.figure(figsize=(10, 5))

plt.plot(
    days,
    weekday_pattern.values,
    marker="o"
)

plt.title("Average Energy Consumption by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Global Active Power (kW)")

plt.xticks(rotation=30)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "day_of_week_pattern.png"),
    dpi=150
)

plt.close()

print("Graph 5 saved.")

# ============================================================
# FINISHED
# ============================================================

print("\n========================================")
print("EDA COMPLETED SUCCESSFULLY!")
print("========================================")

print("\nGraphs saved in:")
print("artifacts/")

print("\nFiles created:")

print("1. hourly_energy_consumption.png")
print("2. first_week_energy.png")
print("3. daily_average_energy.png")
print("4. hour_of_day_pattern.png")
print("5. day_of_week_pattern.png")