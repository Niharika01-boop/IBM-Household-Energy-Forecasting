from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_FILE = Path("artifacts/results/results_table.csv")
OUTPUT_DIR = Path("artifacts/results/plots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_FILE)

# -----------------------------
# 1. RMSE by Model and Horizon
# -----------------------------

for horizon in sorted(df["Horizon"].unique()):

    subset = df[df["Horizon"] == horizon]

    plt.figure(figsize=(10, 6))

    for model in sorted(subset["Model"].unique()):

        model_data = subset[
            subset["Model"] == model
        ].sort_values("Sequence_Length")

        plt.plot(
            model_data["Sequence_Length"],
            model_data["RMSE_kW"],
            marker="o",
            label=model.upper()
        )

    plt.xlabel("Sequence Length (T)")
    plt.ylabel("RMSE (kW)")
    plt.title(f"RMSE vs Sequence Length — Horizon {horizon}")
    plt.xticks([24, 72, 168])
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output = OUTPUT_DIR / f"rmse_horizon_{horizon}.png"

    plt.savefig(output, dpi=200)
    plt.close()

# -----------------------------
# 2. MAE by Model and Horizon
# -----------------------------

for horizon in sorted(df["Horizon"].unique()):

    subset = df[df["Horizon"] == horizon]

    plt.figure(figsize=(10, 6))

    for model in sorted(subset["Model"].unique()):

        model_data = subset[
            subset["Model"] == model
        ].sort_values("Sequence_Length")

        plt.plot(
            model_data["Sequence_Length"],
            model_data["MAE_kW"],
            marker="o",
            label=model.upper()
        )

    plt.xlabel("Sequence Length (T)")
    plt.ylabel("MAE (kW)")
    plt.title(f"MAE vs Sequence Length — Horizon {horizon}")
    plt.xticks([24, 72, 168])
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output = OUTPUT_DIR / f"mae_horizon_{horizon}.png"

    plt.savefig(output, dpi=200)
    plt.close()

# -----------------------------
# 3. Overall RMSE comparison
# -----------------------------

plt.figure(figsize=(12, 7))

labels = (
    df["Model"].str.upper()
    + " T"
    + df["Sequence_Length"].astype(str)
    + " H"
    + df["Horizon"].astype(str)
)

plt.bar(labels, df["RMSE_kW"])

plt.xlabel("Experiment")
plt.ylabel("RMSE (kW)")
plt.title("RMSE Comparison Across All Experiments")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "overall_rmse_comparison.png",
    dpi=200
)

plt.close()

print("=" * 60)
print("PLOTS CREATED SUCCESSFULLY")
print("=" * 60)

print(f"Saved in: {OUTPUT_DIR}")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(file)