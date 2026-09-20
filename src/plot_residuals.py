from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = Path("artifacts/results")
OUTPUT_DIR = Path("artifacts/results/plots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "lstm"
T = 168
H = 48

file = RESULTS_DIR / f"{MODEL}_T{T}_H{H}_predictions.npz"

data = np.load(file)

actual = data["actual"].reshape(-1)
predicted = data["predicted"].reshape(-1)

residuals = actual - predicted

# Residual histogram
plt.figure(figsize=(10, 6))

plt.hist(residuals, bins=50)

plt.xlabel("Residual (Actual - Predicted) [kW]")
plt.ylabel("Frequency")
plt.title("Residual Distribution — LSTM (T=168, H=48)")
plt.grid(True, alpha=0.3)

plt.tight_layout()

output = OUTPUT_DIR / "residual_distribution_lstm_T168_H48.png"

plt.savefig(output, dpi=200)
plt.close()

print("=" * 60)
print("RESIDUAL PLOT CREATED")
print("=" * 60)
print(f"Mean residual: {np.mean(residuals):.6f} kW")
print(f"Std residual : {np.std(residuals):.6f} kW")
print(f"Saved: {output}")