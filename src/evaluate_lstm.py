import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

from models import VanillaLSTM


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/sequences"
SCALER_DIR = "data/scaled"
MODEL_DIR = "artifacts/models"
OUTPUT_DIR = "artifacts/evaluation"

T = 24
H = 1
HIDDEN_SIZE = 64


# ============================================================
# SETUP
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

X_test = np.load(
    os.path.join(
        DATA_DIR,
        f"X_test_T{T}_H{H}.npy"
    )
)

y_test = np.load(
    os.path.join(
        DATA_DIR,
        f"y_test_T{T}_H{H}.npy"
    )
)


# ============================================================
# LOAD TARGET SCALER
# ============================================================

target_scaler = joblib.load(
    os.path.join(
        SCALER_DIR,
        "target_scaler.joblib"
    )
)


# ============================================================
# CREATE MODEL
# ============================================================

input_size = X_test.shape[2]

model = VanillaLSTM(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    output_size=H
).to(device)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "vanilla_lstm_T24_H1_best.pt"
)

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.eval()

print("Vanilla LSTM loaded successfully!")


# ============================================================
# PREDICTION
# ============================================================

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(device)

print("\nGenerating predictions...")

with torch.no_grad():

    predictions_scaled = model(
        X_test_tensor
    ).cpu().numpy()


# ============================================================
# INVERSE TRANSFORM
# ============================================================

y_actual = target_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

y_predicted = target_scaler.inverse_transform(
    predictions_scaled.reshape(-1, 1)
).flatten()


# ============================================================
# METRICS
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        y_actual,
        y_predicted
    )
)

mae = mean_absolute_error(
    y_actual,
    y_predicted
)

epsilon = 1e-8

mape = np.mean(
    np.abs(
        (y_actual - y_predicted)
        /
        np.maximum(
            np.abs(y_actual),
            epsilon
        )
    )
) * 100


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========== VANILLA LSTM RESULTS ==========")

print(f"RMSE : {rmse:.4f} kW")
print(f"MAE  : {mae:.4f} kW")
print(f"MAPE : {mape:.2f}%")


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    y_actual,
    label="Actual",
    linewidth=1
)

plt.plot(
    y_predicted,
    label="Predicted",
    linewidth=1
)

plt.xlabel("Test Time Step")
plt.ylabel("Global Active Power (kW)")

plt.title(
    "Vanilla LSTM: Actual vs Predicted Power"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "vanilla_lstm_actual_vs_predicted.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# RESIDUAL DISTRIBUTION
# ============================================================

residuals = y_actual - y_predicted

plt.figure(figsize=(10, 6))

plt.hist(
    residuals,
    bins=50
)

plt.xlabel(
    "Residual (Actual - Predicted)"
)

plt.ylabel("Frequency")

plt.title(
    "Vanilla LSTM Residual Distribution"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "vanilla_lstm_residual_distribution.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# SAVE PREDICTIONS
# ============================================================

results = np.column_stack(
    (
        y_actual,
        y_predicted,
        residuals
    )
)

np.savetxt(
    os.path.join(
        OUTPUT_DIR,
        "vanilla_lstm_predictions.csv"
    ),
    results,
    delimiter=",",
    header="Actual_kW,Predicted_kW,Residual_kW",
    comments=""
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("VANILLA LSTM EVALUATION COMPLETED!")
print("========================================")

print("\nSaved files:")
print("vanilla_lstm_actual_vs_predicted.png")
print("vanilla_lstm_residual_distribution.png")
print("vanilla_lstm_predictions.csv")