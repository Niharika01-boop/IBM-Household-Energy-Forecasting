import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models import SimpleRNN


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
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

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

import joblib

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

model = SimpleRNN(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    output_size=H
).to(device)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "simple_rnn_T24_H1_best.pt"
)

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.eval()

print("Trained model loaded successfully!")


# ============================================================
# CONVERT TEST DATA TO TENSOR
# ============================================================

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(device)


# ============================================================
# MAKE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

with torch.no_grad():

    predictions_scaled = model(
        X_test_tensor
    ).cpu().numpy()


# ============================================================
# INVERSE TRANSFORM
# ============================================================

y_test_original = target_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()

predictions_original = target_scaler.inverse_transform(
    predictions_scaled.reshape(-1, 1)
).flatten()


# ============================================================
# METRICS
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        y_test_original,
        predictions_original
    )
)

mae = mean_absolute_error(
    y_test_original,
    predictions_original
)


# MAPE with epsilon to avoid division by zero
epsilon = 1e-8

mape = np.mean(
    np.abs(
        (y_test_original - predictions_original)
        /
        np.maximum(
            np.abs(y_test_original),
            epsilon
        )
    )
) * 100


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========== TEST RESULTS ==========")

print(f"RMSE : {rmse:.4f} kW")
print(f"MAE  : {mae:.4f} kW")
print(f"MAPE : {mape:.2f}%")


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

print("\nCreating Actual vs Predicted graph...")

plt.figure(figsize=(14, 6))

plt.plot(
    y_test_original,
    label="Actual",
    linewidth=1
)

plt.plot(
    predictions_original,
    label="Predicted",
    linewidth=1
)

plt.xlabel("Test Time Step")
plt.ylabel("Global Active Power (kW)")
plt.title(
    "Simple RNN: Actual vs Predicted Power"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "simple_rnn_actual_vs_predicted.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# RESIDUALS
# ============================================================

residuals = (
    y_test_original -
    predictions_original
)


print("Creating residual distribution...")


plt.figure(figsize=(10, 6))

plt.hist(
    residuals,
    bins=50
)

plt.xlabel("Residual (Actual - Predicted)")

plt.ylabel("Frequency")

plt.title(
    "Simple RNN Residual Distribution"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "simple_rnn_residual_distribution.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# SAVE PREDICTIONS
# ============================================================

results = np.column_stack(
    (
        y_test_original,
        predictions_original,
        residuals
    )
)

np.savetxt(
    os.path.join(
        OUTPUT_DIR,
        "simple_rnn_predictions.csv"
    ),
    results,
    delimiter=",",
    header="Actual_kW,Predicted_kW,Residual_kW",
    comments=""
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n========================================")
print("EVALUATION COMPLETED SUCCESSFULLY!")
print("========================================")

print("\nSaved files:")

print(
    "simple_rnn_actual_vs_predicted.png"
)

print(
    "simple_rnn_residual_distribution.png"
)

print(
    "simple_rnn_predictions.csv"
)