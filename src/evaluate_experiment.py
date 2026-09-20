import os
import argparse
import numpy as np
import torch
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error

from models import SimpleRNN, VanillaLSTM, StackedLSTM


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    choices=["rnn", "lstm", "stacked"],
    required=True
)

parser.add_argument(
    "--T",
    type=int,
    required=True
)

parser.add_argument(
    "--H",
    type=int,
    required=True
)

args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data/sequences"
SCALER_DIR = "data/scaled"
MODEL_DIR = "artifacts/models"
RESULT_DIR = "artifacts/results"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

X_test = np.load(
    os.path.join(
        DATA_DIR,
        f"X_test_T{args.T}_H{args.H}.npy"
    )
)

y_test = np.load(
    os.path.join(
        DATA_DIR,
        f"y_test_T{args.T}_H{args.H}.npy"
    )
)


# ============================================================
# SCALER
# ============================================================

target_scaler = joblib.load(
    os.path.join(
        SCALER_DIR,
        "target_scaler.joblib"
    )
)


# ============================================================
# MODEL
# ============================================================

input_size = X_test.shape[2]
hidden_size = 64

if args.model == "rnn":

    model = SimpleRNN(
        input_size,
        hidden_size,
        args.H
    )

elif args.model == "lstm":

    model = VanillaLSTM(
        input_size,
        hidden_size,
        args.H
    )

else:

    model = StackedLSTM(
        input_size,
        hidden_size,
        args.H,
        num_layers=2
    )


model = model.to(device)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    f"{args.model}_T{args.T}_H{args.H}_best.pt"
)

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.eval()


# ============================================================
# PREDICT
# ============================================================

X_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(device)


with torch.no_grad():

    predictions_scaled = model(
        X_tensor
    ).cpu().numpy()


# ============================================================
# INVERSE TRANSFORM
# ============================================================

actual = target_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).reshape(y_test.shape)

predicted = target_scaler.inverse_transform(
    predictions_scaled.reshape(-1, 1)
).reshape(predictions_scaled.shape)


# ============================================================
# METRICS
# ============================================================

actual_flat = actual.reshape(-1)
predicted_flat = predicted.reshape(-1)

rmse = np.sqrt(
    mean_squared_error(
        actual_flat,
        predicted_flat
    )
)

mae = mean_absolute_error(
    actual_flat,
    predicted_flat
)

epsilon = 1e-8

mape = np.mean(
    np.abs(
        (actual_flat - predicted_flat)
        /
        np.maximum(
            np.abs(actual_flat),
            epsilon
        )
    )
) * 100


# ============================================================
# SAVE
# ============================================================

result_file = os.path.join(
    RESULT_DIR,
    f"{args.model}_T{args.T}_H{args.H}_predictions.npz"
)

np.savez(
    result_file,
    actual=actual,
    predicted=predicted
)


print("\n================================")
print("EVALUATION COMPLETED")
print("================================")

print("Model:", args.model)
print("T:", args.T)
print("H:", args.H)

print(f"RMSE: {rmse:.4f} kW")
print(f"MAE : {mae:.4f} kW")
print(f"MAPE: {mape:.2f}%")

print("\nSaved:", result_file)