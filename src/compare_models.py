import os
import numpy as np
import pandas as pd
import torch

from models import SimpleRNN, VanillaLSTM, StackedLSTM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "artifacts/models"
EVAL_DIR = "artifacts/evaluation"
OUTPUT_DIR = "artifacts"

T = 24
H = 1
INPUT_SIZE = 13
HIDDEN_SIZE = 64


# ============================================================
# MODEL PARAMETER COUNT
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# READ METRICS FROM PREDICTION FILES
# ============================================================

def calculate_metrics(filename):

    path = os.path.join(
        EVAL_DIR,
        filename
    )

    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1
    )

    actual = data[:, 0]
    predicted = data[:, 1]

    rmse = np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )

    mae = np.mean(
        np.abs(
            actual - predicted
        )
    )

    epsilon = 1e-8

    mape = np.mean(
        np.abs(
            (actual - predicted)
            /
            np.maximum(
                np.abs(actual),
                epsilon
            )
        )
    ) * 100

    return rmse, mae, mape


# ============================================================
# CREATE MODELS
# ============================================================

rnn_model = SimpleRNN(
    INPUT_SIZE,
    HIDDEN_SIZE,
    H
)

lstm_model = VanillaLSTM(
    INPUT_SIZE,
    HIDDEN_SIZE,
    H
)

stacked_model = StackedLSTM(
    INPUT_SIZE,
    HIDDEN_SIZE,
    H,
    num_layers=2
)


# ============================================================
# PARAMETER COUNTS
# ============================================================

rnn_params = count_parameters(
    rnn_model
)

lstm_params = count_parameters(
    lstm_model
)

stacked_params = count_parameters(
    stacked_model
)


# ============================================================
# LOAD METRICS
# ============================================================

rnn_metrics = calculate_metrics(
    "simple_rnn_predictions.csv"
)

lstm_metrics = calculate_metrics(
    "vanilla_lstm_predictions.csv"
)

stacked_metrics = calculate_metrics(
    "stacked_lstm_predictions.csv"
)


# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Simple RNN",
        "Vanilla LSTM",
        "Stacked LSTM"
    ],

    "Parameters": [
        rnn_params,
        lstm_params,
        stacked_params
    ],

    "RMSE_kW": [
        rnn_metrics[0],
        lstm_metrics[0],
        stacked_metrics[0]
    ],

    "MAE_kW": [
        rnn_metrics[1],
        lstm_metrics[1],
        stacked_metrics[1]
    ],

    "MAPE_percent": [
        rnn_metrics[2],
        lstm_metrics[2],
        stacked_metrics[2]
    ]
})


# ============================================================
# ROUND VALUES
# ============================================================

results["RMSE_kW"] = results["RMSE_kW"].round(4)
results["MAE_kW"] = results["MAE_kW"].round(4)
results["MAPE_percent"] = results["MAPE_percent"].round(2)


# ============================================================
# PRINT
# ============================================================

print("\n==============================================")
print("MODEL COMPARISON")
print("==============================================")

print(results.to_string(index=False))


# ============================================================
# SAVE CSV
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

results.to_csv(
    output_file,
    index=False
)


print("\nComparison saved to:")
print(output_file)


# ============================================================
# PARAMETER DETAILS
# ============================================================

print("\n========== PARAMETER COUNT ==========")

print(
    f"Simple RNN     : {rnn_params:,}"
)

print(
    f"Vanilla LSTM   : {lstm_params:,}"
)

print(
    f"Stacked LSTM   : {stacked_params:,}"
)


print("\n==============================================")
print("MODEL COMPARISON COMPLETED!")
print("==============================================")