from pathlib import Path
import numpy as np
import pandas as pd


RESULTS_DIR = Path("artifacts/results")
OUTPUT_FILE = RESULTS_DIR / "results_table.csv"


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    # Flatten all forecast horizons
    actual = actual.reshape(-1)
    predicted = predicted.reshape(-1)

    error = actual - predicted

    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))

    # Avoid division by zero in MAPE
    nonzero = np.abs(actual) > 1e-8

    if np.any(nonzero):
        mape = np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
    else:
        mape = np.nan

    return rmse, mae, mape


def parse_filename(filename):
    # Example:
    # lstm_T72_H48_predictions.npz

    stem = filename.replace("_predictions.npz", "")
    parts = stem.split("_")

    model = parts[0]
    T = int(parts[1][1:])
    H = int(parts[2][1:])

    return model, T, H


def main():

    files = sorted(
        RESULTS_DIR.glob("*_predictions.npz")
    )

    if not files:
        print("No prediction files found.")
        return

    rows = []

    for file in files:

        model, T, H = parse_filename(file.name)

        data = np.load(file)

        actual = data["actual"]
        predicted = data["predicted"]

        rmse, mae, mape = calculate_metrics(
            actual,
            predicted
        )

        rows.append({
            "Model": model,
            "Sequence_Length": T,
            "Horizon": H,
            "RMSE_kW": rmse,
            "MAE_kW": mae,
            "MAPE_percent": mape
        })

    df = pd.DataFrame(rows)

    model_order = {
        "rnn": 0,
        "lstm": 1,
        "stacked": 2
    }

    df["Model_Order"] = df["Model"].map(model_order)

    df = df.sort_values(
        [
            "Sequence_Length",
            "Horizon",
            "Model_Order"
        ]
    )

    df = df.drop(columns=["Model_Order"])

    # Round values for readability
    df["RMSE_kW"] = df["RMSE_kW"].round(4)
    df["MAE_kW"] = df["MAE_kW"].round(4)
    df["MAPE_percent"] = df["MAPE_percent"].round(2)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("=" * 60)
    print("FINAL RESULTS TABLE CREATED")
    print("=" * 60)

    print(f"Experiments found: {len(df)}")
    print(f"Saved: {OUTPUT_FILE}")

    print("\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()