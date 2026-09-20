import os
import numpy as np

DATA_DIR = "data/scaled"
OUTPUT_DIR = "data/sequences"

SEQUENCE_LENGTHS = [24, 72, 168]
HORIZONS = [1, 24, 48]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_sequences(X, y, T, H):

    X_sequences = []
    y_sequences = []

    for i in range(len(X) - T - H + 1):

        X_sequences.append(
            X[i:i + T]
        )

        y_sequences.append(
            y[i + T:i + T + H, 0]
        )

    return (
        np.array(X_sequences),
        np.array(y_sequences)
    )


print("Loading scaled data...")

X_train = np.load(
    os.path.join(DATA_DIR, "X_train.npy")
)

y_train = np.load(
    os.path.join(DATA_DIR, "y_train.npy")
)

X_val = np.load(
    os.path.join(DATA_DIR, "X_val.npy")
)

y_val = np.load(
    os.path.join(DATA_DIR, "y_val.npy")
)

X_test = np.load(
    os.path.join(DATA_DIR, "X_test.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, "y_test.npy")
)


for T in SEQUENCE_LENGTHS:

    for H in HORIZONS:

        print("\n================================")
        print(f"T = {T}, H = {H}")
        print("================================")

        X_train_seq, y_train_seq = create_sequences(
            X_train, y_train, T, H
        )

        X_val_seq, y_val_seq = create_sequences(
            X_val, y_val, T, H
        )

        X_test_seq, y_test_seq = create_sequences(
            X_test, y_test, T, H
        )

        print("Train:", X_train_seq.shape, y_train_seq.shape)
        print("Val  :", X_val_seq.shape, y_val_seq.shape)
        print("Test :", X_test_seq.shape, y_test_seq.shape)

        np.save(
            os.path.join(
                OUTPUT_DIR,
                f"X_train_T{T}_H{H}.npy"
            ),
            X_train_seq
        )

        np.save(
            os.path.join(
                OUTPUT_DIR,
                f"y_train_T{T}_H{H}.npy"
            ),
            y_train_seq
        )

        np.save(
            os.path.join(
                OUTPUT_DIR,
                f"X_val_T{T}_H{H}.npy"
            ),
            X_val_seq
        )

        np.save(
            os.path.join(
                OUTPUT_DIR,
                f"y_val_T{T}_H{H}.npy"
            ),
            y_val_seq
        )

        np.save(
            os.path.join(
                OUTPUT_DIR,
                f"X_test_T{T}_H{H}.npy"
            ),
            X_test_seq
        )

        np.save(
            os.path.join(
                OUTPUT_DIR,
                f"y_test_T{T}_H{H}.npy"
            ),
            y_test_seq
        )


print("\n================================")
print("ALL HORIZON SEQUENCES CREATED!")
print("================================")