import os
import numpy as np

# ==============================
# CONFIGURATION
# ==============================

DATA_DIR = "data/scaled"
OUTPUT_DIR = "data/sequences"

# Past time steps
T = 24

# Forecast horizon
H = 1


# ==============================
# CREATE OUTPUT DIRECTORY
# ==============================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# LOAD DATA
# ==============================

print("Loading scaled data...")

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))

y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))


print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ==============================
# SEQUENCE CREATION FUNCTION
# ==============================

def create_sequences(X, y, T, H):

    X_sequences = []
    y_sequences = []

    for i in range(len(X) - T - H + 1):

        # Past T time steps
        X_sequences.append(
            X[i:i + T]
        )

        # Future H target values
        y_sequences.append(
            y[i + T:i + T + H]
        )

    return np.array(X_sequences), np.array(y_sequences)


# ==============================
# CREATE SEQUENCES
# ==============================

print("\nCreating sequences...")

X_train_seq, y_train_seq = create_sequences(
    X_train, y_train, T, H
)

X_val_seq, y_val_seq = create_sequences(
    X_val, y_val, T, H
)

X_test_seq, y_test_seq = create_sequences(
    X_test, y_test, T, H
)


# ==============================
# PRINT SHAPES
# ==============================

print("\n========== SEQUENCE SHAPES ==========")

print("X_train_seq:", X_train_seq.shape)
print("y_train_seq:", y_train_seq.shape)

print("X_val_seq:", X_val_seq.shape)
print("y_val_seq:", y_val_seq.shape)

print("X_test_seq:", X_test_seq.shape)
print("y_test_seq:", y_test_seq.shape)


# ==============================
# SAVE SEQUENCES
# ==============================

np.save(
    os.path.join(OUTPUT_DIR, f"X_train_T{T}_H{H}.npy"),
    X_train_seq
)

np.save(
    os.path.join(OUTPUT_DIR, f"y_train_T{T}_H{H}.npy"),
    y_train_seq
)

np.save(
    os.path.join(OUTPUT_DIR, f"X_val_T{T}_H{H}.npy"),
    X_val_seq
)

np.save(
    os.path.join(OUTPUT_DIR, f"y_val_T{T}_H{H}.npy"),
    y_val_seq
)

np.save(
    os.path.join(OUTPUT_DIR, f"X_test_T{T}_H{H}.npy"),
    X_test_seq
)

np.save(
    os.path.join(OUTPUT_DIR, f"y_test_T{T}_H{H}.npy"),
    y_test_seq
)


print("\n========================================")
print("SEQUENCE GENERATION COMPLETED SUCCESSFULLY!")
print("========================================")

print("\nSaved inside:")
print(OUTPUT_DIR)