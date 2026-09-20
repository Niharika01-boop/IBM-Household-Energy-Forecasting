import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models import VanillaLSTM


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/sequences"
MODEL_DIR = "artifacts/models"

T = 24
H = 1

BATCH_SIZE = 64
HIDDEN_SIZE = 64
EPOCHS = 30

LEARNING_RATE = 0.001
PATIENCE = 5


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# MODEL DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")

X_train = np.load(
    os.path.join(DATA_DIR, f"X_train_T{T}_H{H}.npy")
)

y_train = np.load(
    os.path.join(DATA_DIR, f"y_train_T{T}_H{H}.npy")
)

X_val = np.load(
    os.path.join(DATA_DIR, f"X_val_T{T}_H{H}.npy")
)

y_val = np.load(
    os.path.join(DATA_DIR, f"y_val_T{T}_H{H}.npy")
)


# ============================================================
# CONVERT TO TENSORS
# ============================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val,
    dtype=torch.float32
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = TensorDataset(
    X_train,
    y_train
)

val_dataset = TensorDataset(
    X_val,
    y_val
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

input_size = X_train.shape[2]

model = VanillaLSTM(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    output_size=H
).to(device)

print("\nModel:")
print(model)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")
patience_counter = 0

print("\n========== LSTM TRAINING STARTED ==========")

for epoch in range(EPOCHS):

    # ---------------- TRAIN ----------------

    model.train()

    train_loss = 0.0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        predictions = model(batch_X)

        loss = criterion(
            predictions,
            batch_y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        train_loss += (
            loss.item() * batch_X.size(0)
        )

    train_loss /= len(train_loader.dataset)


    # ---------------- VALIDATION ----------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_X)

            loss = criterion(
                predictions,
                batch_y
            )

            val_loss += (
                loss.item() * batch_X.size(0)
            )

    val_loss /= len(val_loader.dataset)


    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f}"
    )


    # ---------------- CHECKPOINT ----------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        patience_counter = 0

        torch.save(
            model.state_dict(),
            os.path.join(
                MODEL_DIR,
                "vanilla_lstm_T24_H1_best.pt"
            )
        )

        print("  → Best LSTM model saved!")

    else:

        patience_counter += 1


    # ---------------- EARLY STOPPING ----------------

    if patience_counter >= PATIENCE:

        print("\nEarly stopping triggered.")

        break


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("VANILLA LSTM TRAINING COMPLETED!")
print("========================================")

print(
    "Best validation loss:",
    best_val_loss
)

print(
    "Model saved at:",
    os.path.join(
        MODEL_DIR,
        "vanilla_lstm_T24_H1_best.pt"
    )
)