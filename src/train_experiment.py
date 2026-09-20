import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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
# CONFIGURATION
# ============================================================

DATA_DIR = "data/sequences"
MODEL_DIR = "artifacts/models"

BATCH_SIZE = 64
HIDDEN_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 5

T = args.T
H = args.H


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Model:", args.model)
print("T:", T)
print("H:", H)


# ============================================================
# LOAD DATA
# ============================================================

X_train = np.load(
    os.path.join(
        DATA_DIR,
        f"X_train_T{T}_H{H}.npy"
    )
)

y_train = np.load(
    os.path.join(
        DATA_DIR,
        f"y_train_T{T}_H{H}.npy"
    )
)

X_val = np.load(
    os.path.join(
        DATA_DIR,
        f"X_val_T{T}_H{H}.npy"
    )
)

y_val = np.load(
    os.path.join(
        DATA_DIR,
        f"y_val_T{T}_H{H}.npy"
    )
)


print("\nTrain:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)


# ============================================================
# TENSORS
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
# DATALOADER
# ============================================================

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

input_size = X_train.shape[2]

if args.model == "rnn":

    model = SimpleRNN(
        input_size,
        HIDDEN_SIZE,
        H
    )

elif args.model == "lstm":

    model = VanillaLSTM(
        input_size,
        HIDDEN_SIZE,
        H
    )

else:

    model = StackedLSTM(
        input_size,
        HIDDEN_SIZE,
        H,
        num_layers=2
    )


model = model.to(device)

print("\nModel:")
print(model)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# CHECKPOINT NAME
# ============================================================

model_name = args.model

checkpoint_name = (
    f"{model_name}_T{T}_H{H}_best.pt"
)

checkpoint_path = os.path.join(
    MODEL_DIR,
    checkpoint_name
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# TRAIN
# ============================================================

best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):

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
            1.0
        )

        optimizer.step()

        train_loss += (
            loss.item() *
            batch_X.size(0)
        )

    train_loss /= len(
        train_loader.dataset
    )


    # ========================================================
    # VALIDATION
    # ========================================================

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
                loss.item() *
                batch_X.size(0)
            )

    val_loss /= len(
        val_loader.dataset
    )


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train: {train_loss:.6f} | "
        f"Val: {val_loss:.6f}"
    )


    if val_loss < best_val_loss:

        best_val_loss = val_loss
        patience_counter = 0

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        print("  Best model saved.")

    else:

        patience_counter += 1


    if patience_counter >= PATIENCE:

        print("Early stopping.")

        break


print("\n================================")
print("TRAINING COMPLETED")
print("================================")

print("Saved:", checkpoint_path)