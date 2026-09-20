import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = "data/sequences"

T = 24
H = 1

BATCH_SIZE = 64


# ==========================================
# LOAD SEQUENCES
# ==========================================

print("Loading sequence data...")

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

X_test = np.load(
    os.path.join(DATA_DIR, f"X_test_T{T}_H{H}.npy")
)

y_test = np.load(
    os.path.join(DATA_DIR, f"y_test_T{T}_H{H}.npy")
)


# ==========================================
# CONVERT NUMPY → PYTORCH TENSORS
# ==========================================

X_train_tensor = torch.tensor(
    X_train, dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train, dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val, dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val, dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test, dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test, dtype=torch.float32
)


# ==========================================
# CREATE DATASETS
# ==========================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)


# ==========================================
# CREATE DATALOADERS
# ==========================================

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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# CHECK DATA
# ==========================================

print("\n========== DATASET SHAPES ==========")

print("X_train:", X_train_tensor.shape)
print("y_train:", y_train_tensor.shape)

print("X_val:", X_val_tensor.shape)
print("y_val:", y_val_tensor.shape)

print("X_test:", X_test_tensor.shape)
print("y_test:", y_test_tensor.shape)


print("\n========== DATALOADER CHECK ==========")

batch_X, batch_y = next(iter(train_loader))

print("Batch X shape:", batch_X.shape)
print("Batch y shape:", batch_y.shape)

print("Batch X dtype:", batch_X.dtype)
print("Batch y dtype:", batch_y.dtype)


# ==========================================
# DEVICE CHECK
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n========== DEVICE ==========")
print("Using device:", device)


print("\n========================================")
print("PYTORCH DATALOADER SETUP COMPLETED!")
print("========================================")