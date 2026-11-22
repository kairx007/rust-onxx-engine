"""
src/train.py
────────────
Training engine for LOB mid-price prediction models.

Supports:
    - DeepLOB (CNN + LSTM baseline)
    - TransformerLOB (CNN + Transformer)

Features:
    - FI-2010 benchmark protocol: train days 1-7, val day 8, test days 9-10
    - OneCycleLR scheduler: adaptive learning rate scheduling
    - Class-weighted loss: handles dataset imbalance
    - Gradient clipping: stabilises sequence model training

Usage (run from project root):
        python src/train.py

Expected project layout:
        data/raw/FI2010_train.csv
        data/raw/FI2010_test.csv
        src/dataset.py        <- LOBDataset
        src/models.py         <- DeepLOB, TransformerLOB
        src/train.py          <- this file
"""

import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import OneCycleLR

from dataset import LOBDataset
from models import DeepLOB, TransformerLOB


# ══════════════════════════════════════════════════════════════════════════════
# 1. HYPER-PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_CSV_PATH = "data/raw/FI2010_train.csv"
TEST_CSV_PATH  = "data/raw/FI2010_test.csv"
MAX_TRAIN_ROWS = int(os.getenv("MAX_TRAIN_ROWS", "200000"))  # cap train rows for faster iterations
_MAX_TEST_ROWS_ENV = os.getenv("MAX_TEST_ROWS", "")
MAX_TEST_ROWS  = int(_MAX_TEST_ROWS_ENV) if _MAX_TEST_ROWS_ENV else None  # None = use full FI2010_test.csv
BATCH_SIZE  = 32
EPOCHS      = int(os.getenv("EPOCHS", "15"))
LR          = 1e-3
NUM_CLASSES = 3            # 0=Down, 1=Stationary, 2=Up
MODEL_TYPE  = os.getenv("MODEL_TYPE", "transformer")  # "deeplob" or "transformer"


# ══════════════════════════════════════════════════════════════════════════════
# 2. DEVICE  (CUDA → MPS → CPU)
# ══════════════════════════════════════════════════════════════════════════════
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATASET & DATALOADER (FI-2010 BENCHMARK SPLIT)
# ══════════════════════════════════════════════════════════════════════════════
def _count_effective_rows(csv_path: str) -> int:
    """Count usable rows in FI-2010 CSV (excluding first metadata row)."""
    with open(csv_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    return max(0, total_lines - 1)


def build_loaders(train_csv_path: str, test_csv_path: str, max_train_rows: int,
                  max_test_rows: int | None, batch_size: int) -> tuple:
    """
    Build train/val/test loaders using FI-2010 benchmark protocol.
    
    Args:
        train_csv_path (str): FI2010_train.csv (days 1-7)
        test_csv_path (str): FI2010_test.csv (days 8-10)
        max_train_rows (int): Max rows to read from train CSV
        max_test_rows (int | None): Max rows to read from test CSV (None = all)
        batch_size (int): Batch size
    
    Returns:
        (train_loader, val_loader, test_loader, train_dataset)
    """
    train_dataset = LOBDataset(train_csv_path, nrows=max_train_rows)

    test_effective_rows = _count_effective_rows(test_csv_path)
    if max_test_rows is not None:
        test_effective_rows = min(test_effective_rows, max_test_rows)

    # FI-2010 benchmark split on test CSV: day 8 for validation, days 9-10 for test.
    # In this dataset format, the test file is chronological day 8 → day 10.
    # We approximate days by contiguous row blocks (1/3 for day 8, 2/3 for days 9-10).
    val_rows = test_effective_rows // 3
    test_rows = test_effective_rows - val_rows
    if val_rows <= 100 or test_rows <= 100:
        raise ValueError(
            f"Not enough rows for benchmark split: val_rows={val_rows}, test_rows={test_rows}"
        )

    val_dataset = LOBDataset(
        test_csv_path,
        nrows=val_rows,
        mu=train_dataset.mu,
        sigma=train_dataset.sigma,
        start_row=0,
    )
    test_dataset = LOBDataset(
        test_csv_path,
        nrows=test_rows,
        mu=train_dataset.mu,
        sigma=train_dataset.sigma,
        start_row=val_rows,
    )

    # DataLoaders: shuffle=False mandatory for time-series (no future leakage)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    print(f"  Train windows : {len(train_dataset):,}  (days 1-7)")
    print(f"  Val windows   : {len(val_dataset):,}  (day 8)")
    print(f"  Test windows  : {len(test_dataset):,}  (days 9-10)")
    print(
        f"  Batches/epoch : train={len(train_loader):,}  |  "
        f"val={len(val_loader):,}  |  test={len(test_loader):,}"
    )
    
    return train_loader, val_loader, test_loader, train_dataset




# ══════════════════════════════════════════════════════════════════════════════
# 4. CLASS WEIGHTS (FIX FOR IMBALANCE)
# ══════════════════════════════════════════════════════════════════════════════
def compute_class_weights(train_data, num_classes: int, device: torch.device) -> torch.FloatTensor:
    """
    Compute class weights to penalise minority classes more heavily.
    
    Weight formula:  w_c = N / (K * count_c)
      N = total training windows
      K = number of classes
      count_c = number of windows with label c
    """
    if isinstance(train_data, Subset):
        full_Y = train_data.dataset.Y
        indices = train_data.indices
        window_size = train_data.dataset.window_size
        labels = full_Y[[i + window_size for i in indices]]
    else:
        full_Y = train_data.Y
        window_size = train_data.window_size
        labels = full_Y[window_size:]
    
    # ⚠️ DEBUG: Check for invalid/missing labels
    unique_labels = np.unique(labels)
    print(f"  Unique training labels: {unique_labels.tolist()}")
    print(f"  Label range: [{labels.min()}, {labels.max()}]")
    
    if not np.all((labels >= 0) & (labels < 3)):
        invalid_mask = (labels < 0) | (labels >= 3)
        print(f"  ⚠️  ERROR: {invalid_mask.sum()} invalid labels found!")
        raise ValueError(f"Invalid labels in training subset: {np.unique(labels[invalid_mask]).tolist()}")
    
    if np.any(np.isnan(labels)):
        print(f"  ⚠️  WARNING: NaN values detected in labels!")
    
    N = len(labels)
    weights = []
    print(" Class weight calculation (training set):")
    print(f"  {'Class':<14} {'Count':>8}  {'Weight':>8}")
    print(f"  {'─'*34}")
    for c in range(num_classes):
        count = (labels == c).sum()
        w     = N / (num_classes * count) if count > 0 else 0.0
        weights.append(w)
        class_name = ["Down", "Stationary", "Up"][c]
        print(f"  {class_name:<14} {int(count):>8,}  {w:>8.4f}")
    
    total_counted = sum((labels == c).sum() for c in range(num_classes))
    print(f"  {'TOTAL':>14} {int(total_counted):>8,}")
    
    weight_tensor = torch.FloatTensor(weights).to(device)
    return weight_tensor

# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING & VALIDATION LOOP (WITH WALK-FORWARD VALIDATION)
# ══════════════════════════════════════════════════════════════════════════════
def validate(model, val_loader, criterion, device):
    """Run validation on the validation set and return loss + accuracy."""
    model.eval()
    val_loss = 0.0
    correct  = 0
    total    = 0
    
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            logits = model(X)
            loss   = criterion(logits, y)
            
            val_loss += loss.item() * X.size(0)
            correct  += (logits.argmax(dim=1) == y).sum().item()
            total    += X.size(0)
    
    avg_val_loss = val_loss / total
    val_acc      = 100.0 * correct / total
    return avg_val_loss, val_acc


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs):
    """
    Train with walk-forward validation and OneCycleLR scheduler.
    
    Args:
        model: PyTorch model
        train_loader: training DataLoader
        val_loader: validation DataLoader (optional)
        criterion: loss function
        optimizer: optimizer
        scheduler: learning rate scheduler (OneCycleLR)
        device: torch device
        epochs: number of epochs
    """
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        # ── TRAINING ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        t0 = time.perf_counter()
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # OneCycleLR scheduler step (per batch, not per epoch!)
            scheduler.step()
            
            train_loss += loss.item() * X.size(0)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += X.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(
                    f"  [Epoch {epoch}/{epochs}]"
                    f"  step {batch_idx+1}/{len(train_loader)}"
                    f"  |  loss: {loss.item():.4f}"
                )
        
        avg_train_loss = train_loss / train_total
        train_acc = 100.0 * train_correct / train_total
        elapsed = time.perf_counter() - t0
        
        # ── VALIDATION ────────────────────────────────────────────────────────
        if val_loader is not None:
            avg_val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(
                f"\n{'─'*70}\n"
                f"Epoch {epoch}/{epochs}"
                f"  |  train loss: {avg_train_loss:.4f}  |  train acc: {train_acc:.2f}%"
                f"  |  val loss: {avg_val_loss:.4f}  |  val acc: {val_acc:.2f}%"
                f"  |  time: {elapsed:.1f}s"
                f"\n{'─'*70}\n"
            )

            # Track best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        else:
            print(
                f"\n{'─'*70}\n"
                f"Epoch {epoch}/{epochs}"
                f"  |  train loss: {avg_train_loss:.4f}  |  train acc: {train_acc:.2f}%"
                f"  |  time: {elapsed:.1f}s"
                f"\n{'─'*70}\n"
            )


def evaluate(model, loader, criterion, device, split_name: str):
    """Evaluate final model performance on a named split."""
    loss, acc = validate(model, loader, criterion, device)
    print(f"{split_name:<14}: loss={loss:.4f} | acc={acc:.2f}%")
    return loss, acc


# ══════════════════════════════════════════════════════════════════════════════
# 6. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  LOB Mid-Price Prediction — Training Engine")
    print("=" * 70)

    device = get_device()
    print(f"\n  Device          : {device}")
    print(f"  Model           : {MODEL_TYPE.upper()}")
    print(f"  Optimizer       : Adam (lr={LR})")
    print(f"  Scheduler       : OneCycleLR")
    print(f"  Validation      : FI-2010 benchmark (train 1-7 | val 8 | test 9-10)")
    
    if not os.path.exists(TRAIN_CSV_PATH):
        raise FileNotFoundError(f"Missing training CSV: {TRAIN_CSV_PATH}")
    if not os.path.exists(TEST_CSV_PATH):
        raise FileNotFoundError(f"Missing test CSV: {TEST_CSV_PATH}")

    # Load dataset with FI-2010 benchmark split
    print(f"\n  Loading benchmark datasets...")
    train_loader, val_loader, test_loader, train_dataset = build_loaders(
        TRAIN_CSV_PATH,
        TEST_CSV_PATH,
        max_train_rows=MAX_TRAIN_ROWS,
        max_test_rows=MAX_TEST_ROWS,
        batch_size=BATCH_SIZE,
    )
    
    # Instantiate model
    if MODEL_TYPE.lower() == "transformer":
        model = TransformerLOB(
            num_classes=NUM_CLASSES,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
        ).to(device)
    else:  # deeplob
        model = DeepLOB(
            num_classes=NUM_CLASSES,
            lstm_hidden=64,
            lstm_layers=1,
        ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model           : {MODEL_TYPE}  ({n_params:,} trainable params)")
    print(f"  Loss            : CrossEntropyLoss (class-weighted)")
    print(f"  Epochs          : {EPOCHS}  |  Batch size : {BATCH_SIZE}")
    print(f"\n{'='*70}\n")
    
    # Compute class weights from training data
    class_weights = compute_class_weights(
        train_data=train_dataset,
        num_classes=NUM_CLASSES,
        device=device,
    )
    print(f"\n  Weight tensor   : {class_weights.tolist()}\n")
    
    # Setup loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # OneCycleLR scheduler: automatically adjusts LR over training
    # steps_per_epoch = len(train_loader), total_steps = steps_per_epoch * EPOCHS
    total_steps = len(train_loader) * EPOCHS
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LR,       # strict Phase 3 setting
        total_steps=total_steps,
        pct_start=0.3,   # 30% of training for warmup
        anneal_strategy='cos',  # cosine annealing
    )
    print(f"  OneCycleLR      : max_lr={LR}, total_steps={total_steps}")
    print(f"\n{'='*70}\n")
    
    # Train with validation
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, EPOCHS)

    # Final benchmark evaluation
    print(f"\n{'='*70}")
    print("Final benchmark evaluation")
    print(f"{'='*70}")
    evaluate(model, val_loader, criterion, device, "Validation(D8)")
    evaluate(model, test_loader, criterion, device, "Test(D9-D10)")
    
    # Save checkpoint with architecture-specific name to prevent confusion.
    model_type_norm = MODEL_TYPE.lower().strip()
    ckpt = f"{model_type_norm}_checkpoint.pt"
    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_type": model_type_norm,
        },
        ckpt,
    )
    print(f"\n✅ Training complete.  Checkpoint saved → {ckpt}")