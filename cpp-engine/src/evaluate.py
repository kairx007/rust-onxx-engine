"""
src/evaluate.py
───────────────
Evaluation script for trained LOB models (DeepLOB / TransformerLOB).

Usage (run from project root):
    python src/evaluate.py

Outputs:
    - Classification report printed to console
    - confusion_matrix.png saved to project root

Expected project layout:
    data/raw/FI2010_train.csv
    data/raw/FI2010_test.csv
    src/dataset.py          <- LOBDataset
    src/models.py           <- DeepLOB, TransformerLOB
    src/evaluate.py         <- this file
    *_checkpoint.pt         <- saved by train.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

from dataset import LOBDataset
from models import DeepLOB, TransformerLOB


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CSV_PATH    = "data/raw/FI2010_test.csv"
TRAIN_CSV_PATH = "data/raw/FI2010_train.csv"
CKPT_PATH   = os.getenv("CKPT_PATH", "")
NROWS       = int(os.getenv("EVAL_NROWS", "10000"))  # number of windows to evaluate
BATCH_SIZE  = 32
NUM_CLASSES = 3
CLASS_NAMES = ["Down", "Stationary", "Up"]
CM_SAVE_PATH = "confusion_matrix.png"


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
# 3. DATALOADER
#    LOBDataset has no nrows param — use Subset to cap at NROWS windows
# ══════════════════════════════════════════════════════════════════════════════
def _load_train_stats(train_csv_path: str, processed_dir: str = "data/processed") -> tuple[np.ndarray, np.ndarray]:
    """Load train normalization stats from cache or compute from train CSV."""
    mu_path = os.path.join(processed_dir, "mu.npy")
    sigma_path = os.path.join(processed_dir, "sigma.npy")
    if os.path.exists(mu_path) and os.path.exists(sigma_path):
        mu = np.load(mu_path)
        sigma = np.load(sigma_path)
        print(f"  Normalization : loaded mu/sigma from {processed_dir}/")
        return mu, sigma

    print("  Normalization : processed stats not found, computing from train CSV")
    train_dataset = LOBDataset(train_csv_path)
    return train_dataset.mu, train_dataset.sigma


def build_loader(csv_path: str, train_csv_path: str, nrows: int, batch_size: int) -> DataLoader:
    mu, sigma = _load_train_stats(train_csv_path)
    full_dataset = LOBDataset(csv_path, mu=mu, sigma=sigma)
    n_use        = min(nrows, len(full_dataset))
    dataset      = Subset(full_dataset, range(n_use))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,      # must be False — preserves time-series order
        num_workers=0,
        pin_memory=True,
    )
    print(f"  Test windows  : {n_use:,}  (of {len(full_dataset):,} available)")
    print(f"  Batches       : {len(loader):,}")
    return loader


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL — load matching architecture then inject trained weights
# ══════════════════════════════════════════════════════════════════════════════
def _detect_model_type(ckpt: dict) -> str:
    """Detect model type from checkpoint metadata or state_dict keys."""
    model_type = str(ckpt.get("model_type", "")).lower().strip()
    if model_type in {"deeplob", "transformer"}:
        return model_type

    state_dict = ckpt.get("model_state_dict", {})
    if any(k.startswith("transformer_encoder.") for k in state_dict.keys()):
        return "transformer"
    return "deeplob"


def _resolve_ckpt_path() -> str:
    """Resolve checkpoint path from env or common defaults."""
    if CKPT_PATH:
        return CKPT_PATH

    candidates = ["transformer_checkpoint.pt", "deeplob_checkpoint.pt"]
    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "No checkpoint found. Expected one of: transformer_checkpoint.pt, deeplob_checkpoint.pt"
    )


def load_model(ckpt_path: str, device: torch.device) -> tuple[nn.Module, str]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # map_location ensures the checkpoint loads correctly regardless of whether
    # it was saved on GPU and we're now running on CPU (or vice-versa)
    ckpt = torch.load(ckpt_path, map_location=device)
    model_type = _detect_model_type(ckpt)

    if model_type == "transformer":
        model = TransformerLOB(
            num_classes=NUM_CLASSES,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
        )
    else:
        model = DeepLOB(num_classes=NUM_CLASSES)

    model.load_state_dict(ckpt["model_state_dict"])

    model.to(device)
    model.eval()    # disables Dropout & switches BatchNorm to running stats

    saved_epoch = ckpt.get("epoch", "unknown")
    print(f"  Checkpoint    : {ckpt_path}  (saved at epoch {saved_epoch})")
    print(f"  Model type    : {model_type}")
    return model, model_type


# ══════════════════════════════════════════════════════════════════════════════
# 5. INFERENCE LOOP
# ══════════════════════════════════════════════════════════════════════════════
def run_inference(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        y_true : (N,) int array of ground-truth labels
        y_pred : (N,) int array of predicted labels
    """
    all_true = []
    all_pred = []

    # torch.no_grad() disables gradient tracking entirely — no backward graph
    # is built, which halves memory usage and speeds up inference significantly
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)   # (B, 1, 100, 40)
            y = y.to(device, non_blocking=True)   # (B,)

            # Forward pass only — no loss, no backward
            logits = model(X)                      # (B, 3)

            # argmax over class dimension → index of highest logit = prediction
            preds = torch.argmax(logits, dim=1)    # (B,)

            all_true.append(y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return y_true, y_pred


# ══════════════════════════════════════════════════════════════════════════════
# 6. METRICS
# ══════════════════════════════════════════════════════════════════════════════
def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )
    print(report)


# ══════════════════════════════════════════════════════════════════════════════
# 7. CONFUSION MATRIX PLOT
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    save_path: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)

    # Normalise rows → recall per class (values 0–1)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DeepLOB — Confusion Matrix", fontsize=14, fontweight="bold", y=1.01)

    # ── Left: raw counts ──────────────────────────────────────────────────────
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        ax=axes[0],
    )
    axes[0].set_title("Raw Counts")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # ── Right: normalised (recall) ────────────────────────────────────────────
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.5,
        vmin=0.0,
        vmax=1.0,
        ax=axes[1],
    )
    axes[1].set_title("Normalised (Row = True Label)")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Confusion matrix saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  LOB Model Evaluation")
    print("=" * 60)

    device = get_device()
    print(f"\n  Device        : {device}")

    if not os.path.exists(TRAIN_CSV_PATH):
        raise FileNotFoundError(f"Missing training CSV for normalization stats: {TRAIN_CSV_PATH}")

    print(f"\n  Loading test data ({NROWS:,} windows from {CSV_PATH})...")
    loader = build_loader(CSV_PATH, TRAIN_CSV_PATH, nrows=NROWS, batch_size=BATCH_SIZE)

    ckpt_path = _resolve_ckpt_path()
    print(f"\n  Loading model weights from {ckpt_path}...")
    model, model_type = load_model(ckpt_path, device)

    print("\n  Running inference...")
    y_true, y_pred = run_inference(model, loader, device)

    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, CM_SAVE_PATH)

    # Overall accuracy summary
    acc = (y_true == y_pred).mean() * 100
    print(f"  Overall accuracy : {acc:.2f}%")
    print(f"  Evaluated model  : {model_type}")
    print("\nEvaluation complete.")