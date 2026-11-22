import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LOBDataset(Dataset):
    def __init__(self, csv_path, window_size=100, nrows=50_000, mu=None, sigma=None, start_row=0):
        """
        Args:
            csv_path (str): Path to FI2010_train.csv or FI2010_test.csv
            window_size (int): Sliding window size (T=100)
            nrows (int): Number of rows to load from CSV
            mu (np.ndarray): Pre-computed feature means (40,). If None, computed from this data.
                             Pass training mu when constructing validation/test datasets.
            sigma (np.ndarray): Pre-computed feature stds (40,). If None, computed from this data.
            start_row (int): Starting data row offset (after metadata line) for chronological slicing.
        """
        print(f"⏳ Loading data from {csv_path}...")
        
        # 1. Load data & Clean it
        skiprows = 1 + max(0, int(start_row))
        df = pd.read_csv(csv_path, skiprows=skiprows, header=None, nrows=nrows)
        
        # Drop the garbage index column written by data provider
        df = df.iloc[:, 1:] 
        df.columns = range(df.shape[1])
        
        # 2. Extract Features (X) and Labels (Y)
        # FI-2010 layout used in this project:
        #   Columns 0–39:   40 LOB features (10 levels × bid/ask price+volume)
        #   Column 144:     label for shortest horizon (k=10)
        
        print(f"  CSV shape after dropping index: {df.shape}")
        print(f"  Column count: {df.shape[1]}")
        
        X_raw = df.iloc[:, 0:40].values.astype(np.float32)  # (N, 40)
        
        # Fixed label column by project requirement.
        label_col = 144
        if label_col >= df.shape[1]:
            raise ValueError(
                f"Expected label column {label_col}, but dataset only has {df.shape[1]} columns "
                "after dropping the index column."
            )

        raw_labels = df.iloc[:, label_col].values
        print(f"  ✅ Using fixed label column {label_col}")
        print(f"  Raw label stats: min={raw_labels.min()}, max={raw_labels.max()}, "
              f"unique={np.unique(raw_labels).tolist()}")
        
        # FI-2010 labels: 1=Up, 2=Stationary, 3=Down
        # PyTorch CrossEntropyLoss expects 0-indexed: remap to 0/1/2
        Y = (raw_labels - 1).astype(np.int64)  # (N,) in {0, 1, 2}
        
        # ⚠️ VALIDATION: Check for valid labels
        unique_Y = np.unique(Y)
        valid_range = np.all((Y >= 0) & (Y < 3))
        
        if not valid_range:
            print(f"  ⚠️  WARNING: Some labels outside {{0, 1, 2}}:")
            print(f"     Remapped unique values: {unique_Y.tolist()}")
            print(f"     Raw unique values: {np.unique(raw_labels).tolist()}")
            # Don't clip - this indicates wrong column
            raise ValueError(f"Label column {label_col} contains invalid values: {unique_Y}")
        else:
            print(f"  ✅ Labels valid: {unique_Y.tolist()}")
        
        self.window_size = window_size
        
        # 3. Z-SCORE NORMALIZATION
        # CRITICAL: mu and sigma must be computed on TRAINING data only.
        # If this is training: compute mu/sigma here.
        # If this is val/test: receive pre-computed mu/sigma from outside (prevents data leakage).
        if mu is None or sigma is None:
            self.mu = X_raw.mean(axis=0)  # (40,)
            self.sigma = X_raw.std(axis=0) + 1e-8  # (40,) with epsilon to avoid division by zero
            print(f"  ℹ️  Normalization stats computed from this dataset (training mode).")
        else:
            self.mu = mu
            self.sigma = sigma
            print(f"  ℹ️  Using externally supplied normalization stats (val/test mode).")
        
        # Apply normalization: X_norm = (X - mu) / sigma
        self.X = (X_raw - self.mu) / self.sigma  # (N, 40), float32
        self.Y = Y
        
        total_windows = len(self.X) - self.window_size
        print(f"✅ Dataset built! Rows: {len(self.X):,} | Windows: {total_windows:,}")

    def __len__(self):
        # Window i uses rows [i .. i+window_size-1], label at row i+window_size
        # Last valid i = len(X) - window_size - 1, so total = len(X) - window_size
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        # 4. Sliding Window Logic
        # Grab T consecutive normalized rows: x_window[0..T-1]
        x_window = self.X[idx : idx + self.window_size]  # (100, 40)
        
        # Label at the tick immediately after the window ends
        y_label = self.Y[idx + self.window_size]  # scalar int
        
        # ⚠️ VALIDATION: Check for invalid label before creating tensor
        if y_label < 0 or y_label >= 3:
            raise ValueError(
                f"Invalid label {y_label} at index {idx + self.window_size}. "
                f"Expected label in {{0, 1, 2}}. "
                f"Unique labels: {np.unique(self.Y)}"
            )
        
        # CNNs expect (C, H, W) format. Add channel dimension: (1, 100, 40)
        x_tensor = torch.from_numpy(x_window).unsqueeze(0)  # (1, 100, 40), float32
        y_tensor = torch.tensor(y_label, dtype=torch.long)   # scalar
        
        return x_tensor, y_tensor

    def save_processed(self, out_dir="data/processed"):
        """
        Save normalized features, labels, and normalization stats to disk.
        
        Files written:
            data/processed/X.npy     — normalized features (N, 40) float32
            data/processed/Y.npy     — integer labels       (N,)   int64
            data/processed/mu.npy    — feature means        (40,)  float32
            data/processed/sigma.npy — feature stds         (40,)  float32
        
        Call this ONCE after building the training dataset. Val/test datasets 
        should be constructed via load_processed() or by passing mu/sigma.
        
        Usage:
            train_data = LOBDataset("data/raw/FI2010_train.csv")
            train_data.save_processed()  # fills data/processed/
        """
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "X.npy"),     self.X)
        np.save(os.path.join(out_dir, "Y.npy"),     self.Y)
        np.save(os.path.join(out_dir, "mu.npy"),    self.mu)
        np.save(os.path.join(out_dir, "sigma.npy"), self.sigma)
        print(f"💾 Processed data saved → {out_dir}/ "
              f"(X: {self.X.shape}, Y: {self.Y.shape})")

    @classmethod
    def load_processed(cls, processed_dir="data/processed", window_size=100):
        """
        Reconstruct a LOBDataset instantly from saved .npy files.
        Skips CSV parsing entirely — much faster on subsequent runs.
        
        Usage:
            # First run: build from CSV and save
            train_data = LOBDataset("data/raw/FI2010_train.csv")
            train_data.save_processed()
            
            # Later runs: load from disk in milliseconds
            train_data = LOBDataset.load_processed()
        
        Args:
            processed_dir (str): Folder containing X.npy, Y.npy, mu.npy, sigma.npy
            window_size (int): Must match the value used when save_processed() was called
        
        Returns:
            A fully initialized LOBDataset (no CSV touched)
        """
        print(f"⚡ Loading preprocessed data from {processed_dir}/...")
        
        required = ["X.npy", "Y.npy", "mu.npy", "sigma.npy"]
        for fname in required:
            fpath = os.path.join(processed_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Missing {fpath}. Run LOBDataset(...).save_processed() first."
                )
        
        # Bypass __init__ entirely — build a hollow instance and populate manually
        obj = cls.__new__(cls)
        obj.X           = np.load(os.path.join(processed_dir, "X.npy"))
        obj.Y           = np.load(os.path.join(processed_dir, "Y.npy"))
        obj.mu          = np.load(os.path.join(processed_dir, "mu.npy"))
        obj.sigma       = np.load(os.path.join(processed_dir, "sigma.npy"))
        obj.window_size = window_size

        # Validate cached files to catch stale/legacy processed data.
        if len(obj.X) != len(obj.Y):
            raise ValueError(
                f"Invalid processed cache: len(X)={len(obj.X)} != len(Y)={len(obj.Y)}."
            )
        if len(obj.X) <= window_size:
            raise ValueError(
                f"Invalid processed cache: len(X)={len(obj.X)} must be > window_size={window_size}."
            )

        unique_Y = np.unique(obj.Y)
        valid_labels = np.all((obj.Y >= 0) & (obj.Y < 3))
        if not valid_labels:
            raise ValueError(
                "Invalid processed labels found in Y.npy. "
                f"Expected labels in {{0,1,2}}, got unique labels: {unique_Y.tolist()}. "
                "This usually means data/processed was generated with an older label mapping."
            )
        
        total_windows = len(obj.X) - window_size
        print(f"✅ Loaded from disk! Rows: {len(obj.X):,} | Windows: {total_windows:,}")
        return obj

# ── Smoke Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    file_path = "data/raw/FI2010_train.csv" 
    import os
    if not os.path.exists(file_path):
        file_path = "../data/raw/FI2010_train.csv"
    
    PROCESSED_DIR = "data/processed"
    
    # ── Build from CSV (first run) or load from disk (subsequent runs) ────────
    if all(os.path.exists(os.path.join(PROCESSED_DIR, f))
           for f in ["X.npy", "Y.npy", "mu.npy", "sigma.npy"]):
        # Fast path: load from disk; if cache is stale, rebuild from CSV.
        try:
            train_data = LOBDataset.load_processed(PROCESSED_DIR, window_size=100)
        except ValueError as e:
            print(f"⚠️  Processed cache invalid: {e}")
            print("🔄 Rebuilding data/processed from raw CSV...")
            for fname in ["X.npy", "Y.npy", "mu.npy", "sigma.npy"]:
                fpath = os.path.join(PROCESSED_DIR, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
            train_data = LOBDataset(file_path, window_size=100, nrows=50_000)
            train_data.save_processed(PROCESSED_DIR)
    else:
        # Slow path: build from CSV and save
        train_data = LOBDataset(file_path, window_size=100, nrows=50_000)
        train_data.save_processed(PROCESSED_DIR)
    
    # ── (Optional) Val/test: load training stats from disk, apply to test CSV ────
    # mu    = np.load("data/processed/mu.npy")
    # sigma = np.load("data/processed/sigma.npy")
    # val_data = LOBDataset("data/raw/FI2010_test.csv", mu=mu, sigma=sigma)
    
    # Create the DataLoader
    # SHUFFLE MUST BE FALSE for time-series to preserve temporal order
    dataloader = DataLoader(train_data, batch_size=32, shuffle=False, num_workers=0)
    
    # --- SANITY CHECK ---
    for x_batch, y_batch in dataloader:
        print("\n--- SANITY CHECK ---")
        print(f"X (Features) Shape: {x_batch.shape}")        # expect (32, 1, 100, 40)
        print(f"Y (Labels) Shape:   {y_batch.shape}")        # expect (32,)
        print(f"X mean (≈ 0):       {x_batch.mean():.4f}")   # check normalization
        print(f"X std (≈ 1):        {x_batch.std():.4f}")    # check normalization
        print(f"First 5 labels:     {y_batch[:5].tolist()}")
        assert x_batch.shape == (32, 1, 100, 40), "❌ Wrong X shape!"
        assert y_batch.dtype == torch.long,        "❌ Y must be torch.long!"
        print("✅ All assertions passed.")
        break