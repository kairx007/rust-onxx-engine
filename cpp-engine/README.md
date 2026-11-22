# DeepLOB — Limit Order Book Mid-Price Forecasting

A PyTorch-based deep learning pipeline for forecasting mid-price movements in high-frequency trading using Limit Order Book (LOB) data. Combines convolutional and temporal architectures with rigorous data preprocessing, FI-2010 benchmark split validation, and ONNX export.

## Phase 2: Data Engineering & Preprocessing ✅

### Implemented Features

**`dataset.py` — LOBDataset Class**
- ✅ **Sliding Window Extraction**: T=100 timesteps per sample (configurable)
- ✅ **Z-Score Normalization**: $X_{norm} = \frac{X - \mu}{\sigma + \epsilon}$
  - Computed on training data only (no look-ahead bias)
  - Passed to validation/test datasets to prevent data leakage
- ✅ **Label Extraction**: Fixed label extraction at column 144 (project FI-2010 layout)
  - Remapped: 1/2/3 → 0/1/2 (Down/Stationary/Up)
- ✅ **Data Caching**: 
  - `save_processed()` — writes normalized X, Y, µ, σ to `data/processed/*.npy`
  - `load_processed()` — instant reconstruction from disk (skips CSV parsing)
- ✅ **DataLoader**: Configured with `shuffle=False` for time-series integrity

### Data Preprocessing Pipeline

```
Raw CSV → Load & Clean → Extract Features (cols 0-39) & Labels (col 144)
    ↓
Remap Labels (1/2/3 → 0/1/2) → Compute µ, σ on training split
    ↓
Normalize Features: X_norm = (X - µ) / (σ + ε)
    ↓
Create Sliding Windows (T=100) → Save to data/processed/
    ↓
DataLoader (batch_size=32, shuffle=False) → Ready for training
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data (First Run)

```bash
python src/dataset.py
```

**What happens:**
- Loads `data/raw/FI2010_train.csv` (50,000 rows by default)
- Applies Z-Score normalization
- Saves to `data/processed/{X, Y, mu, sigma}.npy`
- Prints validation checks (X shape, X mean ≈ 0, X std ≈ 1)

**Subsequent runs:**
- Auto-detects `.npy` files and loads from disk (milliseconds)
- Falls back to CSV if files are missing or stale/invalid

### 3. Train a Model

```bash
python src/train.py
```

**Features:**
- FI-2010 benchmark protocol: train 1-7, validate 8, test 9-10
- Class-weighted loss for imbalanced labels
- OneCycleLR learning rate scheduler
- Saves checkpoint with model-specific name:
  - `deeplob_checkpoint.pt`
  - `transformer_checkpoint.pt`

### 4. Other Utilities

```bash
# Evaluate on test set
python src/evaluate.py

# Export to ONNX
python src/export_onnx.py

# Visualize results
python src/visualize.py
```

Optional checkpoint override:

```powershell
$env:CKPT_PATH='transformer_checkpoint.pt'; python src/evaluate.py
$env:CKPT_PATH='transformer_checkpoint.pt'; python src/export_onnx.py
```

## Phase 3: Model Architecture & Training ✅

### Implemented Models

**DeepLOB (CNN + LSTM Baseline)** ✅
- CNN Block: 3 Conv2d layers (1×3 kernels) → 32 channels, preserves T=100 time dimension
- LSTM Block: 64 hidden units, 1 layer, GRU-like temporal processing
- Fully-Connected Head: 3-class classifier (Down/Stationary/Up)
- **Trainable params**: ~66K

**TransformerLOB (CNN + Transformer)** ✅ **NEW**
- CNN Block: Same as DeepLOB (spatial LOB feature extraction)
- PositionalEncoding: Sinusoidal, injects absolute time positions (causal)
- Input projection: 32 → 64 before Transformer
- TransformerEncoder: 2 layers, 4 attention heads, d_model=64, 128-dim feed-forward MLP
- Fully-Connected Head: 3-class classifier
- **Trainable params**: ~114K

### Training Pipeline

✅ **FI-2010 Benchmark Validation**
- Train on: `FI2010_train.csv` (days 1-7)
- Validate on: first part of `FI2010_test.csv` (day 8)
- Test on: remaining part of `FI2010_test.csv` (days 9-10)
- No shuffling (time-series integrity preserved)

✅ **OneCycleLR Scheduler** (Adaptive Learning Rate)
- Warmup: 30% of training
- Decay: 70% of training (cosine annealing back to near-zero)
- Improves convergence speed and final accuracy
- Updates **per batch** (not per epoch)

✅ **Class-Weighted CrossEntropyLoss**
- Rebalances severe dataset imbalance (Stationary >> Up/Down)
- Penalty weights: Stationary < Down/Up
- Forces model to learn minority movements

✅ **Gradient Clipping**
- Prevents exploding gradients (critical for LSTM)
- Max norm: 1.0

### Training Configuration

```python
TRAIN_CSV_PATH = "data/raw/FI2010_train.csv"
TEST_CSV_PATH  = "data/raw/FI2010_test.csv"
MAX_TRAIN_ROWS = 200_000
MAX_TEST_ROWS  = None
BATCH_SIZE  = 32
EPOCHS      = 15
LR          = 1e-3
MODEL_TYPE  = "transformer"   # default (can override via env var)
```

### How to Train

**DeepLOB baseline:**
```bash
MODEL_TYPE=deeplob python src/train.py
```

On PowerShell:
```powershell
$env:MODEL_TYPE='deeplob'; python src/train.py
```

**TransformerLOB (default):**
```bash
python src/train.py
```

**Training output (per epoch):**
```
[Epoch 3/15] step 50/312  | loss: 0.8234
[Epoch 3/15] step 100/312 | loss: 0.7891
...
──────────────────────────────────────────────────────────────────────
Epoch 3/15 | train loss: 0.8012 | train acc: 65.23% 
          | val loss: 0.8145   | val acc: 64.81% | time: 12.3s
──────────────────────────────────────────────────────────────────────
```

### Model Comparison

| Aspect | DeepLOB | TransformerLOB |
|--------|---------|----------------|
| Temporal Module | LSTM (64 hidden, 1 layer) | Transformer (2 layers, 4 heads) |
| Sequence Processing | Sequential (gating) | Parallel (self-attention) |
| Params | ~66K | ~114K |
| Training Speed | Medium | Fast (parallelizable) |
| Long-range context | Good (hidden state) | Excellent (all-to-all attention) |
| Recent data bias | Slight (RNN state) | None (attention weights) |

### Checkpoint Format

Saved to `<model_type>_checkpoint.pt`:
```python
{
    'epoch': 15,
    'model_state_dict': {...},      # Model weights
    'optimizer_state_dict': {...},  # Optimizer state (for resuming)
    'model_type': 'deeplob' or 'transformer',
}
```

Evaluation and ONNX export scripts auto-detect model type from checkpoint metadata/keys and can resolve checkpoints automatically.

## Phase 4: Systems Engineering & C++ Inference ✅

### Deliverables

- `cpp_inference/CMakeLists.txt`:
  - CMake build for C++17 binaries
  - links ONNX Runtime
  - copies `onnxruntime.dll` beside binaries on Windows when available
- `cpp_inference/inference.cpp`:
  - loads ONNX model
  - runs single-sample inference for shape `[1, 1, 100, 40]`
  - prints logits + top-1 class (`0=down, 1=stationary, 2=up`)
- `cpp_inference/benchmark.cpp`:
  - warm-up: 100 runs
  - timed: 1000 runs (configurable)
  - reports mean/std/p99 latency in ms
  - checks target `< 1 ms mean`
- `src/benchmark_python.py`:
  - Python ONNX Runtime benchmark with same warm-up/timed protocol
  - for direct Python vs C++ comparison in Phase 5 report

### Build C++ Inference (Windows + PowerShell)

1. Download ONNX Runtime C++ SDK locally:

```powershell
./cpp_inference/setup_onnxruntime.ps1
```

2. Configure/build using local venv CMake and MinGW generator:

```powershell
$env:ONNXRUNTIME_ROOT='C:/path/to/onnxruntime-win-x64-<version>'
& '.venv/Scripts/cmake.exe' -S cpp_inference -B cpp_inference/build-mingw -G 'MinGW Makefiles' -DONNXRUNTIME_ROOT="$env:ONNXRUNTIME_ROOT"
& '.venv/Scripts/cmake.exe' --build cpp_inference/build-mingw -j
```

### Run C++ Inference

```powershell
./cpp_inference/build-mingw/inference.exe
```

If needed, pass model path explicitly:

```powershell
./cpp_inference/build-mingw/inference.exe transformer.onnx
```

### Run C++ Latency Benchmark

```powershell
./cpp_inference/build-mingw/benchmark.exe
```

Arguments:
- arg1: model path (auto-detects `transformer.onnx` then `deeplob.onnx` if omitted)
- arg2: warm-up runs (default `100`)
- arg3: timed runs (default `1000`)

### Run Full-Dataset C++ Inference From CSV

```powershell
./cpp_inference/build-mingw/full_inference.exe
```

Optional arguments:

```powershell
./cpp_inference/build-mingw/full_inference.exe transformer.onnx data/raw/FI2010_test.csv data/processed/mu.npy data/processed/sigma.npy 1 -1 100
```

Arguments:
- arg1: model path (auto-detect if omitted)
- arg2: CSV path (default `data/raw/FI2010_test.csv`)
- arg3: `mu.npy` path (default `data/processed/mu.npy`)
- arg4: `sigma.npy` path (default `data/processed/sigma.npy`)
- arg5: batch size (default `1`; current ONNX export is safest with batch=1 in C++)
- arg6: max windows (`-1` means all)
- arg7: warm-up runs (default `100`)

### Run Python Latency Benchmark

```bash
python src/benchmark_python.py --warmup 100 --runs 1000
```

### Phase 5 Comparison Template

Collect and report:
- Python mean ± std, p99 (ms)
- C++ mean ± std, p99 (ms)
- speedup = Python mean / C++ mean
- target check: C++ mean < 1.0 ms

### One-Command Phase 5 Report (Recommended)

Run the automated report generator:

```bash
python src/generate_phase5_report.py
```

This script will:
- run Python ONNX benchmark (`src/benchmark_python.py`)
- run C++ benchmark (`cpp_inference/build-mingw/benchmark.exe`)
- run full CSV C++ inference (`cpp_inference/build-mingw/full_inference.exe`)
- write a consolidated markdown report to `logs/phase5_report.md`
- save raw command outputs to:
  - `logs/phase5_python_output.txt`
  - `logs/phase5_cpp_benchmark_output.txt`
  - `logs/phase5_cpp_full_output.txt`

Useful options:

```bash
python src/generate_phase5_report.py \
  --model transformer.onnx \
  --py-warmup 100 --py-runs 1000 \
  --cpp-warmup 100 --cpp-runs 1000 \
  --full-batch-size 1 --full-max-windows -1 --full-warmup 20 \
  --out logs/phase5_report.md
```

If your C++ binaries are in a custom location:

```bash
python src/generate_phase5_report.py \
  --cpp-benchmark cpp_inference/build-mingw/benchmark.exe \
  --cpp-full cpp_inference/build-mingw/full_inference.exe
```

## Project Structure

```
├── data/
│   ├── raw/                    # FI-2010 CSV files
│   │   ├── FI2010_train.csv
│   │   └── FI2010_test.csv
│   └── processed/              # Cached normalized data (.npy)
│       ├── X.npy              # Features (N, 40), float32
│       ├── Y.npy              # Labels (N,), int64
│       ├── mu.npy             # Means (40,), float32
│       └── sigma.npy          # Stds (40,), float32
│
├── notebooks/
│   └── 01_visualize_data.ipynb
│
├── src/
│   ├── dataset.py              # LOBDataset + preprocessing
│   ├── models.py               # DeepLOB + TransformerLOB models
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Inference
│   ├── export_onnx.py          # ONNX export
│   ├── benchmark_python.py     # Python latency benchmark
│   ├── generate_phase5_report.py # Automated Python vs C++ report
│   ├── inspect_data.py         # EDA utilities
│   └── visualize.py            # Plotting
│
├── deeplob_checkpoint.pt       # DeepLOB checkpoint (optional)
├── transformer_checkpoint.pt   # Transformer checkpoint (optional)
├── deeplob.onnx                # DeepLOB ONNX export (optional)
├── transformer.onnx            # Transformer ONNX export (optional)
├── README.md
└── requirements.txt
```

## Data Format

**Project FI-2010 Layout** (after dropping index column):
- **Columns 0–39** (40 features): 10 LOB levels × (bid/ask price + volume)
- **Column 144** (label): 1/2/3 remapped to 0/1/2 for training

## Known Constraints

- Single market/day (FI-2010 from 2010)
- Offline training (static data)
- k=10 horizon only (1 second ahead)
- No transaction costs

## References

- **FI-2010**: [LOB Reconstruction Classification & Prediction Dataset](https://etsin.fairdata.fi/dataset/73eb48d7-4ddc-4c10-9f51-264ab4522fef)
- **DeepLOB**: Tsantekidis et al. on spatio-temporal LOB modeling

## License

Provided as-is for research and educational purposes.