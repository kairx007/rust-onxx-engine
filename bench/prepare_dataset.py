"""Reconstruct a raw-scale FI-2010 test CSV for the Rust pipeline.

The repository ships the FI-2010 *test* split only as z-scored arrays
(`data/normalization/FI2010_test_test_n50000_w100_z.npz`, normalized with the
training statistics in `mu.npy` / `sigma.npy`). The original raw CSV is not in
the repository. To exercise the complete Rust pipeline (CSV -> z-score ->
windows -> tensor) we invert the z-score:

    raw = z * sigma + mu        (float32)

and write 40 feature columns followed by the FI-2010 label (1/2/3). The Rust
pipeline then re-applies (raw - mu) / (sigma + 1e-8). The round-trip error is
measured and recorded in the manifest; it is float32 rounding only.

Usage: bench/.venv/bin/python bench/prepare_dataset.py --out-dir bench/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TEST_NPZ = REPO / "data/normalization/FI2010_test_test_n50000_w100_z.npz"
MU = REPO / "data/normalization/mu.npy"
SIGMA = REPO / "data/normalization/sigma.npy"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO / "bench/data")
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "FI2010_test_reconstructed.csv"
    manifest_path = out_dir / "FI2010_test_reconstructed.json"
    if csv_path.exists():
        print(f"exists, leaving untouched: {csv_path}")
        return

    archive = np.load(TEST_NPZ)
    z = archive["X"].astype(np.float32)
    y = archive["Y"].astype(np.int64)
    mu = np.load(MU).astype(np.float32)
    sigma = np.load(SIGMA).astype(np.float32)
    assert z.shape[1] == 40 and mu.shape == (40,) and sigma.shape == (40,)
    assert set(np.unique(y)) <= {0, 1, 2}

    raw = (z * sigma + mu).astype(np.float32)
    # What the Rust pipeline will compute, in float32.
    renorm = ((raw - mu) / (sigma + np.float32(1e-8))).astype(np.float32)
    err = np.abs(renorm - z)

    with csv_path.open("w", newline="") as handle:
        for row, label in zip(raw, y):
            handle.write(",".join(f"{v:.9g}" for v in row))
            handle.write(f",{int(label) + 1}\n")

    manifest = {
        "description": "Raw-scale FI-2010 test split reconstructed from z-scored arrays (raw = z*sigma + mu)",
        "source_npz": str(TEST_NPZ.relative_to(REPO)),
        "source_npz_sha256": sha256(TEST_NPZ),
        "mu_sha256": sha256(MU),
        "sigma_sha256": sha256(SIGMA),
        "rows": int(z.shape[0]),
        "columns": "0-39 features, 40 = FI-2010 label (1=Down, 2=Stationary, 3=Up)",
        "label_counts_internal_0_1_2": np.bincount(y, minlength=3).tolist(),
        "complete_100_row_windows": int(z.shape[0] // 100),
        "roundtrip_max_abs_error": float(err.max()),
        "roundtrip_mean_abs_error": float(err.mean()),
        "float_format": "%.9g (round-trips float32 exactly)",
        "csv_sha256": sha256(csv_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
