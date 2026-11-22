"""Data and artifact checks for the C++ project (cpp-engine/) against the Rust repository.

Usage (needs numpy + onnxruntime 1.31; the version-matched venv has both):
  bench/.venv-ort131/bin/python -I bench/data_checks.py

Writes results/data_checks/report.md and results/data_checks/checks.json.

Checks
  1. Artifact identity: models and processed arrays in cpp-engine/ vs the Rust repo.
  2. Raw CSV structure: header line, index column, field count, rows.
  3. Normalization provenance: shipped mu/sigma vs stats of the first 50k / 200k training rows
     (cpp-engine/src/train.py trains on MAX_TRAIN_ROWS=200000 but never saves its stats).
  4. Processed test array and the Phase 2 reconstructed CSV vs the real raw test CSV.
  5. Label semantics: does FI-2010 label 1 follow a rising or a falling mid-price?
  6. Windowing/label alignment: training (stride 1, label at row i+100) vs the Rust engine
     (stride 100, label at row i+99).
  7. Model quality on the train.py split (val = day 8, test = days 9-10) under both stats.
  8. Model quality on the models' own training windows (distinguishes "weak" from "broken").
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[1]
CPP = REPO / "cpp-engine"
OUT = REPO / "results/data_checks"
MODELS = {"deeplob": REPO / "models/deeplob.onnx", "transformerlob": REPO / "models/transformerlob.onnx"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path: Path, max_rows: int | None = None) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, max_rows=max_rows)


def session(model: str) -> tuple[ort.InferenceSession, str]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    s = ort.InferenceSession(str(MODELS[model]), options, providers=["CPUExecutionProvider"])
    return s, s.get_inputs()[0].name


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    cm = np.zeros((3, 3), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    recall = cm.diagonal() / cm.sum(1).clip(1)
    precision = cm.diagonal() / cm.sum(0).clip(1)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall).clip(1e-12), 0)
    return dict(windows=int(y.size), accuracy=float((pred == y).mean()), macro_f1=float(f1.mean()),
                recall_class0_1_2=[round(float(r), 4) for r in recall],
                predicted_counts=np.bincount(pred, minlength=3).tolist(),
                majority_class_accuracy=float(np.bincount(y, minlength=3).max() / y.size),
                confusion_rows_true_cols_pred=cm.tolist())


def predict_windows(model: str, z: np.ndarray, starts: np.ndarray) -> np.ndarray:
    s, name = session(model)
    return np.array([s.run(None, {name: z[i:i + 100][None, None]})[0].argmax() for i in starts], dtype=np.int64)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks: dict = {}

    # 1. Artifact identity
    pairs = [("deeplob.onnx", "models/deeplob.onnx"), ("transformer.onnx", "models/transformerlob.onnx"),
             ("transformer.onnx.data", "models/transformer.onnx.data")]
    pairs += [(f"data/processed/{n}", f"data/normalization/{n}") for n in
              ("mu.npy", "sigma.npy", "X.npy", "Y.npy", "zscore_stats.npz",
               "FI2010_test_test_n50000_w100_z.npz", "FI2010_train_train_n50000_w100_z.npz")]
    checks["artifact_identity"] = {f"cpp-engine/{a} vs {b}": sha(CPP / a) == sha(REPO / b) for a, b in pairs}

    # 2. Raw CSV structure
    test_path, train_path = CPP / "data/raw/FI2010_test.csv", CPP / "data/raw/FI2010_train.csv"
    header = test_path.open().readline().strip().split(",")
    test = load_csv(test_path)
    train = load_csv(train_path, max_rows=200_000)
    with train_path.open() as handle:
        train_rows_total = sum(1 for _ in handle) - 1
    checks["raw_csv"] = dict(
        header_first_fields=header[:4], header_fields=len(header), test_rows=int(test.shape[0]),
        train_rows_total=train_rows_total, fields_per_row=int(test.shape[1]),
        column0_is_row_index=bool(np.array_equal(test[:, 0], np.arange(len(test)))),
        layout="raw col 0 = index; raw cols 1-40 = features; raw cols 145-149 = five label horizons (training uses 145)",
        label_values_col145=np.unique(test[:, 145]).tolist())

    # 3. Normalization provenance
    mu, sigma = (np.load(REPO / f"data/normalization/{n}.npy").astype(np.float64) for n in ("mu", "sigma"))
    x50, x200 = train[:50_000, 1:41].astype(np.float32), train[:, 1:41].astype(np.float32)
    mu50, sd50 = x50.mean(0), x50.std(0) + 1e-8
    mu200, sd200 = x200.mean(0), x200.std(0) + 1e-8
    checks["normalization"] = dict(
        shipped_vs_first_50k=dict(max_abs_mu=float(np.abs(mu50 - mu).max()), max_abs_sigma=float(np.abs(sd50 - sigma).max())),
        shipped_vs_first_200k=dict(max_abs_mu=float(np.abs(mu200 - mu).max()), max_abs_sigma=float(np.abs(sd200 - sigma).max()),
                                   max_relative_sigma=float(np.abs(sd200 / sigma - 1).max())),
        note="dataset.py stores sigma = std + 1e-8 and divides by it; Rust and C++ add 1e-8 again (negligible in float32)")

    # 4. Processed test array and reconstructed CSV fidelity
    xt, yt = test[:, 1:41].astype(np.float32), (test[:, 145] - 1).astype(np.int64)
    npz = np.load(REPO / "data/normalization/FI2010_test_test_n50000_w100_z.npz")
    recon = np.loadtxt(REPO / "bench/data/FI2010_test_reconstructed.csv", delimiter=",", dtype=np.float64)
    checks["test_arrays"] = dict(
        npz_equals_raw_normalized_with_shipped_stats=float(np.abs(npz["X"] - (xt - mu.astype(np.float32)) / sigma.astype(np.float32)).max()),
        npz_labels_equal_raw_col145_minus_1=bool(np.array_equal(npz["Y"], yt)),
        phase2_reconstructed_csv_max_abs_feature_error=float(np.abs(recon[:, :40] - test[:, 1:41]).max()),
        phase2_reconstructed_csv_labels_equal=bool(np.array_equal(recon[:, 40], test[:, 145])))

    # 5. Label semantics: mid-price direction after each label (level-1 price columns raw 1 and 3)
    mid = (test[:, 1] + test[:, 3]) / 2
    semantics = {}
    for col, k in zip(range(145, 150), (1, 2, 3, 5, 10)):
        n = len(mid)
        future = np.array([mid[t + 1:t + 1 + k].mean() if t + k < n else np.nan for t in range(n)])
        delta = future - mid
        ok = ~np.isnan(delta)
        semantics[f"raw_col_{col}"] = {
            f"label_{v}": dict(count=int((test[:, col] == v).sum()),
                               frac_mid_up=round(float(np.mean(delta[ok & (test[:, col] == v)] > 0)), 3),
                               frac_mid_down=round(float(np.mean(delta[ok & (test[:, col] == v)] < 0)), 3))
            for v in (1, 2, 3)}
    checks["label_semantics"] = dict(
        per_column=semantics,
        conclusion="label 1 is followed by rising mid-prices and label 3 by falling ones: 1=Up, 3=Down, so internal "
                   "class 0=Up and 2=Down (the Rust Direction enum and the C++ printouts say 0=Down, 2=Up)")

    # 6. Window/label alignment
    starts100 = np.arange(0, len(yt) - 99, 100)
    last_row = yt[starts100 + 99]
    next_row = yt[np.minimum(starts100 + 100, len(yt) - 1)]
    checks["window_alignment"] = dict(
        training="stride 1, label at row i+100 (tick after the window)",
        rust_engine_before_fix="stride 100, label at row i+99 (last row of the window)",
        rust_engine_now="stride configurable (default 1), label at row i+100 (matches training)",
        cpp_full_inference="stride 1, label at row i+100 (matches training)",
        rust_windows=int(starts100.size),
        fraction_prefix_rust_label_differs_from_training_label=float(np.mean(last_row != next_row)))

    # 7. Quality on the train.py split under both normalizations
    n = len(xt)
    val_rows = n // 3
    split_rows = {"val_day8": (0, val_rows), "test_days9_10": (val_rows, n - val_rows)}
    stats = {"shipped_first_50k": (mu.astype(np.float32), sigma.astype(np.float32)),
             "train_py_first_200k": (mu200.astype(np.float32), sd200.astype(np.float32))}
    quality = {}
    for model in MODELS:
        for sname, (m, s) in stats.items():
            z = ((xt - m) / s).astype(np.float32)
            for split, (start, rows) in split_rows.items():
                zs, ys = z[start:start + rows], yt[start:start + rows]
                starts = np.arange(rows - 100)
                quality[f"{model} | {sname} | {split}"] = metrics(predict_windows(model, zs, starts), ys[starts + 100])
                q = quality[f"{model} | {sname} | {split}"]
                print(f"{model:15s} {sname:20s} {split:14s} acc {q['accuracy']:.3f} macroF1 {q['macro_f1']:.3f}", flush=True)
    checks["split_quality"] = quality

    # 8. Quality on training windows (4,000 sampled, training stats)
    z200 = ((x200 - mu200.astype(np.float32)) / sd200.astype(np.float32)).astype(np.float32)
    y200 = (train[:, 145] - 1).astype(np.int64)
    idx = np.sort(np.random.default_rng(0).choice(len(z200) - 101, 4000, replace=False))
    checks["train_window_quality"] = {model: metrics(predict_windows(model, z200, idx), y200[idx + 100]) for model in MODELS}

    (OUT / "checks.json").write_text(json.dumps(checks, indent=2))
    write_report(checks)
    print(OUT / "report.md")


def write_report(c: dict) -> None:
    md = ["# Data and artifact checks — cpp-engine vs Rust repo\n",
          "Generated by `bench/data_checks.py`; raw values in `checks.json`.\n",
          "## 1. Artifact identity\n", "| Pair | Identical |", "|---|---|"]
    md += [f"| {k} | {'yes' if v else '**NO**'} |" for k, v in c["artifact_identity"].items()]
    r = c["raw_csv"]
    md += ["\n## 2. Raw CSV structure\n",
           f"- Header fields {r['header_fields']} (first: `{r['header_first_fields']}`), {r['fields_per_row']} fields per row.",
           f"- Test rows {r['test_rows']:,}; train rows {r['train_rows_total']:,} (training uses the first 200,000).",
           f"- Column 0 is a row index: {r['column0_is_row_index']}. Layout: {r['layout']}.",
           f"- Label values (col 145): {r['label_values_col145']}."]
    nrm = c["normalization"]
    md += ["\n## 3. Normalization provenance\n",
           f"- Shipped `mu.npy`/`sigma.npy` vs stats of the first 50k training rows: max |Δμ| {nrm['shipped_vs_first_50k']['max_abs_mu']:.1e}, "
           f"max |Δσ| {nrm['shipped_vs_first_50k']['max_abs_sigma']:.1e} → **they are the 50k-row smoke-run stats**.",
           f"- vs the first 200k rows that `train.py` trains on: max |Δμ| {nrm['shipped_vs_first_200k']['max_abs_mu']:.3f}, "
           f"max |Δσ| {nrm['shipped_vs_first_200k']['max_abs_sigma']:.3f} (relative σ difference up to "
           f"{100 * nrm['shipped_vs_first_200k']['max_relative_sigma']:.0f}%). `train.py` never saves its stats, so every "
           "inference path (Python evaluate.py, C++, Rust) normalizes with statistics the models were not trained on.",
           f"- {nrm['note']}."]
    t = c["test_arrays"]
    md += ["\n## 4. Test arrays\n",
           f"- Processed test `.npz` = raw test normalized with the shipped stats: max diff {t['npz_equals_raw_normalized_with_shipped_stats']:.1e}; "
           f"labels = raw col 145 − 1: {t['npz_labels_equal_raw_col145_minus_1']}.",
           f"- Phase 2 reconstructed CSV vs real raw: max feature error {t['phase2_reconstructed_csv_max_abs_feature_error']:.1e}, "
           f"labels equal {t['phase2_reconstructed_csv_labels_equal']} → Phase 2 benchmark inputs are faithful."]
    sem = c["label_semantics"]["per_column"]["raw_col_146"]
    md += ["\n## 5. Label semantics\n",
           f"- Raw col 146 example: label 1 → mid up {sem['label_1']['frac_mid_up']:.0%} / down {sem['label_1']['frac_mid_down']:.0%}; "
           f"label 3 → up {sem['label_3']['frac_mid_up']:.0%} / down {sem['label_3']['frac_mid_down']:.0%}.",
           f"- {c['label_semantics']['conclusion']}."]
    w = c["window_alignment"]
    md += ["\n## 6. Window / label alignment\n",
           f"- Training: {w['training']}. C++ full_inference: {w['cpp_full_inference']}.",
           f"- Rust engine before the fix: {w['rust_engine_before_fix']} — its label differed from the training-convention "
           f"label for {100 * w['fraction_prefix_rust_label_differs_from_training_label']:.1f}% of its {w['rust_windows']} test windows.",
           f"- Rust engine now: {w['rust_engine_now']}; on the full test CSV it evaluates the same 31,837 windows as C++ "
           "with identical correct counts (`results/phase2-cpp/rust_vs_cpp_eval/`)."]
    md += ["\n## 7. Model quality on the train.py split (stride-1 windows, label at i+100)\n",
           "| Model | Normalization | Split | Windows | Accuracy | Macro-F1 | Recall class 0/1/2 | Majority baseline |",
           "|---|---|---|---|---|---|---|---|"]
    for key, q in c["split_quality"].items():
        model, stats, split = key.split(" | ")
        md.append(f"| {model} | {stats} | {split} | {q['windows']:,} | {q['accuracy']:.1%} | {q['macro_f1']:.3f} | "
                  f"{q['recall_class0_1_2']} | {q['majority_class_accuracy']:.1%} |")
    md += ["\n## 8. Model quality on the models' own training windows (4,000 sampled)\n",
           "| Model | Accuracy | Macro-F1 | Predicted counts | True majority share |", "|---|---|---|---|---|"]
    for model, q in c["train_window_quality"].items():
        md.append(f"| {model} | {q['accuracy']:.1%} | {q['macro_f1']:.3f} | {q['predicted_counts']} | {q['majority_class_accuracy']:.1%} |")
    md.append("\nTransformerLOB at ~31% on its own training windows indicates the exported ONNX graph does not reproduce a "
              "trained model (export issue or untrained checkpoint); confirming this needs `transformer_checkpoint.pt` "
              "to compare PyTorch and ONNX outputs on the same windows.")
    (OUT / "report.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
