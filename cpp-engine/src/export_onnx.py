"""
src/export_onnx.py
──────────────────
Exports a trained LOB model (DeepLOB or TransformerLOB) to ONNX format.

Usage (run from project root):
    python src/export_onnx.py

Outputs:
    <model_type>.onnx   — ONNX graph ready for C++ / TensorRT / ONNX Runtime

Dependencies:
    pip install onnx onnxruntime      (onnxruntime is optional — used for verification)

Expected project layout:
    src/models.py             <- DeepLOB, TransformerLOB
    *_checkpoint.pt           <- saved by train.py
    src/export_onnx.py        <- this file
"""

import torch
import torch.nn as nn
import os

from models import DeepLOB, TransformerLOB


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CKPT_PATH   = os.getenv("CKPT_PATH", "")
ONNX_PATH   = os.getenv("ONNX_PATH", "")
NUM_CLASSES = 3
OPSET       = 18


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


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD MODEL
#    Always export on CPU — ONNX graphs are device-agnostic. The C++ runtime
#    (ONNXRuntime / TensorRT) decides at inference time whether to use GPU.
# ══════════════════════════════════════════════════════════════════════════════
def _detect_model_type(ckpt: dict) -> str:
    model_type = str(ckpt.get("model_type", "")).lower().strip()
    if model_type in {"deeplob", "transformer"}:
        return model_type

    state_dict = ckpt.get("model_state_dict", {})
    if any(k.startswith("transformer_encoder.") for k in state_dict.keys()):
        return "transformer"
    return "deeplob"


def load_model(ckpt_path: str) -> tuple[nn.Module, str]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
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

    # eval() is mandatory before export:
    #   - disables Dropout (stochastic nodes break ONNX determinism)
    #   - switches BatchNorm to use fixed running mean/variance
    #     instead of per-batch statistics
    model.eval()

    saved_epoch = ckpt.get("epoch", "unknown")
    print(f"  Checkpoint    : {ckpt_path}  (epoch {saved_epoch})")
    print(f"  Model type    : {model_type}")
    print(f"  Device        : CPU  (ONNX export is always CPU-side)")
    return model, model_type


# ══════════════════════════════════════════════════════════════════════════════
# 2. DUMMY INPUT
#    torch.onnx.export traces the model by running a real forward pass with
#    this tensor. Shape must match exactly what the model expects at inference.
#
#    (1, 1, 100, 40)
#     │  │   │   └─ 40 LOB features
#     │  │   └───── 100 time steps
#     │  └─────────  1 input channel
#     └────────────  batch size = 1 for tracing (made dynamic below)
# ══════════════════════════════════════════════════════════════════════════════
def make_dummy_input() -> torch.Tensor:
    dummy = torch.randn(1, 1, 100, 40, dtype=torch.float32)
    print(f"  Dummy input   : {tuple(dummy.shape)}  dtype={dummy.dtype}")
    return dummy


# ══════════════════════════════════════════════════════════════════════════════
# 3. EXPORT TO ONNX
# ══════════════════════════════════════════════════════════════════════════════
def export_onnx(model: nn.Module, dummy: torch.Tensor, onnx_path: str) -> None:
    try:
        torch.onnx.export(
            model,
            dummy,
            onnx_path,

        # ── Embed all learned weights into the .onnx file ─────────────────────
        # Set to False only if you want to store weights separately (rare).
        export_params=True,

        # ── Operator set version ──────────────────────────────────────────────
        # opset 14 supports all ops used by Conv2d, LSTM, BatchNorm, and Linear.
        # Higher opsets add new ops but reduce runtime compatibility.
        opset_version=OPSET,

        # ── Fold constants at export time ─────────────────────────────────────
        # Pre-computes any sub-graphs whose inputs are all constants
        # (e.g. weight reshapes). Reduces runtime graph size and latency.
        do_constant_folding=True,

        # ── Tensor names exposed to the C++ runtime ───────────────────────────
        # These become the keys you use in C++ to bind input/output buffers:
        #   session.Run({{"input", input_tensor}}, {{"output"}})
        input_names=["input"],
        output_names=["output"],

        # ── Dynamic axes ──────────────────────────────────────────────────────
        # By default ONNX bakes the traced batch size (1) as a hard constant.
        # Marking axis 0 as dynamic lets the C++ engine pass any batch size
        # at runtime without re-exporting.
        #   axis 0 of "input"  → symbolic name "batch_size"  (can be 1, 32, …)
        #   axis 0 of "output" → same symbolic name
            dynamic_axes={
                "input":  {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
    except ModuleNotFoundError as e:
        if "onnxscript" in str(e):
            raise ModuleNotFoundError(
                "Missing dependency 'onnxscript' required by torch.onnx.export. "
                "Install with: pip install onnxscript"
            ) from e
        raise
    print(f"  ONNX file     : {onnx_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. VERIFICATION  (requires: pip install onnx onnxruntime)
#    Catches silent shape / type errors before the model reaches C++.
# ══════════════════════════════════════════════════════════════════════════════
def verify_onnx(onnx_path: str, dummy: torch.Tensor, model: nn.Module) -> None:
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np

        # ── Graph-level check (no runtime needed) ─────────────────────────────
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("  ONNX check    : ✅ graph is valid")

        # ── Numerical check — compare PyTorch vs ONNX Runtime outputs ─────────
        # Load the exported graph into ONNXRuntime
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        # Run inference with the same dummy tensor
        ort_inputs  = {"input": dummy.numpy()}
        ort_outputs = sess.run(["output"], ort_inputs)

        # Compare against the in-memory PyTorch model used for export.
        ref_model = model
        ref_model.eval()
        with torch.no_grad():
            pt_output = ref_model(dummy).numpy()

        max_diff = np.abs(pt_output - ort_outputs[0]).max()
        print(f"  Numerical diff: max |PyTorch − ONNX| = {max_diff:.2e}  ", end="")
        print("✅ OK" if max_diff < 1e-4 else "⚠️  larger than expected — check ops")

    except ImportError:
        print("  Verification  : skipped  (run: pip install onnx onnxruntime)")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  LOB Model → ONNX Export")
    print("=" * 60 + "\n")

    ckpt_path = _resolve_ckpt_path()
    model, model_type = load_model(ckpt_path)
    dummy = make_dummy_input()

    onnx_path = ONNX_PATH or f"{model_type}.onnx"

    print(f"\n  Exporting to ONNX  (opset {OPSET}) ...")
    export_onnx(model, dummy, onnx_path)

    print("\n  Verifying exported graph ...")
    verify_onnx(onnx_path, dummy, model)

    print(f"\n{'='*60}")
    print(f"  ✅  Export complete!")
    print(f"  File : {onnx_path}")
    print(f"  Input  shape : (batch_size, 1, 100, 40)   [dynamic batch]")
    print(f"  Output shape : (batch_size, {NUM_CLASSES})              [dynamic batch]")
    print(f"{'='*60}")
    print("\nC++ usage (ONNXRuntime):")
    print(f"  Ort::Session session(env, L\"{onnx_path}\", session_options);")
    print("  // bind 'input' tensor of shape {B, 1, 100, 40}")
    print("  // read  'output' tensor of shape {B, 3}")
