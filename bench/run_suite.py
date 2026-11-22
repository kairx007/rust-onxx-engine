"""Phase 2 benchmark suite orchestrator (process-isolated, randomized, interleaved).

Isolation protocol
  * Every measured trial is a fresh OS process (no shared allocator / ORT arena /
    thread-pool state between trials).
  * All (configuration, repetition) pairs are shuffled with a fixed seed and run
    interleaved, so slow drift (thermal, background load) is spread across
    configurations instead of biasing one of them.
  * A cool-down sleep separates processes; resource sampling is off.
  * The machine is kept awake with `caffeinate`; an environment snapshot and the
    load average before every run are recorded.

Usage:
  bench/.venv/bin/python bench/run_suite.py [--reps 10] [--quick]

Output: results/phase2/<run_id>/ (see bench/README.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "target/release/lob-inference"
# Same Rust code built with `--features ort-dynamic`; loads ONNX Runtime from ORT_DYLIB_PATH.
BIN_DYNAMIC = REPO / "target/ort-dynamic/release/lob-inference"
CONFIG = REPO / "bench/configs/phase2.toml"
PY = Path(sys.executable)
# Version-matched Python baseline: the PyPI onnxruntime 1.31.0 wheel (same ORT commit 402979a as the Rust build).
PY_ENGINE = REPO / "bench/.venv-ort131/bin/python"
# C++ engine adapter (bench/cpp) built against Microsoft's ONNX Runtime 1.31.0 C++ SDK (NuGet osx-arm64).
CPP_SDK = REPO / "cpp-engine/cpp_inference/third_party/onnxruntime-osx-arm64-1.31.0"
CPP_BIN = REPO / "bench/cpp/build/rustlob_cpp_bench"
MODEL_NAME = {"deeplob": "DeepLOB", "transformerlob": "TransformerLOB"}
MS_ORT_DYLIB = REPO / "bench/.venv-ort131/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.31.0.dylib"
PY_BENCH = REPO / "bench/python_ort_bench.py"
MODELS = {"deeplob": REPO / "models/deeplob.onnx", "transformerlob": REPO / "models/transformerlob.onnx"}


def sh(*args: str, check: bool = False) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=check)
        return result.stdout.strip() or None
    except Exception:
        return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment_snapshot() -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "macos": sh("sw_vers"),
        "cpu_brand": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "ncpu": sh("sysctl", "-n", "hw.ncpu"),
        "perf_levels": {
            "p_cores": sh("sysctl", "-n", "hw.perflevel0.physicalcpu"),
            "e_cores": sh("sysctl", "-n", "hw.perflevel1.physicalcpu"),
        },
        "memory_bytes": sh("sysctl", "-n", "hw.memsize"),
        "power": sh("pmset", "-g", "batt"),
        "thermal": sh("pmset", "-g", "therm"),
        "low_power_mode": sh("pmset", "-g", "custom"),
        "low_power_mode_active": low_power_mode_active(),
        "load_average": sh("sysctl", "-n", "vm.loadavg"),
        "top_processes_by_cpu": sh("ps", "-Ao", "pcpu,comm", "-r"),
        "git_commit": sh("git", "-C", str(REPO), "rev-parse", "HEAD"),
        "git_branch": sh("git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_files": sh("git", "-C", str(REPO), "status", "--porcelain"),
        "rustc": sh("rustc", "--version"),
        "cargo": sh("cargo", "--version"),
        "binary_sha256": sha256(BIN),
        "cpp_binary_sha256": sha256(CPP_BIN) if CPP_BIN.exists() else None,
        "cpp_ort_dylib_sha256": sha256(CPP_SDK / "lib/libonnxruntime.dylib") if (CPP_SDK / "lib/libonnxruntime.dylib").exists() else None,
        "ort_build_info": sh(str(BIN), "--config", str(CONFIG), "runtime-info"),
        "python": platform.python_version(),
        "python_onnxruntime": sh(str(PY), "-c", "import onnxruntime;print(onnxruntime.__version__)"),
        "python_numpy": sh(str(PY), "-c", "import numpy;print(numpy.__version__)"),
        "model_sha256": {name: sha256(path) for name, path in MODELS.items()},
    }


def low_power_mode_active() -> bool:
    text = sh("pmset", "-g") or ""
    return any(line.split()[:2] == ["lowpowermode", "1"] for line in text.splitlines())


def thread_args(threads: str) -> list[str]:
    return [] if threads == "default" else ["--intra-threads", threads]


ENABLED: set[str] = {"rust", "python", "cpp", "rust-coreml", "rust-msdylib", "python-ort132"}


def plan_runs(reps: int, warmup: int, iterations: int) -> list[dict]:
    runs = []
    # Main matrix: engine x model x threads x input policy.
    # rust   = bundled static ORT 1.31.0, device policy cpu (genuine CPU-only)
    # python = PyPI onnxruntime 1.31.0 wheel (version-matched baseline)
    for engine in ("rust", "python", "cpp"):
        for model in MODELS:
            for threads in ("1", "default"):
                for policy in ("first", "cycle"):
                    for rep in range(reps):
                        runs.append(dict(kind="inference", engine=engine, model=model, threads=threads,
                                         policy=policy, graph_opt="default", rep=rep,
                                         warmup=warmup, iterations=iterations))
    # Ablation: device policy. `rust-coreml` = the ort crate's implicit MaxEfficiency policy,
    # which offloads partitions to CoreML on Apple silicon (the pre-fix behaviour).
    # Ablation: binary. `rust-msdylib` = same Rust code, Microsoft's official ORT 1.31.0 dylib.
    for engine in ("rust-coreml", "rust-msdylib"):
        for model in MODELS:
            for threads in ("1", "default"):
                for rep in range(reps):
                    runs.append(dict(kind="inference", engine=engine, model=model, threads=threads, policy="cycle",
                                     graph_opt="default", rep=rep, warmup=warmup, iterations=iterations))
    # Ablation: ONNX Runtime graph optimization level (Rust, 1 thread, cycle).
    for model in MODELS:
        for graph_opt in ("disable", "all"):
            for rep in range(max(3, reps // 2)):
                runs.append(dict(kind="inference", engine="rust", model=model, threads="1", policy="cycle",
                                 graph_opt=graph_opt, rep=rep, warmup=warmup, iterations=iterations))
    # Cold start: process spawn -> first prediction -> exit.
    for engine in ("rust", "python", "cpp", "rust-coreml"):
        for model in MODELS:
            for rep in range(reps):
                runs.append(dict(kind="coldstart", engine=engine, model=model, threads="default", rep=rep))
    # Rust-only pipeline stages.
    for rep in range(max(3, reps // 2)):
        runs.append(dict(kind="preprocessing", engine="rust", model="deeplob", threads="default", rep=rep))
        for model in MODELS:
            runs.append(dict(kind="end-to-end", engine="rust", model=model, threads="default", rep=rep))
    return runs


def run_name(run: dict) -> str:
    parts = [run["kind"], run["engine"], run["model"], f"t{run['threads']}"]
    if run["kind"] == "inference":
        parts += [run["policy"], f"opt-{run['graph_opt']}"]
    return "__".join(parts) + f"__rep{run['rep']:02d}"


def rust_command(engine: str) -> tuple[list[str], dict]:
    """Binary, extra global flags, and environment for a Rust engine variant."""
    env = dict(os.environ)
    if engine == "rust-msdylib":
        env["ORT_DYLIB_PATH"] = str(MS_ORT_DYLIB)
        return [str(BIN_DYNAMIC), "--device-policy", "cpu"], env
    if engine == "rust-coreml":
        return [str(BIN), "--device-policy", "auto-efficiency"], env
    return [str(BIN), "--device-policy", "cpu"], env


def execute(run: dict, out: Path, windows: Path) -> dict:
    name = run_name(run)
    record = dict(run, name=name, load_average_before=sh("sysctl", "-n", "vm.loadavg"))
    env = dict(os.environ)
    rust_engine = run["engine"].startswith("rust")
    if rust_engine:
        rust_prefix, env = rust_command(run["engine"])
    t0 = time.perf_counter()
    if run["kind"] == "inference":
        target = out / "raw" / run["engine"] / f"{name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if run["engine"] == "cpp":
            cmd = [str(CPP_BIN), "bench", str(MODELS[run["model"]]), str(windows), MODEL_NAME[run["model"]],
                   run["threads"], run["policy"], str(run["warmup"]), str(run["iterations"]), str(target)]
        elif rust_engine:
            cmd = [*rust_prefix, "--config", str(CONFIG), *thread_args(run["threads"])]
            if run["graph_opt"] != "default":
                cmd += ["--graph-opt", run["graph_opt"]]
            cmd += ["benchmark", "--model", run["model"], "--mode", "inference",
                    "--warmup", str(run["warmup"]), "--iterations", str(run["iterations"]), "--trials", "1",
                    "--input-policy", run["policy"], "--output", str(target)]
        else:
            cmd = [str(PY_ENGINE), "-I", str(PY_BENCH), "--mode", "bench", "--model", run["model"],
                   "--model-path", str(MODELS[run["model"]]), "--windows", str(windows),
                   "--warmup", str(run["warmup"]), "--iterations", str(run["iterations"]),
                   "--input-policy", run["policy"]]
            if run["threads"] != "default":
                cmd += ["--intra-threads", run["threads"]]
            if run["graph_opt"] != "default":
                cmd += ["--graph-opt", run["graph_opt"]]
            cmd += ["--output", str(target)]
        record["report"] = str(target.relative_to(out))
    elif run["kind"] == "coldstart":
        if run["engine"] == "cpp":
            cmd = [str(CPP_BIN), "coldstart", str(MODELS[run["model"]]), str(windows)]
        elif rust_engine:
            cmd = [*rust_prefix, "--config", str(CONFIG), "--log-level", "error", "smoke-test", "--model", run["model"]]
        else:
            cmd = [str(PY_ENGINE), "-I", str(PY_BENCH), "--mode", "cold-start", "--model", run["model"],
                   "--model-path", str(MODELS[run["model"]]), "--windows", str(windows)]
    else:
        target = out / "raw" / "rust" / f"{name}.json"
        cmd = [*rust_prefix, "--config", str(CONFIG), "benchmark", "--model", run["model"], "--mode", run["kind"],
               "--warmup", "1", "--trials", "1", "--output", str(target)]
        record["report"] = str(target.relative_to(out))
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    record["process_wall_ms"] = (time.perf_counter() - t0) * 1000.0
    record["returncode"] = result.returncode
    record["command"] = cmd
    if run["kind"] == "coldstart" and run["engine"] in ("python", "cpp") and result.returncode == 0:
        record["inner"] = json.loads(result.stdout.strip().splitlines()[-1])
    log = out / "logs" / f"{name}.log"
    log.write_text(f"$ {' '.join(cmd)}\n--- stdout\n{result.stdout}\n--- stderr\n{result.stderr}")
    return record


def numerics(out: Path, windows: Path) -> list[dict]:
    """Logits for every window: Rust (via predict CSV) and Python ORT, at 1 and default threads."""
    records = []
    num = out / "numerics"
    num.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        for threads in ("1", "default"):
            for engine in [e for e in ("rust", "rust-coreml", "rust-msdylib") if e in ENABLED]:
                prefix, env = rust_command(engine)
                target = num / f"{engine}__{model}__t{threads}.csv"
                cmd = [*prefix, "--config", str(CONFIG), *thread_args(threads), "predict", "--model", model,
                       "--output", str(target)]
                r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
                records.append(dict(engine=engine, model=model, threads=threads, path=str(target.relative_to(out)),
                                    returncode=r.returncode, stderr=r.stderr[-2000:]))
            if "cpp" in ENABLED:
                target = num / f"cpp__{model}__t{threads}.npy"
                cmd = [str(CPP_BIN), "logits", str(MODELS[model]), str(windows), threads, str(target)]
                r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
                records.append(dict(engine="cpp", model=model, threads=threads, path=str(target.relative_to(out)),
                                    returncode=r.returncode, stderr=r.stderr[-2000:]))
            for engine, python in [(e, p) for e, p in (("python", PY_ENGINE), ("python-ort132", PY)) if e in ENABLED]:
                target = num / f"{engine}__{model}__t{threads}.npy"
                cmd = [str(python), "-I", str(PY_BENCH), "--mode", "logits", "--model", model,
                       "--model-path", str(MODELS[model]), "--windows", str(windows), "--output", str(target)]
                if threads != "default":
                    cmd += ["--intra-threads", threads]
                r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
                records.append(dict(engine=engine, model=model, threads=threads, path=str(target.relative_to(out)),
                                    returncode=r.returncode, stderr=r.stderr[-2000:]))
    return records


def verify_placement(out: Path) -> list[dict]:
    """Record ONNX Runtime's node placement for each Rust engine variant (trace log)."""
    records = []
    for engine in [e for e in ("rust", "rust-coreml", "rust-msdylib") if e in ENABLED]:
        prefix, env = rust_command(engine)
        for model in MODELS:
            cmd = [*prefix, "--config", str(CONFIG), "--log-level", "trace", "--intra-threads", "1",
                   "smoke-test", "--model", model]
            r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
            text = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + r.stderr)
            placements = {ep: int(n) for ep, n in re.findall(r"Node\(s\) placed on \[(\w+)\]\. Number of nodes: (\d+)", text)}
            build = re.findall(r"ORT Build Info: [^\n]*", text)
            records.append(dict(engine=engine, model=model, returncode=r.returncode,
                                placements=placements or {"CPUExecutionProvider": "all (single provider)"},
                                coreml_nodes=placements.get("CoreMLExecutionProvider", 0),
                                ort_build=build[0] if build else None))
            (out / "logs" / f"placement__{engine}__{model}.log").write_text(text)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=10, help="process repetitions per configuration")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--cooldown", type=float, default=1.0, help="seconds between processes")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--quick", action="store_true", help="smoke the pipeline: 2 reps, 50 iterations")
    parser.add_argument("--run-id")
    parser.add_argument("--engines", help="comma-separated engines to run (default: all): "
                        "rust,python,cpp,rust-coreml,rust-msdylib,python-ort132")
    parser.add_argument("--allow-low-power", action="store_true",
                        help="run even if macOS Low Power Mode is on (results will be tagged)")
    args = parser.parse_args()
    if low_power_mode_active() and not args.allow_low_power and not args.quick:
        raise SystemExit("macOS Low Power Mode is ON: turn it off (System Settings > Battery, or "
                         "`sudo pmset -a lowpowermode 0`) or pass --allow-low-power to tag the run.")
    if args.engines:
        ENABLED.intersection_update(args.engines.split(","))
    if args.quick:
        args.reps, args.iterations, args.cooldown = 2, 50, 0.2

    subprocess.run(["cargo", "build", "--release", "--offline", "--quiet"], cwd=REPO, check=True)
    subprocess.run(["cmake", "-S", "bench/cpp", "-B", "bench/cpp/build", "-DCMAKE_BUILD_TYPE=Release",
                    f"-DONNXRUNTIME_ROOT={CPP_SDK}"], cwd=REPO, check=True, capture_output=True)
    subprocess.run(["cmake", "--build", "bench/cpp/build", "-j"], cwd=REPO, check=True, capture_output=True)
    subprocess.run(["cargo", "build", "--release", "--offline", "--quiet", "--no-default-features",
                    "--features", "ort-dynamic", "--target-dir", "target/ort-dynamic"], cwd=REPO, check=True)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S") + ("-quick" if args.quick else "")
    out = REPO / "results/phase2" / run_id
    if out.exists():
        raise SystemExit(f"refusing to reuse existing run directory {out}")
    for sub in ("raw/rust", "raw/python", "raw/cpp", "logs", "inputs", "numerics"):
        (out / sub).mkdir(parents=True)
    # Keep the machine awake for the lifetime of this process.
    caffeinate = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())]) if shutil.which("caffeinate") else None

    env = environment_snapshot()
    (out / "env.json").write_text(json.dumps(env, indent=2))
    print(f"run {run_id}: load average {env['load_average']}; power: {(env['power'] or '').splitlines()[:1]}")

    subprocess.run([str(BIN), "--config", str(CONFIG), "export-windows", "--output-dir", str(out / "inputs")],
                   cwd=REPO, check=True, capture_output=True)
    reconstructed = REPO / "bench/data/FI2010_test_reconstructed.json"
    if reconstructed.exists():  # only relevant for configs that use the reconstructed CSV
        shutil.copy(reconstructed, out / "inputs/dataset_manifest.json")
    windows = out / "inputs/windows.npy"

    placement = verify_placement(out)
    for record in placement:
        print(f"placement {record['engine']:13s} {record['model']:15s} CoreML nodes: {record['coreml_nodes']}")
    if any(r["coreml_nodes"] for r in placement if r["engine"] in ("rust", "rust-msdylib")):
        raise SystemExit("CPU-only Rust engines placed nodes on CoreML; refusing to benchmark")

    runs = [r for r in plan_runs(args.reps, args.warmup, args.iterations) if r["engine"] in ENABLED]
    random.Random(args.seed).shuffle(runs)
    print(f"{len(runs)} isolated runs, seed {args.seed}, cooldown {args.cooldown}s")
    records = []
    started = time.time()
    for index, run in enumerate(runs, 1):
        record = execute(run, out, windows)
        record["order"] = index
        records.append(record)
        status = "ok" if record["returncode"] == 0 else f"FAILED rc={record['returncode']}"
        if index % 10 == 0 or record["returncode"] != 0:
            print(f"[{index}/{len(runs)}] {record['name']} {status} ({time.time() - started:.0f}s elapsed)", flush=True)
        time.sleep(args.cooldown)

    print("numerics: logits for all windows")
    numeric_records = numerics(out, windows)
    env_after = {"load_average": sh("sysctl", "-n", "vm.loadavg"), "thermal": sh("pmset", "-g", "therm"),
                 "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    suite = dict(run_id=run_id, args=vars(args), seed=args.seed, runs=records, numerics=numeric_records,
                 placement=placement, engines={
                     "rust": "Rust lob-inference, bundled static ORT 1.31.0, device policy cpu",
                     "python": "Python onnxruntime 1.31.0 (PyPI wheel, version-matched)",
                     "rust-coreml": "Rust lob-inference, ort-crate default MaxEfficiency policy (CoreML offload)",
                     "rust-msdylib": "Rust lob-inference (ort-dynamic) + Microsoft libonnxruntime 1.31.0 dylib",
                     "python-ort132": "Python onnxruntime 1.32.0 (numerics only)",
                     "cpp": "C++ (bench/cpp adapter of cpp-engine/cpp_inference): ORT C++ API, Microsoft ONNX Runtime 1.31.0 SDK (NuGet osx-arm64)"},
                 env_after=env_after, duration_seconds=time.time() - started)
    (out / "suite.json").write_text(json.dumps(suite, indent=2))
    failures = [r["name"] for r in records + numeric_records if r.get("returncode") != 0]
    print(f"done in {time.time() - started:.0f}s; failures: {len(failures)}")
    for name in failures:
        print("  FAILED", name)
    if caffeinate:
        caffeinate.terminate()
    print(out)


if __name__ == "__main__":
    main()
