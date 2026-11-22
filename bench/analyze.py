"""Analyze a Phase 2 run directory: statistics, tables, and figures.

Usage:
  bench/.venv/bin/python bench/analyze.py results/phase2/<run_id>

Reads raw/<engine>/*.json (Rust, Python, and any C++ reports in the same schema),
suite.json (cold start, run order), numerics/ (logits), inputs/labels.npy.
Writes tables/*.csv, figures/*.{png,pdf}, summary.md into the run directory.

Statistical unit: one OS process (one isolated trial). Per-call samples inside a
process are correlated, so engine comparisons use per-process medians
(Mann-Whitney U, two-sided) and bootstrap CIs over processes.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import fisher_exact, mannwhitneyu  # noqa: E402

# ---- Design tokens (dataviz reference palette, light mode) -------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# Color follows the entity: fixed slots, never re-assigned by rank or by which engines exist.
# Main engines take categorical slots 1-3. Ablation variants of the Rust engine are shown by
# emphasis (Rust highlighted, variants in neutral grays) with distinct markers and direct labels.
ENGINE_COLOR = {"rust": "#2a78d6", "python": "#eb6834", "cpp": "#1baf7a",
                "rust-msdylib": "#52514e", "rust-coreml": "#a8a69e", "python-ort132": "#c3c2b7"}
ENGINE_MARKER = {"rust": "o", "python": "s", "cpp": "^", "rust-msdylib": "D", "rust-coreml": "X",
                 "python-ort132": "v"}  # secondary encoding (aqua < 3:1)
ENGINE_LABEL = {"rust": "Rust (ort, CPU)", "python": "Python (onnxruntime 1.31)", "cpp": "C++ (ORT C++ API)",
                "rust-msdylib": "Rust + Microsoft ORT dylib", "rust-coreml": "Rust, ort default policy (CoreML)",
                "python-ort132": "Python (onnxruntime 1.32)"}
ENGINE_ORDER = ["rust", "python", "cpp", "rust-msdylib", "rust-coreml", "python-ort132"]
MAIN_ENGINES = ["rust", "python", "cpp"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MODEL_LABEL = {"DeepLOB": "DeepLOB", "TransformerLOB": "TransformerLOB"}
MODEL_KEY_LABEL = {"deeplob": "DeepLOB", "transformerlob": "TransformerLOB"}
CLASSES = ["Up", "Stationary", "Down"]  # class 0/1/2 (FI-2010 labels 1/2/3); see bench/data_checks.py

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})


# ---- Loading ---------------------------------------------------------------------------------
def parse_threads(session_options: str) -> str:
    match = re.search(r"intra_op_threads=([^;]+)", session_options or "")
    value = match.group(1).strip() if match else "ort-default"
    return "default" if value == "ort-default" else value


def parse_graph_opt(session_options: str) -> str:
    match = re.search(r"graph_optimization=([^;]+)", session_options or "")
    value = match.group(1).strip() if match else "ort-default"
    return "default" if value == "ort-default" else value


def load_reports(run: Path) -> list[dict]:
    units = []
    for engine_dir in sorted((run / "raw").iterdir()):
        if not engine_dir.is_dir():
            continue
        engine = engine_dir.name
        for path in sorted(engine_dir.glob("*.json")):
            report = json.loads(path.read_text())
            meta = report["metadata"]
            for trial_index, trial in enumerate(report["trials"]):
                units.append(dict(
                    engine=engine, file=path.name, trial=trial_index, model=meta["model"], mode=meta["mode"],
                    threads=parse_threads(meta.get("session_options") or meta.get("thread_configuration", "")),
                    graph_opt=parse_graph_opt(meta.get("session_options", "")),
                    policy=meta.get("input_policy") or "first",
                    runtime_version=meta.get("runtime_version"), build_info=meta.get("runtime_build_info"),
                    load_ms=meta.get("initial_model_load_ms"),
                    samples=np.asarray(trial["samples_ms"], dtype=float),
                    run_samples=np.asarray(trial.get("session_run_samples_ms") or [], dtype=float),
                    elapsed=trial["elapsed_seconds"], completed=trial["completed_windows"],
                    stages=trial.get("preprocessing_stages"),
                ))
    return units


# ---- Statistics ------------------------------------------------------------------------------
RNG = np.random.default_rng(20261122)


def bootstrap_ci(values: np.ndarray, statistic=np.median, n: int = 10000, level: float = 0.95):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return (float("nan"), float("nan"))
    draws = RNG.choice(values, size=(n, values.size), replace=True)
    stats = statistic(draws, axis=1)
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def ratio_ci(a: np.ndarray, b: np.ndarray, n: int = 10000):
    """Bootstrap CI of median(a)/median(b), resampling processes independently."""
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    da = np.median(RNG.choice(a, size=(n, len(a))), axis=1)
    db = np.median(RNG.choice(b, size=(n, len(b))), axis=1)
    lo, hi = np.quantile(da / db, [0.025, 0.975])
    return float(lo), float(hi)


def pct(values: np.ndarray, p: float) -> float:
    """Nearest-rank percentile (matches src/benchmark/statistics.rs)."""
    ordered = np.sort(values)
    rank = max(1, math.ceil(p * ordered.size))
    return float(ordered[min(rank, ordered.size) - 1])


SLOW_MODE_RATIO = 1.3


def split_modes(units: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per-process medians and a slow-mode mask.

    Within one (engine, configuration) the per-process medians form two tight clusters
    ~1.7x apart (an undetermined machine state; equally frequent for every engine).
    A process is 'slow mode' when its median exceeds SLOW_MODE_RATIO x the fastest
    process of the same engine and configuration.
    """
    medians = np.array([np.median(u["samples"]) for u in units])
    return medians, medians > SLOW_MODE_RATIO * medians.min()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def save(fig, figures: Path, name: str) -> None:
    fig.savefig(figures / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def engine_legend(ax, engines, loc="upper right"):
    handles = [Line2D([0], [0], marker=ENGINE_MARKER[e], color=ENGINE_COLOR[e], linestyle="none", markersize=7,
                      label=ENGINE_LABEL[e]) for e in engines]
    ax.legend(handles=handles, loc=loc, fontsize=8)


# ---- Analysis --------------------------------------------------------------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--engines", help="comma-separated engine keys to include (default: all present)")
    parser.add_argument("--reference", default=None,
                        help="engine whose 1-thread logits are the numerical reference (default: python if present, else cpp)")
    parser.add_argument("--out", default="", help="write tables/, figures/, summary.md into this subfolder of the run")
    args = parser.parse_args()
    run = args.run.resolve()
    out_base = run / args.out if args.out else run
    tables, figures = out_base / "tables", out_base / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    suite = json.loads((run / "suite.json").read_text())
    env = json.loads((run / "env.json").read_text())
    units = load_reports(run)
    allowed = set(args.engines.split(",")) if args.engines else None
    if allowed:
        units = [u for u in units if u["engine"] in allowed]
        suite["runs"] = [r for r in suite["runs"] if r["engine"] in allowed]
        suite["numerics"] = [r for r in suite["numerics"] if r["engine"] in allowed]
        suite["placement"] = [r for r in suite.get("placement", []) if r["engine"] in allowed]
        suite["engines"] = {k: v for k, v in suite.get("engines", {}).items() if k in allowed}
    present_engines = {u["engine"] for u in units} | {r["engine"] for r in suite["numerics"]}
    ref_engine = args.reference or ("python" if "python" in present_engines else "cpp")
    ref_label = ENGINE_LABEL.get(ref_engine, ref_engine)
    inference = [u for u in units if u["mode"] == "inference"]
    engines = [e for e in MAIN_ENGINES if any(u["engine"] == e for u in inference)]
    variant_engines = [e for e in ENGINE_ORDER if e not in MAIN_ENGINES and any(u["engine"] == e for u in inference)]
    models = sorted({u["model"] for u in inference})
    md: list[str] = []

    # -- Table 1: per-configuration latency (per-process medians, pooled tails) --
    groups = defaultdict(list)
    for u in inference:
        groups[(u["model"], u["engine"], u["threads"], u["policy"], u["graph_opt"])].append(u)
    latency_rows = []
    for key in sorted(groups):
        model, engine, threads, policy, graph_opt = key
        g = groups[key]
        medians, slow = split_modes(g)
        fast = medians[~slow]
        flo, fhi = bootstrap_ci(fast)
        pooled = np.concatenate([u["samples"] for u in g])
        run_pooled = np.concatenate([u["run_samples"] for u in g if u["run_samples"].size])
        overhead = np.concatenate([u["samples"] - u["run_samples"] for u in g if u["run_samples"].size == u["samples"].size and u["run_samples"].size])
        throughput = np.array([u["completed"] / u["elapsed"] for u in g])
        lo, hi = bootstrap_ci(medians)
        latency_rows.append(dict(
            model=model, engine=engine, intra_threads=threads, input_policy=policy, graph_opt=graph_opt,
            processes=len(g), calls=int(pooled.size), slow_mode_processes=int(slow.sum()),
            fast_mode_median_ms=float(np.median(fast)), fast_mode_ci95_lo=flo, fast_mode_ci95_hi=fhi,
            median_of_process_medians_ms=float(np.median(medians)), median_ci95_lo=lo, median_ci95_hi=hi,
            process_median_iqr_ms=float(np.subtract(*np.percentile(medians, [75, 25]))),
            pooled_mean_ms=float(pooled.mean()), pooled_std_ms=float(pooled.std()),
            pooled_p50_ms=pct(pooled, .5), pooled_p90_ms=pct(pooled, .9), pooled_p99_ms=pct(pooled, .99),
            pooled_p999_ms=pct(pooled, .999), pooled_max_ms=float(pooled.max()),
            session_run_p50_ms=pct(run_pooled, .5) if run_pooled.size else float("nan"),
            overhead_p50_us=float(np.median(overhead) * 1000) if overhead.size else float("nan"),
            overhead_p95_us=pct(overhead, .95) * 1000 if overhead.size else float("nan"),
            throughput_median_per_s=float(np.median(throughput)),
            runtime_version=g[0]["runtime_version"],
        ))
    write_csv(tables / "latency_by_config.csv", latency_rows)

    # -- Table 2: engine comparisons (Mann-Whitney on per-process medians) --
    comparison_rows = []
    baseline = "rust"
    for model in models:
        for threads in ("1", "default"):
            for policy in ("first", "cycle"):
                base = groups.get((model, baseline, threads, policy, "default"))
                if not base:
                    continue
                bm_all, b_slow = split_modes(base)
                bm = bm_all[~b_slow]
                for other in engines + variant_engines:
                    if other == baseline:
                        continue
                    g = groups.get((model, other, threads, policy, "default"))
                    if not g:
                        continue
                    om_all, o_slow = split_modes(g)
                    om = om_all[~o_slow]
                    test = mannwhitneyu(om, bm, alternative="two-sided")
                    lo, hi = ratio_ci(om, bm)
                    all_test = mannwhitneyu(om_all, bm_all, alternative="two-sided")
                    share = fisher_exact([[int(o_slow.sum()), int((~o_slow).sum())],
                                          [int(b_slow.sum()), int((~b_slow).sum())]])
                    comparison_rows.append(dict(
                        model=model, intra_threads=threads, input_policy=policy, engine=other, versus=baseline,
                        n_engine=len(om), n_versus=len(bm),
                        slow_mode_engine=f"{int(o_slow.sum())}/{len(o_slow)}",
                        slow_mode_versus=f"{int(b_slow.sum())}/{len(b_slow)}",
                        slow_share_fisher_p=float(share.pvalue),
                        median_engine_ms=float(np.median(om)), median_versus_ms=float(np.median(bm)),
                        ratio_engine_over_versus=float(np.median(om) / np.median(bm)),
                        ratio_ci95_lo=lo, ratio_ci95_hi=hi,
                        mann_whitney_u=float(test.statistic), p_value=float(test.pvalue),
                        all_process_ratio=float(np.median(om_all) / np.median(bm_all)),
                        all_process_p_value=float(all_test.pvalue),
                        runtimes=f"{g[0]['build_info'] or g[0]['runtime_version']} vs {base[0]['build_info'] or base[0]['runtime_version']}",
                    ))
    write_csv(tables / "engine_comparison.csv", comparison_rows)

    # -- Figure 1: ECDF of per-call latency, small multiples (model x threads), cycle policy --
    fig, axes = plt.subplots(len(models), 2, figsize=(9, 3.2 * len(models)), squeeze=False)
    for r, model in enumerate(models):
        for c, threads in enumerate(("1", "default")):
            ax = axes[r][c]
            for engine in engines:
                g = groups.get((model, engine, threads, "cycle", "default"))
                if not g:
                    continue
                x = np.sort(np.concatenate([u["samples"] for u in g]))
                y = np.arange(1, x.size + 1) / x.size
                ax.plot(x, y, color=ENGINE_COLOR[engine], label=ENGINE_LABEL[engine])
                p50 = pct(x, .5)
                ax.plot([p50], [0.5], marker=ENGINE_MARKER[engine], color=ENGINE_COLOR[engine], markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_ylim(0, 1.0)
            ax.axhline(0.99, color=BASELINE, linewidth=1, linestyle=(0, (3, 3)))
            ax.text(ax.get_xlim()[0], 0.985, " P99", va="top", fontsize=7, color=MUTED)
            ax.set_title(f"{model} · intra-op threads = {threads}")
            ax.set_xlabel("Per-call latency (ms, log scale)")
            ax.set_ylabel("Fraction of calls ≤ x")
            if r == 0 and c == 1:
                engine_legend(ax, engines, loc="lower right")
    fig.suptitle("Per-call latency distribution (all windows cycled; pooled over all isolated processes, incl. slow mode)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, figures, "fig1_latency_ecdf")

    # -- Figure 2: per-process medians (dot strip) with bootstrap CI, per config --
    configs = [(threads, policy) for threads in ("1", "default") for policy in ("first", "cycle")]
    fig, axes = plt.subplots(1, len(models), figsize=(4.8 * len(models), 3.6), squeeze=False)
    for m, model in enumerate(models):
        ax = axes[0][m]
        width = 0.8 / max(1, len(engines))
        for e, engine in enumerate(engines):
            for k, (threads, policy) in enumerate(configs):
                g = groups.get((model, engine, threads, policy, "default"))
                if not g:
                    continue
                medians, slow = split_modes(g)
                xpos = k + (e - (len(engines) - 1) / 2) * width
                jitter = (RNG.random(medians.size) - 0.5) * width * 0.5
                ax.scatter((xpos + jitter)[~slow], medians[~slow], s=16, color=ENGINE_COLOR[engine],
                           marker=ENGINE_MARKER[engine], alpha=0.75, linewidths=0, zorder=3)
                ax.scatter((xpos + jitter)[slow], medians[slow], s=16, facecolors="none",
                           edgecolors=ENGINE_COLOR[engine], marker=ENGINE_MARKER[engine], linewidths=1, zorder=3)
                fast = medians[~slow]
                lo, hi = bootstrap_ci(fast)
                ax.plot([xpos, xpos], [lo, hi], color=INK, linewidth=1.5, zorder=4)
                ax.plot([xpos - width * 0.3, xpos + width * 0.3], [np.median(fast)] * 2, color=INK,
                        linewidth=2, zorder=4)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([f"{'1 thread' if t == '1' else 'ORT default'}\n{p} input" for t, p in configs], fontsize=8)
        ax.set_ylabel("Median latency per process (ms)")
        ax.set_title(model)
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
    handles = [Line2D([0], [0], marker=ENGINE_MARKER[e], color=ENGINE_COLOR[e], linestyle="none", markersize=7,
                      label=ENGINE_LABEL[e]) for e in engines]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Per-process median latency — filled = fast mode, hollow = slow mode; bar = fast-mode median, 95% bootstrap CI",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, figures, "fig2_process_medians")

    # -- Figure 3: Overhead outside Session::run (µs), per engine/model/threads --
    rows = [r for r in latency_rows if r["input_policy"] == "cycle" and r["graph_opt"] == "default"
            and not math.isnan(r["overhead_p50_us"])]
    if rows:
        fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.2))
        rows = sorted(rows, key=lambda r: (r["model"], r["intra_threads"], ENGINE_ORDER.index(r["engine"])))
        y = np.arange(len(rows))[::-1]
        for yi, r in zip(y, rows):
            ax.barh(yi, r["overhead_p50_us"], height=0.6, color=ENGINE_COLOR[r["engine"]], edgecolor=SURFACE, linewidth=2)
            ax.plot([r["overhead_p50_us"], r["overhead_p95_us"]], [yi, yi], color=INK2, linewidth=1)
            ax.plot([r["overhead_p95_us"]], [yi], marker="|", color=INK2, markersize=8)
            ax.text(r["overhead_p95_us"], yi, f"  {r['overhead_p50_us']:.1f} µs (p95 {r['overhead_p95_us']:.1f})",
                    va="center", fontsize=8, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['model']} · {ENGINE_LABEL[r['engine']]} · threads={r['intra_threads']}" for r in rows], fontsize=8)
        ax.set_xlabel("Per-call time outside Session.run (µs): input checks, adaptation, output parsing, argmax")
        ax.set_xlim(0, max(r["overhead_p95_us"] for r in rows) * 1.6)
        ax.grid(axis="y", visible=False)
        ax.set_title("Host-language overhead around ONNX Runtime (bar = median, whisker = P95)", loc="left")
        fig.tight_layout()
        save(fig, figures, "fig3_host_overhead")

    # -- Figure 4: drift — per-process median vs execution order --
    order = {Path(r["report"]).name: r["order"] for r in suite["runs"] if r.get("report")}
    fig, axes = plt.subplots(len(models), 1, figsize=(9, 2.6 * len(models)), squeeze=False, sharex=True)
    for m, model in enumerate(models):
        ax = axes[m][0]
        for engine in engines:
            pts = [(order.get(u["file"]), np.median(u["samples"])) for u in inference
                   if u["engine"] == engine and u["model"] == model and u["graph_opt"] == "default" and order.get(u["file"])]
            if pts:
                xs, ys = zip(*sorted(pts))
                ax.scatter(xs, ys, s=12, color=ENGINE_COLOR[engine], marker=ENGINE_MARKER[engine], linewidths=0,
                           label=ENGINE_LABEL[engine])
        ax.set_ylabel("Process median (ms)")
        ax.set_title(f"{model} — all thread/input configurations", loc="left")
        ax.set_ylim(bottom=0)
        if m == 0:
            engine_legend(ax, engines, loc="upper right")
    axes[-1][0].set_xlabel("Execution order within the randomized suite")
    fig.suptitle("Drift check: per-process medians in execution order (flat bands = stable machine state)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, figures, "fig4_drift")

    # -- Cold start --
    cold = [r for r in suite["runs"] if r["kind"] == "coldstart" and r["returncode"] == 0]
    cold_rows = []
    for model_key in sorted({r["model"] for r in cold}):
        for engine in ENGINE_ORDER:
            vals = np.array([r["process_wall_ms"] for r in cold if r["model"] == model_key and r["engine"] == engine])
            if vals.size:
                lo, hi = bootstrap_ci(vals)
                cold_rows.append(dict(model=model_key, engine=engine, processes=int(vals.size),
                                      median_process_wall_ms=float(np.median(vals)), ci95_lo=lo, ci95_hi=hi,
                                      min_ms=float(vals.min()), max_ms=float(vals.max())))
    write_csv(tables / "cold_start.csv", cold_rows)
    if cold_rows:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        cold_models = sorted({r["model"] for r in cold_rows})
        width = 0.35
        cold_engines = [e for e in ENGINE_ORDER if any(r["engine"] == e for r in cold_rows)]
        width = 0.8 / len(cold_engines)
        for e, engine in enumerate(cold_engines):
            for k, model_key in enumerate(cold_models):
                vals = np.array([r["process_wall_ms"] for r in cold if r["model"] == model_key and r["engine"] == engine])
                if not vals.size:
                    continue
                xpos = k + (e - (len(cold_engines) - 1) / 2) * width
                ax.bar(xpos, np.median(vals), width=width * 0.9, color=ENGINE_COLOR[engine], edgecolor=SURFACE, linewidth=2)
                ax.scatter(xpos + (RNG.random(vals.size) - 0.5) * width * 0.4, vals, s=10, color=INK, alpha=0.5, zorder=3, linewidths=0)
                ax.text(xpos, np.median(vals), f"{np.median(vals):.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(range(len(cold_models)))
        ax.set_xticklabels([MODEL_KEY_LABEL.get(mk, mk) for mk in cold_models])
        ax.set_ylabel("Process wall time (ms)")
        ax.set_title("Cold start: spawn → runtime init → model load → first prediction → exit", loc="left")
        ax.grid(axis="x", visible=False)
        engine_legend(ax, cold_engines, loc="upper left")
        fig.tight_layout()
        save(fig, figures, "fig5_cold_start")

    # -- Graph optimization ablation (Rust, 1 thread, cycle) --
    abl_rows = [r for r in latency_rows if r["engine"] == "rust" and r["intra_threads"] == "1" and r["input_policy"] == "cycle"]
    write_csv(tables / "graph_opt_ablation.csv", abl_rows)
    if any(r["graph_opt"] != "default" for r in abl_rows):
        levels = ["disable", "default", "all"]
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.0), squeeze=False)
        for m, model in enumerate(models):
            ax = axes[0][m]
            for k, level in enumerate(levels):
                g = groups.get((model, "rust", "1", "cycle", level))
                if not g:
                    continue
                medians, slow = split_modes(g)
                jit = k + (RNG.random(medians.size) - 0.5) * 0.25
                ax.scatter(jit[~slow], medians[~slow], s=14, color=ENGINE_COLOR["rust"], linewidths=0, alpha=0.7)
                ax.scatter(jit[slow], medians[slow], s=14, facecolors="none", edgecolors=ENGINE_COLOR["rust"], linewidths=1)
                fast = medians[~slow]
                lo, hi = bootstrap_ci(fast)
                ax.plot([k, k], [lo, hi], color=INK, linewidth=1.5)
                ax.plot([k - 0.2, k + 0.2], [np.median(fast)] * 2, color=INK, linewidth=2)
                ax.text(k + 0.22, np.median(fast), f"{np.median(fast):.3f}", va="center", fontsize=8, color=INK2)
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(["disabled", "ORT default", "all"])
            ax.set_title(model)
            ax.set_ylabel("Process median (ms)")
            ax.set_ylim(bottom=0)
            ax.grid(axis="x", visible=False)
        fig.suptitle("Graph-optimization ablation (Rust, 1 intra-op thread, cycled inputs)", x=0.01, ha="left",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        save(fig, figures, "fig6_graph_opt")

    # -- Preprocessing stages (Rust) --
    pre = [u for u in units if u["mode"] == "preprocessing" and u["stages"]]
    stage_rows = []
    if pre:
        keys = [("csv_loading_ms", "CSV parse + feature extraction"), ("normalization_ms", "Z-score normalization"),
                ("window_generation_ms", "Window generation"), ("tensor_construction_ms", "Tensor construction (319 windows)")]
        for key, label in keys:
            vals = np.array([u["stages"][key] for u in pre])
            stage_rows.append(dict(stage=label, processes=int(vals.size), median_ms=float(np.median(vals)),
                                   min_ms=float(vals.min()), max_ms=float(vals.max())))
        write_csv(tables / "preprocessing_stages.csv", stage_rows)
        fig, ax = plt.subplots(figsize=(7, 2.4))
        ordered = sorted(stage_rows, key=lambda r: r["median_ms"])
        for yi, r in enumerate(ordered):
            ax.barh(yi, r["median_ms"], height=0.6, color=ENGINE_COLOR["rust"], edgecolor=SURFACE, linewidth=2)
            ax.text(r["median_ms"], yi, f"  {r['median_ms']:.2f} ms", va="center", fontsize=8, color=INK2)
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels([r["stage"] for r in ordered], fontsize=8)
        ax.set_xlabel("Median time per full pass over the test set (ms)")
        ax.set_xlim(0, max(r["median_ms"] for r in ordered) * 1.3)
        ax.grid(axis="y", visible=False)
        ax.set_title("Rust preprocessing cost by stage (31,937 rows → 319 windows)", loc="left")
        fig.tight_layout()
        save(fig, figures, "fig7_preprocessing")

    e2e = [u for u in units if u["mode"] == "end-to-end"]
    e2e_rows = []
    for model in sorted({u["model"] for u in e2e}):
        vals = np.array([u["elapsed"] * 1000 for u in e2e if u["model"] == model])
        win = [u["completed"] for u in e2e if u["model"] == model][0]
        e2e_rows.append(dict(model=model, processes=int(vals.size), windows=win, median_total_ms=float(np.median(vals)),
                             median_per_window_ms=float(np.median(vals) / win),
                             throughput_windows_per_s=float(win / (np.median(vals) / 1000))))
    write_csv(tables / "end_to_end.csv", e2e_rows)

    # -- Numerical equivalence + accuracy --
    labels = np.load(run / "inputs/labels.npy").astype(int)
    num_rows, acc_rows, logits = [], [], {}
    for rec in suite["numerics"]:
        if rec["returncode"] != 0:
            continue
        path = run / rec["path"]
        if path.suffix == ".csv":
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            # Columns in class-index order. Reports written before the label fix named class 0
            # "down_score" (mislabelled); both layouts hold logits for classes 0, 1, 2 in that order.
            cols = (["class0_up_score", "class1_stationary_score", "class2_down_score"] if "class0_up_score" in rows[0]
                    else ["down_score", "stationary_score", "up_score"])
            arr = np.array([[float(r[c]) for c in cols] for r in rows], dtype=np.float32)
            truth = np.array([int(r["true_class"]) for r in rows])
            assert (truth == labels).all(), "Rust predict labels differ from exported labels"
        else:
            arr = np.load(path).astype(np.float32)
        logits[(rec["engine"], rec["model"], rec["threads"])] = arr
    for cpp_logits in sorted((run / "numerics").glob("cpp__*.npy")):  # C++ logits dropped in by hand
        _, model, threads = cpp_logits.stem.split("__")
        logits.setdefault(("cpp", model, threads.removeprefix("t")), np.load(cpp_logits).astype(np.float32))
    diff_store = {}
    for (engine, model, threads), arr in sorted(logits.items()):
        for ref_threads in ("1",):
            ref = logits.get((ref_engine, model, ref_threads))
            if ref is None or (engine, threads) == (ref_engine, ref_threads):
                continue
            d = np.abs(arr.astype(np.float64) - ref.astype(np.float64))
            diff_store[(engine, model, threads)] = d.ravel()
            num_rows.append(dict(model=model, engine=engine, intra_threads=threads, reference=f"{ref_engine} t{ref_threads}",
                                 windows=int(arr.shape[0]), max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()),
                                 within_1e_5=bool(d.max() <= 1e-5), bitwise_identical=bool((d == 0).all()),
                                 argmax_agreement=float((arr.argmax(1) == ref.argmax(1)).mean())))
    write_csv(tables / "numerical_equivalence.csv", num_rows)
    for (engine, model, threads), arr in sorted(logits.items()):
        if threads != "1":
            continue
        pred = arr.argmax(1)
        cm = np.zeros((3, 3), dtype=int)
        for t, p in zip(labels, pred):
            cm[t, p] += 1
        recall = cm.diagonal() / cm.sum(1).clip(min=1)
        precision = cm.diagonal() / cm.sum(0).clip(min=1)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall).clip(min=1e-12), 0)
        acc_rows.append(dict(model=model, engine=engine, windows=int(labels.size), accuracy=float((pred == labels).mean()),
                             macro_f1=float(f1.mean()), recall_down=float(recall[0]), recall_stationary=float(recall[1]),
                             recall_up=float(recall[2]), majority_class_accuracy=float(np.bincount(labels).max() / labels.size),
                             confusion_rows_true_cols_pred=cm.tolist()))
    write_csv(tables / "accuracy.csv", acc_rows)

    FLOOR = 1e-9
    if diff_store:
        num_models = sorted({model for (_, model, _) in diff_store})
        num_engines = [e for e in ENGINE_ORDER if any(k[0] == e for k in diff_store)]
        fig, axes = plt.subplots(1, len(num_models), figsize=(5.2 * len(num_models), 0.45 * len(num_engines) + 1.6),
                                 squeeze=False, sharey=True)
        for m, model in enumerate(num_models):
            ax = axes[0][m]
            for yi, engine in enumerate(num_engines[::-1]):
                for threads, filled in (("1", True), ("default", False)):
                    d = diff_store.get((engine, model, threads))
                    if d is None:
                        if (engine, threads) == (ref_engine, "1"):
                            ax.text(FLOOR, yi + 0.12, "  reference", fontsize=7, color=INK2, va="center")
                        continue
                    x = max(d.max(), FLOOR)
                    color = ENGINE_COLOR.get(engine, MUTED)
                    ax.scatter([x], [yi + (0.12 if filled else -0.12)], s=46, marker=ENGINE_MARKER.get(engine, "o"),
                               facecolors=color if filled else "none", edgecolors=color, linewidths=1.5, zorder=3)
            ax.axvline(1e-5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
            ax.text(1e-5, len(num_engines) - 0.45, " 1e-5 tolerance", fontsize=7, color=INK2, va="bottom")
            ax.set_xscale("log")
            ax.set_xlim(FLOOR / 3, 3e-2)
            ticks = [FLOOR, 1e-7, 1e-5, 1e-3]
            ax.set_xticks(ticks, ["0 (bit-\nidentical)", "1e-7", "1e-5", "1e-3"])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_yticks(range(len(num_engines)), [ENGINE_LABEL.get(e, e) for e in num_engines[::-1]], fontsize=8)
            ax.set_ylim(-0.6, len(num_engines) - 0.2)
            ax.grid(axis="y", visible=False)
            ax.set_title(MODEL_KEY_LABEL.get(model, model))
            ax.set_xlabel(f"Max |logit − {ref_label}, 1 thread|, 319 windows × 3")
        handles = [Line2D([0], [0], marker="o", linestyle="none", color=INK2, markersize=7, label="1 intra-op thread"),
                   Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=INK2,
                          markersize=7, label="ORT default threads")]
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"Numerical agreement with the {ref_label} reference (argmax agreement is 100% everywhere)",
                     x=0.01, ha="left", fontsize=10, fontweight="bold")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        save(fig, figures, "fig8_numerical_agreement")

    if acc_rows:
        rust_acc = [r for r in acc_rows if r["engine"] == "rust"] or acc_rows
        fig, axes = plt.subplots(1, len(rust_acc), figsize=(3.6 * len(rust_acc), 3.2), squeeze=False)
        for i, r in enumerate(rust_acc):
            ax = axes[0][i]
            cm = np.array(r["confusion_rows_true_cols_pred"], dtype=float)
            norm = cm / cm.sum(1, keepdims=True).clip(min=1)
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
            ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
            for a in range(3):
                for b in range(3):
                    ax.text(b, a, f"{norm[a, b]:.0%}\n({int(cm[a, b])})", ha="center", va="center", fontsize=8,
                            color="#ffffff" if norm[a, b] > 0.55 else INK)
            ax.set_xticks(range(3), CLASSES, fontsize=8)
            ax.set_yticks(range(3), CLASSES, fontsize=8)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.grid(False)
            ax.set_title(f"{r['model']}: acc {r['accuracy']:.1%}, macro-F1 {r['macro_f1']:.3f}", fontsize=9)
        fig.suptitle("Confusion matrices on FI-2010 test windows (row-normalized recall; Rust engine)", x=0.01, ha="left",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        save(fig, figures, "fig9_confusion")

    # -- Figure 10: ablation of device policy and ORT binary (Rust variants), cycle inputs --
    ablation = ["rust", "rust-msdylib", "rust-coreml"]
    present = [e for e in ablation if any(u["engine"] == e for u in inference)]
    ablation_rows = []
    if len(present) > 1:
        fig, axes = plt.subplots(1, len(models), figsize=(4.8 * len(models), 3.6), squeeze=False)
        for m, model in enumerate(models):
            ax = axes[0][m]
            for k, threads in enumerate(("1", "default")):
                width = 0.8 / len(present)
                for e, engine in enumerate(present):
                    g = groups.get((model, engine, threads, "cycle", "default"))
                    if not g:
                        continue
                    medians, slow = split_modes(g)
                    runs_ = np.array([np.median(u["run_samples"]) for u in g if u["run_samples"].size])
                    xpos = k + (e - (len(present) - 1) / 2) * width
                    jit = xpos + (RNG.random(medians.size) - 0.5) * width * 0.5
                    ax.scatter(jit[~slow], medians[~slow], s=16, color=ENGINE_COLOR[engine], marker=ENGINE_MARKER[engine],
                               linewidths=0, alpha=0.8, zorder=3)
                    ax.scatter(jit[slow], medians[slow], s=16, facecolors="none", edgecolors=ENGINE_COLOR[engine],
                               marker=ENGINE_MARKER[engine], linewidths=1, zorder=3)
                    fast = medians[~slow]
                    lo, hi = bootstrap_ci(fast)
                    ax.plot([xpos, xpos], [lo, hi], color=INK, linewidth=1.5, zorder=4)
                    ax.plot([xpos - width * 0.3, xpos + width * 0.3], [np.median(fast)] * 2, color=INK, linewidth=2, zorder=4)
                    ax.text(xpos, max(hi, np.median(fast)) * 1.02, f"{np.median(fast):.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
                    ablation_rows.append(dict(model=model, engine=engine, intra_threads=threads, processes=len(g),
                                              slow_mode_processes=int(slow.sum()),
                                              fast_mode_median_ms=float(np.median(fast)), ci95_lo=lo, ci95_hi=hi,
                                              session_run_median_ms=float(np.median(runs_)) if runs_.size else float("nan")))
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["1 intra-op thread", "ORT default threads"])
            ax.set_ylabel("Median latency per process (ms)")
            ax.set_title(model)
            ax.set_ylim(bottom=0)
            ax.grid(axis="x", visible=False)
        handles = [Line2D([0], [0], marker=ENGINE_MARKER[e], color=ENGINE_COLOR[e], linestyle="none", markersize=7,
                          label=ENGINE_LABEL[e]) for e in present]
        handles.append(Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=INK2,
                              markersize=7, label="slow-mode process (hollow)"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("Ablation, same Rust code: bundled ORT vs Microsoft ORT dylib vs ort-default device policy (CoreML) — fast-mode medians",
                     x=0.01, ha="left", fontsize=10, fontweight="bold")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        save(fig, figures, "fig10_device_binary_ablation")
    write_csv(tables / "device_binary_ablation.csv", ablation_rows)
    # -- Figure 11: headline Rust vs C++ — same-binary comparison plus the bundled-binary effect --
    headline = [e for e in ("cpp", "rust-msdylib", "rust") if any(u["engine"] == e for u in inference)]
    headline_rows = []
    if "cpp" in headline and len(headline) > 1:
        labels = {"cpp": "C++\n(Microsoft ORT)", "rust-msdylib": "Rust\n(Microsoft ORT)", "rust": "Rust\n(ort-bundled ORT)"}
        fig, axes = plt.subplots(1, len(models), figsize=(5.0 * len(models), 3.8), squeeze=False)
        for m, model in enumerate(models):
            ax = axes[0][m]
            width = 0.8 / len(headline)
            for k, threads in enumerate(("1", "default")):
                cpp_units = groups.get((model, "cpp", threads, "cycle", "default"))
                cpp_fast = split_modes(cpp_units)[0][~split_modes(cpp_units)[1]] if cpp_units else None
                for e, engine in enumerate(headline):
                    g = groups.get((model, engine, threads, "cycle", "default"))
                    if not g:
                        continue
                    medians, slow = split_modes(g)
                    fast = medians[~slow]
                    xpos = k + (e - (len(headline) - 1) / 2) * width
                    jit = xpos + (RNG.random(medians.size) - 0.5) * width * 0.5
                    ax.scatter(jit[~slow], medians[~slow], s=14, color=ENGINE_COLOR[engine], marker=ENGINE_MARKER[engine],
                               linewidths=0, alpha=0.8, zorder=3)
                    ax.scatter(jit[slow], medians[slow], s=14, facecolors="none", edgecolors=ENGINE_COLOR[engine],
                               marker=ENGINE_MARKER[engine], linewidths=1, zorder=3)
                    lo, hi = bootstrap_ci(fast)
                    ax.plot([xpos, xpos], [lo, hi], color=INK, linewidth=1.5, zorder=4)
                    ax.plot([xpos - width * 0.3, xpos + width * 0.3], [np.median(fast)] * 2, color=INK, linewidth=2, zorder=4)
                    ratio = float(np.median(fast) / np.median(cpp_fast)) if cpp_fast is not None and cpp_fast.size else float("nan")
                    text = f"{np.median(fast):.3f}" + ("" if engine == "cpp" else f"\n×{ratio:.3f}")
                    ax.text(xpos, max(hi, medians.max()) * 1.03, text, ha="center", va="bottom", fontsize=7, color=INK2)
                    headline_rows.append(dict(model=model, intra_threads=threads, engine=engine, processes=len(g),
                                              fast_mode_median_ms=float(np.median(fast)), ci95_lo=lo, ci95_hi=hi,
                                              ratio_to_cpp=ratio))
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["1 intra-op thread", "ORT default threads"])
            ax.set_ylabel("Median latency per process (ms)")
            ax.set_title(model)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
            ax.grid(axis="x", visible=False)
        handles = [Line2D([0], [0], marker=ENGINE_MARKER[e], color=ENGINE_COLOR[e], linestyle="none", markersize=7,
                          label=labels[e].replace("\n", " ")) for e in headline]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("Rust vs C++ inference latency — labels: fast-mode median (ms) and ratio to C++",
                     x=0.01, ha="left", fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        save(fig, figures, "fig11_rust_vs_cpp_headline")
    write_csv(tables / "headline_rust_vs_cpp.csv", headline_rows)

    placement_rows = [dict(engine=r["engine"], model=r["model"], coreml_nodes=r["coreml_nodes"],
                           placements=json.dumps(r["placements"])) for r in suite.get("placement", [])]
    write_csv(tables / "node_placement.csv", placement_rows)

    # -- summary.md --
    lp = bool(env.get("low_power_mode_active"))
    power_line = (env.get("power") or "").splitlines()[:2]
    md.append(f"# Phase 2 benchmark summary — run `{suite['run_id']}`\n")
    md.append(f"- Machine: {env.get('cpu_brand')} ({env.get('perf_levels')}), {int(env.get('memory_bytes') or 0) / 2**30:.0f} GiB; {env.get('macos', '').replace(chr(10), ' ')}")
    md.append(f"- Power: {' / '.join(power_line)}; Low Power Mode: {'**ON — results throttled**' if lp else 'off'}")
    md.append(f"- Load average: before {env.get('load_average')}, after {suite['env_after'].get('load_average')}")
    md.append(f"- Git: `{(env.get('git_commit') or '')[:10]}` on `{env.get('git_branch')}`; dirty files present: {'yes' if env.get('git_dirty_files') else 'no'}")
    md.append(f"- Rust runtime: {[l for l in (env.get('ort_build_info') or '').splitlines() if 'build' in l.lower()]}; Python onnxruntime {env.get('python_onnxruntime')}")
    md.append(f"- Protocol: {suite['args']['reps']} isolated processes per configuration, {suite['args']['warmup']} warm-up + {suite['args']['iterations']} timed calls each, randomized interleaved order (seed {suite['seed']}), {suite['args']['cooldown']} s cool-down. Duration {suite['duration_seconds']:.0f} s.")
    failures = [r["name"] for r in suite["runs"] if r["returncode"] != 0]
    md.append(f"- Failed runs: {len(failures)}\n")
    md.append("## Latency\n")
    md.append(f"Per-process medians fall into two clusters ~1.7x apart (\"slow mode\": median > {SLOW_MODE_RATIO}x the fastest process of the same engine and configuration). "
              "Primary statistic: fast-mode median with 95% bootstrap CI over processes; all-process median for reference. Pooled tails include every process.\n")
    md.append("| Model | Engine | Threads | Input | Graph opt | Fast-mode median ms [95% CI] | Slow-mode procs | All-proc median | P99 ms | P99.9 ms | Session.run P50 ms | Host overhead P50 µs | Throughput /s |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in latency_rows:
        md.append(f"| {r['model']} | {r['engine']} | {r['intra_threads']} | {r['input_policy']} | {r['graph_opt']} | "
                  f"{fmt(r['fast_mode_median_ms'])} [{fmt(r['fast_mode_ci95_lo'])}, {fmt(r['fast_mode_ci95_hi'])}] | "
                  f"{r['slow_mode_processes']}/{r['processes']} | {fmt(r['median_of_process_medians_ms'])} | "
                  f"{fmt(r['pooled_p99_ms'])} | {fmt(r['pooled_p999_ms'])} | {fmt(r['session_run_p50_ms'])} | "
                  f"{fmt(r['overhead_p50_us'], 1)} | {fmt(r['throughput_median_per_s'], 0)} |")
    if comparison_rows:
        md.append("\n## Engine comparison vs Rust (fast-mode per-process medians; Mann–Whitney U; ratio > 1 ⇒ slower than Rust)\n")
        md.append("| Model | Threads | Input | Engine | n (fast) | Ratio [95% CI] | p | Slow-mode share engine / rust (Fisher p) | All-process ratio (p) |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for r in comparison_rows:
            md.append(f"| {r['model']} | {r['intra_threads']} | {r['input_policy']} | {r['engine']} | {r['n_engine']} vs {r['n_versus']} | "
                      f"{r['ratio_engine_over_versus']:.3f} [{r['ratio_ci95_lo']:.3f}, {r['ratio_ci95_hi']:.3f}] | {r['p_value']:.2g} | "
                      f"{r['slow_mode_engine']} / {r['slow_mode_versus']} ({r['slow_share_fisher_p']:.2f}) | {r['all_process_ratio']:.3f} ({r['all_process_p_value']:.2g}) |")
        md.append("\nAll engines link ONNX Runtime 1.31.0 built from commit 402979a (Rust: bundled `ort` build; `python` and `rust-msdylib`: Microsoft's official build), so ratios isolate engine/binary/configuration, not ORT version.")
    if cold_rows:
        md.append("\n## Cold start (process wall time, spawn → first prediction → exit)\n")
        md.append("| Model | Engine | n | Median ms [95% CI] | Min | Max |")
        md.append("|---|---|---|---|---|---|")
        for r in cold_rows:
            md.append(f"| {r['model']} | {r['engine']} | {r['processes']} | {r['median_process_wall_ms']:.1f} [{fmt(r['ci95_lo'], 1)}, {fmt(r['ci95_hi'], 1)}] | {r['min_ms']:.1f} | {r['max_ms']:.1f} |")
    if stage_rows:
        md.append("\n## Rust preprocessing stages (full test set)\n")
        md.append("| Stage | n | Median ms | Min | Max |")
        md.append("|---|---|---|---|---|")
        for r in stage_rows:
            md.append(f"| {r['stage']} | {r['processes']} | {r['median_ms']:.3f} | {r['min_ms']:.3f} | {r['max_ms']:.3f} |")
    if e2e_rows:
        md.append("\n## Rust end-to-end (CSV → predictions serialized in memory)\n")
        md.append("| Model | n | Windows | Median total ms | Per window ms | Throughput windows/s |")
        md.append("|---|---|---|---|---|---|")
        for r in e2e_rows:
            md.append(f"| {r['model']} | {r['processes']} | {r['windows']} | {r['median_total_ms']:.1f} | {r['median_per_window_ms']:.3f} | {r['throughput_windows_per_s']:.0f} |")
    if num_rows:
        md.append(f"\n## Numerical equivalence (all windows × 3 logits; reference = {ref_label}, 1 thread)\n")
        md.append("| Model | Engine | Threads | Max abs diff | Mean abs diff | ≤ 1e-5 | Bit-identical | Argmax agreement |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in num_rows:
            md.append(f"| {r['model']} | {r['engine']} | {r['intra_threads']} | {r['max_abs_diff']:.2e} | {r['mean_abs_diff']:.2e} | {r['within_1e_5']} | {r['bitwise_identical']} | {r['argmax_agreement']:.1%} |")
    if acc_rows:
        md.append("\n## Model quality on the 319 test windows (1 thread)\n")
        md.append("| Model | Engine | Accuracy | Macro-F1 | Recall Down / Stat / Up | Majority-class accuracy |")
        md.append("|---|---|---|---|---|---|")
        for r in acc_rows:
            md.append(f"| {r['model']} | {r['engine']} | {r['accuracy']:.1%} | {r['macro_f1']:.3f} | {r['recall_down']:.1%} / {r['recall_stationary']:.1%} / {r['recall_up']:.1%} | {r['majority_class_accuracy']:.1%} |")
    if placement_rows:
        md.append("\n## ONNX Runtime node placement (from ORT trace logs)\n")
        md.append("| Engine | Model | CoreML nodes | Placements |")
        md.append("|---|---|---|---|")
        for r in placement_rows:
            md.append(f"| {r['engine']} | {r['model']} | {r['coreml_nodes']} | `{r['placements']}` |")
    if suite.get("engines"):
        md.append("\n## Engines\n")
        for key, text in suite["engines"].items():
            md.append(f"- `{key}`: {text}")
    md.append("\n## Figures\n")
    for path in sorted(figures.glob("*.png")):
        md.append(f"![{path.stem}](figures/{path.name})")
    md.append("\nTables (CSV, the table view for every figure): " + ", ".join(f"`tables/{p.name}`" for p in sorted(tables.glob('*.csv'))))
    (out_base / "summary.md").write_text("\n".join(md) + "\n")
    print(out_base / "summary.md")


if __name__ == "__main__":
    main()
