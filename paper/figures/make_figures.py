"""Generate the paper's figures from the final benchmark run.

Usage (repository root):
  bench/.venv/bin/python -I paper/figures/make_figures.py

Reads results/phase2/20261122-final-rust-vs-cpp/raw/ (per-process JSON reports, kept locally)
and writes PDF figures sized for the IEEE two-column format into paper/figures/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "results/phase2/20261122-final-rust-vs-cpp"
OUT = Path(__file__).resolve().parent

# Same validated categorical slots as bench/analyze.py: C++ = aqua, Rust = blue; the
# same-binary Rust control is a neutral dark gray. Markers give secondary encoding.
COLOR = {"cpp": "#1baf7a", "rust-msdylib": "#52514e", "rust": "#2a78d6"}
MARKER = {"cpp": "^", "rust-msdylib": "D", "rust": "o"}
LABEL = {"cpp": "C++ (Microsoft ORT)", "rust-msdylib": "Rust (Microsoft ORT)", "rust": "Rust (ort-bundled ORT)"}
ENGINES = ["cpp", "rust-msdylib", "rust"]
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": BASE, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.4, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
RNG = np.random.default_rng(7)


def load() -> dict:
    """(engine, model, threads) -> list of (per-call samples, session-run samples), cycled inputs, default graph opt."""
    groups: dict = {}
    for engine in ENGINES:
        for path in sorted((RUN / "raw" / engine).glob("*.json")):
            report = json.loads(path.read_text())
            meta = report["metadata"]
            options = meta.get("session_options", "")
            if meta["mode"] != "inference" or meta.get("input_policy") != "cycle":
                continue
            if "graph_optimization=ort-default" not in options:
                continue
            threads = re.search(r"intra_op_threads=([^;]+)", options).group(1)
            threads = "1" if threads == "1" else "default"
            trial = report["trials"][0]
            groups.setdefault((engine, meta["model"], threads), []).append(np.asarray(trial["samples_ms"]))
    return groups


def boot_ci(values: np.ndarray) -> tuple[float, float]:
    draws = np.median(RNG.choice(values, size=(10000, values.size)), axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def fig_latency(groups: dict) -> None:
    models = ["DeepLOB", "TransformerLOB"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.15))
    width = 0.26
    for ax, model in zip(axes, models):
        for k, threads in enumerate(("1", "default")):
            cpp = np.median([np.median(s) for s in groups[("cpp", model, threads)]])
            for e, engine in enumerate(ENGINES):
                medians = np.array([np.median(s) for s in groups[(engine, model, threads)]])
                x = k + (e - 1) * width
                ax.scatter(x + (RNG.random(medians.size) - 0.5) * width * 0.55, medians, s=7,
                           color=COLOR[engine], marker=MARKER[engine], linewidths=0, alpha=0.8, zorder=3)
                lo, hi = boot_ci(medians)
                med = np.median(medians)
                ax.plot([x - width * 0.32, x + width * 0.32], [med, med], color=INK, lw=1.3, zorder=4)
                ax.plot([x, x], [lo, hi], color=INK, lw=0.9, zorder=4)
                text = f"{med:.3f}" if engine == "cpp" else f"{med:.3f}\n×{med / cpp:.3f}"
                ax.text(x, medians.max() + 0.02, text, ha="center", va="bottom", fontsize=6.2, color=INK2,
                        linespacing=0.95)
        ax.set_xticks([0, 1], ["1 intra-op thread", "ORT default threads"])
        ax.set_title(model, fontweight="bold")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.55, 1.55)
    axes[0].set_ylabel("Latency per process (ms)")
    handles = [Line2D([0], [0], marker=MARKER[e], color=COLOR[e], linestyle="none", markersize=4.5, label=LABEL[e])
               for e in ENGINES]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.5)
    fig.savefig(OUT / "fig_latency.pdf")
    plt.close(fig)


def fig_ecdf(groups: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75), sharey=True)
    for ax, threads in zip(axes, ("1", "default")):
        samples = {e: np.sort(np.concatenate(groups[(e, "TransformerLOB", threads)])) for e in ENGINES}
        for engine, x in samples.items():
            ax.plot(x, np.arange(1, x.size + 1) / x.size, color=COLOR[engine], lw=1.1)
        # x-axis capped at the largest P99.9 so the body of the distributions stays readable.
        lo = min(x[0] for x in samples.values()) * 0.97
        hi = max(np.quantile(x, 0.999) for x in samples.values()) * 1.03
        ax.set_xlim(lo, hi)
        ax.axhline(0.99, color=BASE, lw=0.7, ls=(0, (3, 2)))
        ax.text(hi, 0.985, "P99 ", fontsize=6, color=MUTED, ha="right", va="top")
        ax.set_title("1 thread" if threads == "1" else "ORT default threads")
        ax.set_xlabel("Per-call latency (ms)")
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_ylabel("Fraction of calls")
    handles = [Line2D([0], [0], color=COLOR[e], lw=1.4, label=LABEL[e]) for e in ENGINES]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), fontsize=5.8,
               handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.6)
    fig.savefig(OUT / "fig_ecdf.pdf")
    plt.close(fig)


def main() -> None:
    groups = load()
    fig_latency(groups)
    fig_ecdf(groups)
    print("wrote", sorted(p.name for p in OUT.glob("fig_*.pdf")))


if __name__ == "__main__":
    main()
