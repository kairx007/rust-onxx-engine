# Paper (IEEE conference format)

`main.tex` is the paper; `conference_101719.tex`, `IEEEtran.cls` and `IEEEtran_HOWTO.pdf` are the unmodified IEEE
template files it is based on. Page limit: 6 pages (current length: 4).

Build (Tectonic, `brew install tectonic`):

```sh
cd paper
tectonic -X compile main.tex --outdir build      # -> build/main.pdf
```

Figures are generated from the final benchmark run (raw reports are kept locally under
`results/phase2/20261122-final-rust-vs-cpp/raw/`):

```sh
bench/.venv/bin/python -I paper/figures/make_figures.py   # from the repository root
```

All numbers in the paper come from `results/phase2/20261122-final-rust-vs-cpp/` (`summary.md`,
`tables/paper_table_latency.csv`) and `results/data_checks/report.md`; see `research/08_final_results.md`.
