# Functional simultaneous confidence bands

The current manuscript's self-contained reproduction release is in
**[`reproducibility/`](reproducibility/README.md)**.

It contains the current local-linear varying-coefficient and functional-mean
procedures, CLZ and Cai--Hu comparisons, the public PBC snapshot, original
random seeds, full-precision plotting inputs, and compressed records for all
72,000 independent simulation datasets. One command rebuilds all 90 design
cells and the four manuscript simulation tables from the retained records.

```bash
cd reproducibility
python -m pip install -r requirements-lock.txt
python reproduce.py check
python -m pytest tests -q
```

For complete Monte Carlo reconstruction, seed replays, figure regeneration and
real-data analysis, follow the [reproduction instructions](reproducibility/README.md).
New output is written under `reproducibility/generated/`; reference results are
not overwritten. The main studies target one coefficient, five coefficients
jointly, a scalar mean, and high-dimensional means with coordinate-specific
functional errors and nested dimensions 50 and 100.

The older top-level `scb/`, `comparators/`, `simulations/`, `realdata/`,
`results/` and `manuscript/` directories are retained as historical versions.
Their former commands, formulas and summaries are **not** the current paper's
reproduction pipeline. The [previous README](docs/legacy_implementation_README.md)
is retained for that history.

Code license: [MIT](LICENSE). Public data provenance and upstream notice are
included in [`reproducibility/data/`](reproducibility/data/).
