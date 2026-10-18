# Reproduction code for functional simultaneous confidence bands

This directory is the self-contained code release for the current manuscript
**Simultaneous Confidence Bands for Functional Varying-Coefficient Models**
(Haozhen Shu, Zhenhua Lin and Yinan Lin). Run all commands from this directory.
The repository's older top-level implementations are historical and are not the
entry points for the current results.

## Install and check

Python 3.11 or newer is required; the recorded production environment is Python
3.12.14. NumPy, SciPy, pandas, matplotlib and threadpoolctl are the numerical
requirements. No R, network access, local manuscript folder or private data is
needed after downloading this directory.

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python reproduce.py check
python -m pytest tests -q
```

`requirements-lock.txt` records the exact package versions used for production
and delivery checks. `requirements.txt` gives compatible version ranges when a
locked wheel is unavailable. Tables rebuilt from saved records are exact on
both installations. Seed replays are checked with numerical tolerances, because
linear algebra and finite Gaussian calibration can depend on the package/BLAS
build; production versions are recommended for comparing individual replays.

`check` verifies the release's SHA-256 manifest, checks the unchanged numerical
engines, recomputes **all 90 cells from 72,000 independent datasets**, compares
coverage and widths with the current paper summary, and regenerates the four
LaTeX simulation tables byte for byte. Generated files go under `generated/`;
retained evidence under `reference/` is never overwritten.

## Rebuild tables and figures without rerunning Monte Carlo

```bash
python reproduce.py tables
python reproduce.py figures
```

The figures command redraws the two simulation illustrations, the three PBC
band comparisons and the PBC observation-schedule figure. Numerical plotting
arrays and critical values are retained without rounding. Simulation figures
span `[0,1]`; the high-dimensional illustration displays coordinates 1, 50,
100 but retains calibration over all 100 coordinates. The illustrations were
chosen as the first jointly covered replication in a fixed order, not as
unconditional typical outcomes; this selection does not alter table results.

## Replay original random seeds

```bash
# Twelve sparse representative datasets: four targets times three laws.
python reproduce.py replay --workers 2
# One original replication in every design group: 72 independent datasets.
python reproduce.py replay --all-cells --workers 4
# Also regenerate both full-domain simulation figure datasets and bands.
python reproduce.py figures --regenerate-simulations
# Refit all three real-data analyses, including the comparator calibrations.
python reproduce.py realdata
```

Replay compares coverage indicators exactly and numerical fields at
`rtol=5e-7, atol=5e-8`. Real-data reconstruction uses the original 10,000 draws,
201-point full-domain grid, matched CLZ regression bandwidth and fixed
`1.25*h0` mean bandwidth. It also checks the retained estimates, standard
errors and critical values. Output figures may differ at the font-rendering
or PDF metadata level across systems even when numerical arrays agree.

## Run the complete simulation from scratch

```bash
# Small executable check; these new results do not replace paper evidence.
python reproduce.py simulate --reps 2 --n 200 --m 5 --laws normal --workers 2
# Full paper design. Use a different output folder from the small check.
python reproduce.py simulate --reps 1000 --workers 8 --output generated/full
```

The full design uses `n=200,400`, `m=5,20,80`, and Gaussian, variance-one
Laplace and adjusted chi-square-5 innovations. Targets are one coefficient in
a two-coefficient VCM (`single`), all five coefficients (`joint`), a scalar
mean (`scalar`), and coordinate-specific-error means with nested `p=50,100`
(`highd`). All cells are retained, including failures, which count as
noncoverage; widths are averaged over valid fits with failure counts recorded.

All proposed estimation and residual fits are **local linear**, using all
observations on `[0,1]`. Simulation coverage is the joint event on **51 points
in `[0.15,0.85]`**, not a claim of measured continuous-time coverage. Gaussian
multipliers are shared within subjects and across coordinates. Coefficient and
scalar-mean bands use 2,000 multipliers; high-dimensional means use 1,000.

The common base bandwidth is
`h0 = max(n**(-1/4), (n*mH)**(-1/5)) / sqrt(1+log(n))`.
For `m=5,20,80`, frozen proposed constants are `(1.25,.8,.65)` for the single
coefficient and `(1.5,1.25,.65)` for the joint VCM. They came from the separate
Gaussian pilot described in the manuscript. The scalar mean retains `h0`
after comparing an alternative rule on validation outcomes; the
high-dimensional mean rule is fixed at `h0`. No rule is selected again in
this release.

CLZ uses prediction-based exact leave-one-subject-out CV over seven candidates
from `.0601` to `.0899`, with the chosen bandwidth also used for residuals and
both matrix moments. The normal-reference density bandwidth is separate.
Reference quantiles use 200,000 draws for each `(n,m,h)`; levels `.95` and `.99`
calibrate the single and Bonferroni five-coefficient targets. No extra CLZ
undersmoothing or outcome-based bandwidth selector is introduced. There is no
CLZ comparator for the high-dimensional mean. The scalar mean comparator is
the cubic B-spline Cai--Hu procedure, with BIC-selected knots and subject-score
covariance. See `code/study.py` for the actual execution paths.

The retained CLZ reference quantiles can themselves be regenerated:

```bash
python reproduce.py calibrate-clz --draws 200000 --workers 8
```

This uses the original 1,000-draw chunk seeds, shares times and Gaussian marks
across the seven bandwidths in each draw, and verifies all 84 saved quantiles.
The complete calibration and simulation are computationally substantial;
`--workers` controls process parallelism and BLAS threads are limited to one
per worker. Interrupted simulation runs resume only under the same run
configuration and source hashes. Use a new output folder after changing a
configuration or the code.

## What is retained

- `code/`: 15 unchanged numerical modules from the verified implementation,
  plus the portable current-study driver and drawing helpers. The public entry
  point is `reproduce.py`; older alternative methods present as unused helper
  functions are not called by it.
- `reference/replications.jsonl.gz`: a lossless canonical projection of the
  current proposed/comparator fields for all 72,000 independent datasets.
  Historical alternatives, execution timers and machine-specific checkpoints
  are omitted. Nested dimensions share their original data and multipliers.
- `reference/paper-summary.csv`, `reference/tables/`: the authoritative current
  paper summaries and four table snapshots.
- `reference/clz-critical-values.json`: the frozen candidate-specific
  reference quantiles, with their seeds implemented in the driver.
- `reference/plotdata/`: compact, full-precision curve arrays and figure
  metadata. Full simulation observations and Gaussian multipliers regenerate
  from the original seeds; no large cache or external download is required.
- `reference/PROVENANCE.json`: source-record and original plotting-file hashes,
  the canonical projection rule and dataset counts.
- `data/`: public `survival::pbcseq` snapshot, provenance and upstream notice.
- `tests/`: direct weighted-local-fit checks, innovation checks,
  coordinate-specific versus independent scalar fits, CLZ formula/deletion
  checks, and reconstruction of every manuscript simulation table.
- `MANIFEST.json`: hashes of release inputs. Generated output is excluded.

The coefficient/scalar seed is
`SeedSequence([20261018,771211,2,n,m,p,law_index,rep]).spawn(3)`
(data, multipliers, spline calibration). The high-dimensional seed is
`SeedSequence([20261018,184700,n,m,law_index,rep]).spawn(4)`
(times, common innovations, coordinate-specific innovations, multipliers).
The two sets of datasets are independent. There is one functional error per
response coordinate and no added observation-level measurement error.

PBC source documentation:
<https://stat.ethz.ch/R-manual/R-devel/library/survival/html/pbcseq.html>.
The real-data curves describe retained-visit associations and means; the CLZ
full-domain extension and density-normalization sensitivity are retained in
its code/metadata. Band widths alone do not establish comparative coverage.

Code is distributed under the repository's MIT license. Data provenance and
the upstream notice are supplied separately in `data/`.
