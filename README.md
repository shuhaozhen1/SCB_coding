# SCB_coding

Simultaneous Confidence Bands (SCB) for sparse functional / varying-coefficient
models — implementation, Monte-Carlo benchmark, and a real-data application.

The full written-up results are in **[`docs/numerical_study.md`](docs/numerical_study.md)**.
All generated tables, figures, and the consolidated Excel workbook are tracked
under **[`results/`](results/)**.

---

## What's here

* **`scb/`** — the proposed pooled local-linear SCB: leave-one-subject-out
  residuals, a single undersmoothing rule, and an influence-function
  multiplier bootstrap. Works for a single coefficient, a chosen subset, or a
  genuinely *joint* band over all coefficients.
* **`comparators/`** — paper-faithful re-implementations of the two competitors:
  * `cao_2018.py` — Cao, Wang, Wang & Yang (2018), *Statistica Sinica* **28**,
    2233–2255 (per-target plug-in SCB with Gumbel critical value).
  * `caihu_2025.py` — Cai & Hu (2025), *Statistics and Computing* **35**:146
    (B-spline SCB with sandwich covariance).
* **`simulations/`** — the three Monte-Carlo experiments plus the plotting and
  Excel-export pipeline (see below).
* **`realdata/`** — the Mayo PBC application: data illustration, mean-function
  analysis, and varying-coefficient analysis on log-bilirubin.
* **`data/pbcseq.csv`** — Mayo PBC dataset (Therneau & Grambsch 2000;
  R `survival::pbcseq`).
* **`results/`** — committed deliverables: simulation/real-data figures,
  per-cell summary tables (CSV), and `simulation_results.xlsx`.
* **`docs/numerical_study.md`** — the final write-up (§6 Numerical Study).
* **`tests/`** — pytest smoke tests covering all three method modules.

---

## The undersmoothing rule (single, unified)

The estimation bandwidth `h_est` is chosen by leave-one-subject-out CV. The
inference bandwidth is obtained by **one rule shared across all settings**:

```
h_inf = h_est * ( 1 - c / ( n^(1/2) * (m_bar / q)^(1/4) ) ),    c = 8
```

* `n` — number of subjects; `m_bar` — mean observations per subject;
* `q` — model dimension (number of coefficient functions): `q=1` mean
  function, `q=2` single-covariate VCM, `q=5` joint 5-coefficient VCM.

The denominator `n^(1/2) (m_bar/q)^(1/4)` is the per-coefficient effective
resolution: `n^(1/2)` carries the subject-level undersmoothing-for-inference
rate (Calonico–Cattaneo–Titiunik 2014), while `(m_bar/q)^(1/4)` mildly weakens
the undersmoothing as the design gets denser and strengthens it as the
coefficient vector grows, consistent with `n·m_bar` being the effective sample
size for sparse functional data (Yao–Müller–Wang 2005; Zhang–Wang 2016). The
ratio `h_inf/h_est → 1` as `n → ∞`. The single constant `c = 8` is calibrated
once and held fixed everywhere (see `simulations/_common.undersmooth_h_inf`).

---

## Installation

```bash
pip install -r requirements.txt
```

Python ≥ 3.10. Core dependencies: numpy, pandas, scipy, matplotlib, openpyxl,
pytest.

---

## Reproducing everything

The pipeline has four stages. `R=1000` full runs take a few hours on 6–8 cores;
use small `--reps` for a quick check.

**1. Monte-Carlo simulation** (writes per-cell summaries to `results/part*/`):

```bash
python simulations/part1_vcm.py        --reps 1000 --workers 6   # q=2 VCM  vs Cao 2018
python simulations/part2_joint.py      --reps 1000 --workers 6   # q=5 joint VCM
python simulations/part2_bonferroni.py --reps 1000 --workers 8   # Bonferroni-Cao comparator for Part 2
python simulations/part3_mean.py       --reps 1000 --workers 6   # q=1 mean  vs Cai-Hu 2025 (n in {200,300})
```

**2. Real-data application** (writes to `results/realdata/`):

```bash
python realdata/pbc_data_illustration.py   # Figure: trajectories, sparsity, log transform
python realdata/pbc_mean.py                # mean log-bilirubin SCB vs Cai-Hu
python realdata/pbc_vcm.py                 # joint VCM SCB (treatment + age) vs Cao
```

**3. Illustration figures** (representative replicate per setting):

```bash
python simulations/gen_plot_curves.py      # -> results/plotdata/*.npz (intermediate)
python simulations/plot_bands.py           # -> results/figures/*.png (5 panel figures)
```

**4. Consolidated Excel workbook**:

```bash
python simulations/build_excel.py          # -> results/simulation_results.xlsx
```

Quick smoke check (seconds):

```bash
pytest tests/ -q
python simulations/part1_vcm.py --n-list 200 --m-list 5 --noise-list gauss --reps 20
```

---

## Results at a glance (R = 1000)

| Part | model | comparator | Ours mean cov / width | comparator mean cov / width |
|------|-------|------------|-----------------------|-----------------------------|
| 1 | q=2 VCM, marginal $\beta_1$ | Cao 2018          | 0.945 / 0.822 | 0.996 / 1.292 |
| 2 | q=5 VCM, joint              | Bonferroni-Cao    | 0.962 / 0.956 | 1.000 / 1.752 |
| 3 | q=1 mean function           | Cai–Hu 2025       | 0.954 / 0.849 | 0.926 / 0.703 |

All 39 simulation cells have coverage in `[0.931, 0.967]`. See
`results/simulation_results.xlsx` (sheets: Notes, Aggregates, Part1/2/3) and
`docs/numerical_study.md` for the full tables, figures, and discussion.

Figures (`results/figures/`, `results/realdata/`):
`part{1,3}_bands_n{200,300}.png`, `part2_bands_joint.png`,
`pbc_data_illustration.png`, `pbc_mean_bands.png`, `pbc_vcm_bands.png`.

---

## Repository structure

```
SCB_coding/
├── scb/                          # proposed SCB package
│   ├── config.py                 # BootstrapConfig, LocalLinearConfig, ...
│   ├── kernels.py
│   ├── estimation.py             # local_linear_vcm_fit
│   ├── inference.py              # local_linear_vcm_scb (the SCB)
│   └── bandwidth.py              # LOSO / k-fold CV
├── comparators/
│   ├── cao_2018.py
│   └── caihu_2025.py
├── data/
│   └── pbcseq.csv
├── simulations/
│   ├── _common.py                # DGPs, parallel utils, undersmooth_h_inf rule
│   ├── part1_vcm.py              # Part 1: q=2 VCM vs Cao
│   ├── part2_joint.py            # Part 2: q=5 joint VCM
│   ├── part2_bonferroni.py       # Part 2 comparator: Bonferroni-Cao
│   ├── part3_mean.py             # Part 3: q=1 mean vs Cai-Hu (n in {200,300})
│   ├── gen_plot_curves.py        # representative band curves -> plotdata
│   ├── plot_bands.py             # panel illustration figures
│   └── build_excel.py            # consolidated Excel workbook
├── realdata/
│   ├── pbc_data_illustration.py
│   ├── pbc_mean.py
│   └── pbc_vcm.py
├── results/                      # tracked deliverables (figures, tables, xlsx)
│   ├── figures/                  # simulation band figures
│   ├── part1/ part2/ part3/ part2_bonferroni/   # *_summary.csv tables
│   ├── realdata/                 # real-data figures + summaries + band CSVs
│   └── simulation_results.xlsx
├── docs/
│   └── numerical_study.md        # final write-up (§6)
├── tests/
├── requirements.txt
├── LICENSE                       # MIT
└── README.md
```

---

## License

MIT. See `LICENSE`.

## References

* Cai, T. T. and Hu, J. (2025). From sparse to dense functional data: phase
  transitions from a simultaneous inference perspective. *Statistics and
  Computing* **35**, Article 146. https://doi.org/10.1007/s11222-025-10671-1
* Calonico, S., Cattaneo, M. D. and Titiunik, R. (2014). Robust nonparametric
  confidence intervals for regression-discontinuity designs. *Econometrica*
  **82**(6), 2295–2326.
* Cao, G., Wang, S., Wang, Y. and Yang, L. (2018). Simultaneous nonparametric
  regression analysis of sparse longitudinal data. *Statistica Sinica*
  **28**, 2233–2255.
* Therneau, T. M. and Grambsch, P. M. (2000). *Modeling Survival Data:
  Extending the Cox Model*. Springer.
* Yao, F., Müller, H.-G. and Wang, J.-L. (2005). Functional data analysis for
  sparse longitudinal data. *J. Amer. Statist. Assoc.* **100**, 577–590.
* Zhang, X. and Wang, J.-L. (2016). From sparse to dense functional data and
  beyond. *Ann. Statist.* **44**(5), 2281–2321.
