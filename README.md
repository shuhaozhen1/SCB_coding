# SCB_coding

Simultaneous Confidence Bands (SCB) for sparse functional / varying-coefficient
models — implementation and benchmark code.

This repository contains:

* **`scb/`** — the proposed pooled local-linear SCB with leave-one-subject-out
  residuals, dual-bandwidth undersmoothing rule, and multiplier bootstrap.
* **`comparators/`** — faithful re-implementations of the two competitor
  methods used in the numerical study:
  * `cao_2018.py` — Cao, Wang, Wang & Yang (2018), *Statistica Sinica*
    **28**, 2233–2255 (per-target plug-in SCB for sparse longitudinal
    varying-coefficient regression).
  * `caihu_2025.py` — Cai & Hu (2025), *Statistics and Computing* **35**:146
    (B-spline SCB with sandwich covariance; unified sparse-to-dense
    Gaussian-approximation framework).
* **`simulations/`** — three Monte-Carlo experiments (Part 1: single-coef VCM,
  Part 2: joint q=5 VCM, Part 3: mean function), each parallelized over
  reps with `ProcessPoolExecutor`.
* **`realdata/`** — application to the Mayo PBC sequential biomarker dataset:
  one VCM analysis (joint SCB over treatment + age coefficients) and one
  mean-function analysis on log-bilirubin.
* **`data/pbcseq.csv`** — Mayo PBC dataset (Therneau & Grambsch 2000;
  distributed in R `survival::pbcseq`).
* **`tests/`** — pytest smoke tests covering all three method modules.

## Installation

```bash
pip install -r requirements.txt
```

Python ≥ 3.10. Core dependencies: numpy, pandas, scipy, matplotlib, pytest.

## Quick start

Run smoke tests:

```bash
pytest tests/ -q
```

Run a single simulation cell:

```bash
python simulations/part1_vcm.py --n-list 200 --m-list 5 --noise-list gauss --reps 50
```

Full Monte Carlo (R=1000, 6 cores; ≈ 3 hours total):

```bash
python simulations/part1_vcm.py --reps 1000 --workers 6
python simulations/part2_joint.py --reps 1000 --workers 6
python simulations/part3_mean.py  --reps 1000 --workers 6
```

PBC real-data applications:

```bash
python realdata/pbc_vcm.py
python realdata/pbc_mean.py
```

Results are written under `results/` (gitignored).

## Method summary

**Proposed SCB (`scb/`).** For a varying-coefficient model
$Y_{ij} = Z_{ij}^\top \beta(T_{ij}) + \varepsilon_{ij}$:

1. Estimation bandwidth `h_est` by **leave-one-subject CV** on a candidate grid.
2. Pooled local-linear estimator with subject-level weights
   $K_h/(n m_i)$; **leave-one-subject-out residuals** at the inference
   bandwidth.
3. Influence-function multiplier bootstrap with the standardised supremum
   process; bands of the form
   $\widehat\beta_k(t) \pm \widehat q_{\rm crit}\, \widehat{\rm SE}_k(t)$
   for components $k$ in any user-chosen subset (single, multiple, or all).
4. Dual bandwidth rule
   $\hat h_{\rm inf} = \hat h_{\rm est}\,(1 - c/\sqrt n)$
   with calibrated `c` per scenario (Part 1: c=7; Part 2: c=6.5; Part 3:
   c=4.5).

**Comparators.** See `comparators/cao_2018.py` and `comparators/caihu_2025.py`
for paper-faithful implementations.

## Repository structure

```
SCB_coding/
├── scb/                          # proposed SCB package
│   ├── __init__.py
│   ├── config.py                 # BootstrapConfig, LocalLinearConfig, BandwidthCVConfig
│   ├── kernels.py
│   ├── estimation.py             # local_linear_vcm_fit
│   ├── inference.py              # local_linear_vcm_scb (the SCB)
│   └── bandwidth.py              # LOSO / k-fold CV
├── comparators/
│   ├── __init__.py
│   ├── cao_2018.py
│   └── caihu_2025.py
├── data/
│   └── pbcseq.csv
├── simulations/
│   ├── _common.py                # DGPs + parallel utilities
│   ├── part1_vcm.py
│   ├── part2_joint.py
│   └── part3_mean.py
├── realdata/
│   ├── pbc_vcm.py
│   └── pbc_mean.py
├── tests/
│   └── test_smoke.py
├── requirements.txt
├── LICENSE                       # MIT
└── README.md
```

## License

MIT. See `LICENSE`.

## References

* Cai, T. T. and Hu, J. (2025). From sparse to dense functional data: phase
  transitions from a simultaneous inference perspective. *Statistics and
  Computing* **35**, Article 146.
  https://doi.org/10.1007/s11222-025-10671-1
* Cao, G., Wang, S., Wang, Y. and Yang, L. (2018). Simultaneous nonparametric
  regression analysis of sparse longitudinal data. *Statistica Sinica*
  **28**, 2233–2255.
* Therneau, T. M. and Grambsch, P. M. (2000). *Modeling Survival Data:
  Extending the Cox Model*. Springer. (Mayo PBC sequential biomarker dataset,
  distributed as `pbcseq` in the R `survival` package.)
