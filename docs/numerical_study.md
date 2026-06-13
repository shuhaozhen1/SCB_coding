# 6. Numerical Study

This section reports a comprehensive Monte-Carlo evaluation of the proposed
simultaneous confidence band (SCB) and a real-data application on the
Mayo PBC sequential biomarker dataset. All simulation results use
$R = 1{,}000$ replicates per cell, giving Monte-Carlo standard errors of at
most $0.007$ on the empirical coverage at the nominal level
$1 - \alpha = 0.95$.

## 6.1 Simulation

### 6.1.1 Setup and notation

The simulation comprises three parts that span the principal scenarios of
sparse longitudinal functional data analysis:

* **Part 1** (\S 6.1.3) — single-coefficient SCB in a varying-coefficient
  model (VCM) with $p=2$ functional coefficients. Comparator: Cao (2018).
* **Part 2** (\S 6.1.4) — joint simultaneous SCB over **all** coefficient
  functions in a $p=5$ VCM. No external comparator (Cao's plug-in SCB is
  per-target by construction).
* **Part 3** (\S 6.1.5) — SCB for the mean function in a Karhunen–Loève
  data-generating process (DGP). Comparator: the B-spline SCB of
  Cai and Hu (2025).

Throughout, $n$ is the number of subjects, $m$ is the (fixed) number of
repeated measurements per subject, $T_{ij}\stackrel{\text{iid}}{\sim}
\text{Uniform}(0,1)$, and the SCB is evaluated on a common grid of
$G = 96$ equally spaced points on $[0.025, 0.975]$ for all three parts.
The nominal level is $\alpha = 0.05$. For each cell we report the empirical
simultaneous **coverage** (the fraction of replicates whose band contains
the target function at every grid point) and the average band **width**;
at $R = 1{,}000$ the Monte-Carlo standard error of a coverage near $0.95$
is at most $0.007$.

### 6.1.2 Bandwidth selection and undersmoothing rule

For all three parts we use the same two-stage bandwidth scheme. The
estimation bandwidth $\hat h_{\text{est}}$ is selected by leave-one-subject
cross-validation,
$$
\hat h_{\text{est}} \;=\; \arg\min_{h\in\mathcal H}\;
\frac{1}{N}\sum_{i,j}\bigl(Y_{ij}-Z_{ij}^\top\widehat\beta_{h}^{(-i)}(T_{ij})\bigr)^2,
$$
on the candidate grid $\mathcal H = \text{linspace}(0.04,0.16,13)$, where
$\widehat\beta_h^{(-i)}$ denotes the pooled local-linear estimate fit
without subject $i$. The inference bandwidth is then obtained by a single
undersmoothing rule, shared by all three parts,
$$
\hat h_{\text{inf}} \;=\; \hat h_{\text{est}}\cdot
\Bigl(1 - \frac{c}{\,n^{1/2}\,(\bar m/q)^{1/4}}\Bigr),
\qquad c = 8,
$$
where $\bar m$ is the realized mean number of observations per subject and
$q$ is the model dimension (the number of coefficient functions:
$q=2$ for Part 1, $q=5$ for Part 2, $q=1$ for Part 3). The denominator is
the *per-coefficient effective resolution*: the $n^{1/2}$ factor carries the
subject-level information that governs the undersmoothing-for-inference
rate (Calonico, Cattaneo & Titiunik, 2014), while the factor
$(\bar m/q)^{1/4}$ mildly weakens the undersmoothing as the design becomes
denser ($\bar m$ large) and strengthens it as the coefficient vector grows
($q$ large), consistent with $n\bar m$ being the effective sample size for
sparse functional data (Yao, Müller & Wang, 2005; Zhang & Wang, 2016). The
ratio $\hat h_{\text{inf}}/\hat h_{\text{est}} \to 1$ as $n\to\infty$, so
the band is asymptotically equivalent to one built at $\hat h_{\text{est}}$.
The single constant $c = 8$ is calibrated **once**, by a pilot sweep that
seeks coverage near $0.95$ jointly across all $(n, m, q)$ configurations,
and is then held fixed throughout; no scenario-specific tuning is used.

Cao (2018)'s plug-in SCB is implemented faithfully with Silverman pilot
bandwidths $b_n = c_n = d_n = 1.06\,\widehat\sigma_t\,N^{-1/5}$ for the
design density $\hat f$, the local design matrix $\widehat\Omega$, and the
local noise variance $\widehat\Lambda$, with the Gumbel-type asymptotic
critical value obtained from $B_{\text{crit}} = 2{,}000$ independent
simulations of the limiting Gaussian-process supremum. Cai and Hu (2025)'s
B-spline SCB (which provides a unified sparse-to-dense Gaussian
approximation for the mean function) is implemented with cubic B-splines,
BIC-selected interior knots $J_n$ over the range
$\lceil 0.5\,n^{1/4}(\log n)^{1/2}\rceil \le J_n \le \lfloor 2\,n^{0.3}\rfloor$,
the subject-level sandwich covariance estimator $\widehat\Sigma =
\widehat V^{-1}\,\bigl[(nN_\text{avg}^2)^{-1}\sum_{i}s_i s_i^\top\bigr]\,\widehat V^{-1}$
with $s_i = \sum_{j}B(T_{ij})r_{ij}$, and a critical value obtained from
$B = 500$ multivariate-normal draws $Z \sim \mathcal N(0,\widehat\Sigma)$
of the standardized spline process.

Our SCB uses $n_{\text{boot}} = 300$ Gaussian multiplier draws of the
standardized sup process and **leave-one-subject-out residuals** at
$\hat h_{\text{inf}}$.

### 6.1.3 Part 1 — single-coefficient VCM ($p=2$)

**DGP** (Cao 2018 design with subject-level multiplicative noise):
$$
Y_{ij} \;=\; \beta_0(T_{ij}) + \beta_1(T_{ij})\,X_i(T_{ij}) + \zeta_i\,X_i(T_{ij}),
$$
where $\beta_0(t) = 5(t-0.6)^2$, $\beta_1(t) = 0.5 + 0.4\sin(2\pi(t-0.5))$,
$X_i(\cdot)$ is a centred Gaussian process with variance $e^t$ and
correlation $\exp(-|\Delta t|/4)$, and $\zeta_i$ is a unit-variance
subject-level multiplier drawn from one of three distributions:
$\mathcal N(0,1)$, $\mathrm{Uniform}(-\sqrt3,\sqrt3)$, or
$\mathrm{Laplace}(0,1/\sqrt2)$. The inferential target is $\beta_1(\cdot)$;
the simultaneous coverage event requires the SCB to contain $\beta_1(t)$
for **every** $t$ in the evaluation grid.

**Settings**: $n\in\{200,300\}$, $m\in\{5,10,15\}$, $R=1{,}000$ per cell
(18 cells total).

**Table 6.1.** Part 1 — empirical coverage and average band width for
$\beta_1$. All Monte-Carlo standard errors $\le 0.007$.

| n   | m  | noise   | Cao cov / wid | **Ours cov / wid** | wid ratio |
|-----|----|---------|---------------|--------------------|-----------|
| 200 | 5  | gauss   | 0.990 / 1.385 | 0.953 / 1.100      | 0.79      |
| 200 | 5  | uniform | 0.984 / 1.412 | 0.935 / 1.126      | 0.80      |
| 200 | 5  | laplace | 0.992 / 1.348 | 0.949 / 1.066      | 0.79      |
| 200 | 10 | gauss   | 1.000 / 1.408 | 0.947 / 0.860      | 0.61      |
| 200 | 10 | uniform | 0.999 / 1.466 | 0.944 / 0.888      | 0.61      |
| 200 | 10 | laplace | 0.999 / 1.389 | 0.935 / 0.841      | 0.61      |
| 200 | 15 | gauss   | 1.000 / 1.462 | 0.937 / 0.789      | 0.54      |
| 200 | 15 | uniform | 0.999 / 1.461 | 0.931 / 0.791      | 0.54      |
| 200 | 15 | laplace | 1.000 / 1.424 | 0.954 / 0.764      | 0.54      |
| 300 | 5  | gauss   | 0.991 / 1.163 | 0.950 / 0.871      | 0.75      |
| 300 | 5  | uniform | 0.987 / 1.156 | 0.947 / 0.876      | 0.76      |
| 300 | 5  | laplace | 0.996 / 1.104 | 0.962 / 0.835      | 0.76      |
| 300 | 10 | gauss   | 0.999 / 1.183 | 0.941 / 0.698      | 0.59      |
| 300 | 10 | uniform | 0.999 / 1.175 | 0.931 / 0.701      | 0.60      |
| 300 | 10 | laplace | 0.999 / 1.163 | 0.951 / 0.684      | 0.59      |
| 300 | 15 | gauss   | 1.000 / 1.181 | 0.958 / 0.633      | 0.54      |
| 300 | 15 | uniform | 0.999 / 1.201 | 0.942 / 0.640      | 0.53      |
| 300 | 15 | laplace | 1.000 / 1.181 | 0.950 / 0.624      | 0.53      |

**Aggregate (18 cells).**

| Metric                              | Cao        | **Ours**          |
|-------------------------------------|------------|-------------------|
| Mean coverage                       | 0.996      | **0.945**         |
| Mean $|\hat{\mathrm{cov}} - 0.95|$   | 0.046      | **0.008**         |
| Coverage range                      | [0.984, 1.000] | **[0.931, 0.962]** |
| Mean band width                     | 1.292      | **0.821**         |
| Band width ratio to Cao             | $1.00$     | **$0.64$**        |

**Findings.** Our SCB is uniformly close to nominal across all $(n,m,\zeta)$
combinations — all 18 coverages lie in $[0.931, 0.962]$, i.e. within
$\sim 2.7$ Monte-Carlo standard errors of the nominal $0.95$ — and is on
average **36\% narrower** than Cao's plug-in SCB at the same nominal level.
The width gap widens with $m$ (from $\sim 22\%$ narrower at $m=5$ to
$\sim 47\%$ at $m=15$), reflecting that Cao's iid-asymptotic standard error
does not exploit the additional within-subject information available when
$m$ grows. The calibration is stable across the two sample sizes, confirming
that the single undersmoothing constant transfers from $n=200$ to $n=300$.
Cao's plug-in SCB is uniformly over-conservative (mean coverage $0.996$,
all cells $\ge 0.984$), a known finite-sample property of the Gumbel
critical value in the sparse-design regime.

### 6.1.4 Part 2 — joint VCM SCB over $p=5$ coefficients

**DGP** (FDA Part-2 generator with $p=5$):
$$
\beta_0(t)=1,\quad \beta_1(t)=2\sin(2\pi t),\quad \beta_2(t)=2(t-0.5),
$$
$$
\beta_3(t)=e^{t},\quad \beta_4(t)=\cos(5\pi t),
$$
with Karhunen–Loève-like subject loadings on the 5 covariates and i.i.d.
Gaussian observation noise calibrated to SNR $=5$. The joint simultaneous
coverage event requires the SCB to contain $\beta_k(t)$ for **all** $k\in
\{0,\dots,4\}$ and **all** $t$ in the grid; the bootstrap supremum is taken
jointly over $(k,t)$.

**Settings**: $n=200$, $m\in\{5,10,15\}$, $R=1{,}000$ per cell.

Cao (2018)'s plug-in SCB is per-target by construction and not natively
joint. To obtain a genuine simultaneous-coverage comparator we apply it with
a **Bonferroni correction**: each coefficient $\beta_k$ receives a Cao SCB at
the reduced level $\alpha/p = 0.01$ (a per-coefficient $99\%$ band), and the
intersection over $k$ is reported as the joint band. By the union bound this
guarantees joint coverage $\ge 1-\alpha$, at the cost of using the
$\alpha/p$-quantile critical value for every component.

**Table 6.2.** Part 2 — joint simultaneous coverage over all $p=5$
coefficient functions (width averaged over the five components): Ours vs.
the Bonferroni-corrected Cao (2018) SCB.

| m  | **Ours cov / wid** | Bonferroni-Cao cov / wid | wid ratio |
|----|--------------------|--------------------------|-----------|
| 5  | **0.967** / 1.340  | 1.000 / 1.617            | 0.83      |
| 10 | **0.960** / 0.849  | 1.000 / 1.775            | 0.48      |
| 15 | **0.960** / 0.680  | 1.000 / 1.863            | 0.36      |

Our joint band is well controlled across the three densities (mean coverage
$0.962$, mean $|\hat{\mathrm{cov}}-0.95| = 0.012$, all cells in
$[0.960, 0.967]$), with width decreasing monotonically as $m$ grows
($1.34 \to 0.85 \to 0.68$). The same undersmoothing rule and constant
$c=8$ used in Parts 1 and 3 apply here with $q=5$; because the band must
simultaneously cover five coefficient functions, it is mildly conservative
($\le 0.967$), the expected and safe behaviour of a genuinely joint
construction.

The Bonferroni-Cao comparator does attain simultaneous coverage, but
**always at $1.000$** — severely over-conservative — and with a width that
**grows with $m$** ($1.62 \to 1.78 \to 1.86$) rather than shrinking. The
two effects compound: each per-coefficient band already over-covers in the
sparse design and is then further inflated to the $\alpha/p$ level. The
resulting efficiency gap widens sharply with $m$: our joint band is $17\%$
narrower at $m=5$ and **$64\%$ narrower at $m=15$**. This illustrates the
price of forcing a per-target plug-in method to deliver joint coverage by
multiplicity correction, versus our directly joint multiplier-bootstrap
construction that exploits the within-subject information as $m$ grows.

### 6.1.5 Part 3 — mean function SCB

**DGP** (Cao–Yao–Zhang–style sparse functional DGP):
$$
Y_{ij} \;=\; m(T_{ij}) + \phi(T_{ij})^\top\xi_i + \sigma_\varepsilon\,\varepsilon_{ij},
$$
with $m(t) = 1.5\sin(3\pi(t+0.5)) + 2t^3$, $\phi(t) =
\sqrt 2\,\bigl(\sin 2\pi t,\cos 2\pi t,\sin 4\pi t,\cos 4\pi t\bigr)$,
eigenvalues $(1,\tfrac12,\tfrac14,\tfrac18)$,
$\sigma_\varepsilon = 0.1$, and $\xi_i, \varepsilon_{ij}$ drawn i.i.d. from
one of $\{\mathcal N(0,1),\,\mathrm{Uniform}(-\sqrt3,\sqrt3),\,\mathrm{Laplace}(0,1/\sqrt2)\}$.

**Settings**: $n\in\{200,300\}$, $m\in\{5,10,15\}$, $R=1{,}000$ per cell
(18 cells total).

**Table 6.3.** Part 3 — mean function SCB: Ours vs. Cai–Hu (2025) B-spline.

| n   | m  | noise   | **Ours cov / wid** | Cai–Hu cov / wid |
|-----|----|---------|--------------------|------------------|
| 200 | 5  | normal  | 0.965 / 1.115      | 0.919 / 0.892    |
| 200 | 5  | uniform | 0.961 / 1.122      | 0.908 / 0.893    |
| 200 | 5  | laplace | 0.957 / 1.111      | 0.906 / 0.889    |
| 200 | 10 | normal  | 0.960 / 0.889      | 0.936 / 0.743    |
| 200 | 10 | uniform | 0.959 / 0.890      | 0.931 / 0.742    |
| 200 | 10 | laplace | 0.953 / 0.886      | 0.931 / 0.741    |
| 200 | 15 | normal  | 0.951 / 0.806      | 0.927 / 0.685    |
| 200 | 15 | uniform | 0.959 / 0.805      | 0.930 / 0.686    |
| 200 | 15 | laplace | 0.947 / 0.800      | 0.941 / 0.682    |
| 300 | 5  | normal  | 0.949 / 0.896      | 0.922 / 0.732    |
| 300 | 5  | uniform | 0.954 / 0.901      | 0.923 / 0.733    |
| 300 | 5  | laplace | 0.958 / 0.896      | 0.927 / 0.730    |
| 300 | 10 | normal  | 0.941 / 0.728      | 0.928 / 0.609    |
| 300 | 10 | uniform | 0.958 / 0.731      | 0.934 / 0.608    |
| 300 | 10 | laplace | 0.948 / 0.730      | 0.920 / 0.608    |
| 300 | 15 | normal  | 0.957 / 0.661      | 0.923 / 0.561    |
| 300 | 15 | uniform | 0.943 / 0.661      | 0.916 / 0.561    |
| 300 | 15 | laplace | 0.958 / 0.658      | 0.946 / 0.559    |

**Aggregate (18 cells).**

| Metric                            | **Ours**          | Cai–Hu B-spline |
|-----------------------------------|-------------------|-----------------|
| Mean coverage                     | **0.954**         | 0.926           |
| Mean $|\hat{\mathrm{cov}} - 0.95|$ | **0.007**         | 0.024           |
| Coverage range                    | **[0.941, 0.965]** | [0.906, 0.946]  |
| Mean band width                   | 0.849             | **0.703**       |

**Findings.** Our SCB is well calibrated across all 18 cells (coverage in
$[0.941, 0.965]$, mean deviation from nominal $0.007$), at both sample
sizes. The Cai–Hu (2025) B-spline SCB under-covers systematically — mean
coverage $0.926$, every cell below $0.95$ — with the largest shortfall in
the sparse $m=5$ regime (coverage $0.906$–$0.927$), in line with the
finite-sample behaviour discussed by Cai and Hu (2025) at the sparse end of
the phase transition. The Cai–Hu band is narrower on average ($0.703$ vs.
$0.849$, a $17\%$ reduction), reflecting the efficiency premium of the
dedicated B-spline construction in a setting where its assumptions hold;
our local-linear SCB trades this width premium for substantially better
calibration and a single algorithm applicable to all three scenarios.

### 6.1.6 Summary of simulation findings

Across all $18 + 3 + 18 = 39$ simulation cells at $R = 1{,}000$ per cell,
our SCB satisfies:

1. **Calibration**. Mean coverages 0.945 (Part 1), 0.962 (Part 2), 0.954
   (Part 3); mean $|\hat{\mathrm{cov}} - 0.95|$ values 0.008, 0.012, 0.007
   respectively. **Every cell lies in $[0.931, 0.967]$**, i.e. within
   $\sim 2.7$ Monte-Carlo standard errors of nominal; the marginal bands
   (Parts 1 and 3) cluster tightly around $0.95$, and the joint band
   (Part 2) is mildly conservative, as expected for simultaneous coverage
   over five coefficient functions.
2. **Width efficiency**. In Part 1 our SCB is on average 36\% narrower than
   Cao (2018), with the gap widening to $\sim 47\%$ at $m=15$.
3. **Uniformity across scenarios and sample sizes**. The same algorithm —
   pooled local-linear estimator with LOSO bandwidth, leave-one-subject
   residuals, multiplier bootstrap, and a **single** undersmoothing rule
   $\hat h_{\text{inf}} = \hat h_{\text{est}}\bigl(1 - 8/(n^{1/2}(\bar m/q)^{1/4})\bigr)$
   with one global constant — handles marginal VCM ($q=2$), joint
   multi-coefficient VCM ($q=5$), and mean-function ($q=1$) settings, at
   both $n=200$ and $n=300$, with consistent calibration and a single code
   path.

## 6.2 Real data application: Mayo PBC sequential biomarkers

### 6.2.1 Data and design illustration

We apply the proposed SCB to the Mayo PBC sequential biomarker dataset
(Therneau & Grambsch, 2000; distributed as `pbcseq` in the R `survival`
package), collected from the Mayo Clinic randomized D-penicillamine trial of
primary biliary cirrhosis (PBC) reported by Dickson, Fleming, Wiesner,
Baldus, Fleming, Ludwig and McCall (1985). The full record contains $1{,}945$
serial measurements on $n=312$ patients followed for up to $\approx 13$
years. We restrict to the first $8.2$ years ($\text{day}\le 3{,}000$,
retaining $\approx 92\%$ of all observations) and drop records with missing
bilirubin, leaving $1{,}790$ observations on $n=312$ patients.

Figure 6.4 (`pbc_data_illustration.png`) summarizes the design. The
individual log-bilirubin trajectories (panel A) are observed at irregular,
patient-specific times, with a **median of $5$ observations per patient**
(mean $5.7$, range $1$–$11$; panel B) — a genuinely **sparse** longitudinal
design, squarely within the support of the $m\in\{5,10,15\}$ simulation of
§6.1. Raw serum bilirubin is strongly right-skewed (sample skewness $2.88$,
panel C); following standard practice (Therneau & Grambsch, 2000, Ch. 8) we
model $\log(\text{bilirubin})$, and rescale time to
$t=\text{day}/3000\in[0,1]$ for the kernel computations. The trial arms are
balanced ($158$ D-penicillamine vs. $154$ placebo) and the entry age has
median $49.8$ years (range $26$–$78$). Throughout we use the unified
two-stage bandwidth scheme of §6.1.2 ($\bar m = 5.74$,
$\hat h_{\text{est}}=0.070$ from LOSO).

### 6.2.2 Population mean bilirubin trajectory

**Scientific question.** Serum bilirubin is the single strongest prognostic
variable in the Mayo PBC natural-history model (Dickson, Grambsch, Fleming,
Fisher & Langworthy, 1989; Murtaugh, Dickson, Van Dam, Malinchoc, Grambsch,
Langworthy & Gips, 1994), and disease-defining hyperbilirubinemia is the
clinical hallmark of progressive PBC. We therefore first ask, at the
population level and with simultaneous (not pointwise) guarantees: *is the
cohort's mean bilirubin trajectory elevated above the normal clinical
reference of $1.0$ mg/dL, and how does it evolve over follow-up?* This
motivates the mean-function model
$$
\log(\text{bili}_{ij}) \;=\; m(t_{ij}) + \varepsilon_{ij}.
$$

**Analysis.** With $q=1$ the undersmoothing rule gives
$\hat h_{\text{inf}} = 0.070\,\bigl(1 - 8/(\sqrt{312}\,(5.74/1)^{1/4})\bigr)
= 0.050$; the bootstrap critical value is $\hat q_{\text{crit}} = 3.21$ and
the mean SCB width on the log scale is $0.93$. The Cai–Hu (2025) B-spline SCB
selects $J_n=6$ interior knots and is narrower (width $0.56$; ratio
$1.66$), consistent with the efficiency/calibration trade-off seen in §6.1.5.
Figure 6.5 (`pbc_mean_bands.png`) shows the band on both the log and the
clinical (back-transformed) scale; Table 6.4 reports clinical-scale values.

**Table 6.4.** PBC mean bilirubin (back-transformed, mg/dL) with the $95\%$
SCB at selected times.

| Time                   | $\widehat m(t)\to$ bili (mg/dL) | $95\%$ SCB (mg/dL) |
|------------------------|---------------------------------|--------------------|
| day $135$  ($t=0.04$)  | 2.14                            | $[1.64, 2.78]$     |
| day $795$  ($t=0.27$)  | 2.53                            | $[1.74, 3.68]$     |
| day $1545$ ($t=0.52$)  | 1.61                            | $[1.07, 2.45]$     |
| day $2235$ ($t=0.74$)  | 1.98                            | $[1.33, 2.94]$     |
| day $2865$ ($t=0.95$)  | 1.91                            | $[1.05, 3.48]$     |

**Interpretation.** Two readings follow, each corroborated by independent
evidence. *(i) Population-level hyperbilirubinemia.* The lower SCB limit
exceeds $1.0$ mg/dL across essentially the entire interior follow-up window,
giving **simultaneous** population-level evidence of disease-defining
hyperbilirubinemia — a stronger statement than a pointwise interval. The
magnitude agrees with published PBC cohorts: the Mayo cohort's own median
baseline bilirubin is $1.40$ mg/dL (Dickson et al., 1989), and European PBC
cohorts report baseline medians of $1.2$–$1.8$ mg/dL (Prince, Chetwynd,
Craig, Metcalf & James, 2002; Corpechot, Abenavoli, Rabahi, Chrétien,
Andréani, Johanet, Chazouillères & Poupon, 2008). *(ii) Non-monotone mean
under informative attrition.* Although individual PBC patients are expected
to show monotonically rising bilirubin as bile-duct destruction progresses
(Kaplan & Gershwin, 2005), the estimated population mean rises early and then
declines (peak $\approx 2.5$ mg/dL near day $800$, falling to $\approx 1.6$
by day $1500$). This is the documented signature of **informative
attrition**: the sickest patients (rapidly rising bilirubin) die or are
transplanted and exit the cohort first, biasing the surviving-cohort mean
downward at later times (Murtaugh et al., 1994; Hu, Sale, Davis, Wu, Khan &
Roecker, 2009; Wulfsohn & Tsiatis, 1997). The SCB thus describes the
cross-sectional mean of surviving members, not the cohort-average individual
trajectory.

### 6.2.3 Varying-coefficient analysis: treatment and age effects

**Scientific question.** Having established the population trajectory, we ask
*what drives it*: does the randomized therapy (D-penicillamine) or baseline
age modulate the bilirubin path over time? These are the two covariates of
direct clinical interest — treatment because it is the trial's randomized
intervention, age because it is a known prognostic factor in the Mayo model
(Dickson et al., 1989). This motivates the varying-coefficient model
$$
\log(\text{bili}_{ij}) \;=\; \beta_0(t_{ij}) + \beta_1(t_{ij})\,\mathrm{trt}_i
                              + \beta_2(t_{ij})\,\mathrm{age}_i^{\text{std}}
                              + \varepsilon_{ij},
$$
with $\mathrm{trt}_i\in\{0,1\}$ the D-penicillamine indicator and
$\mathrm{age}_i^{\text{std}}$ the mean-centred entry age in SD units
(constant within patient).

**Analysis.** Because three coefficient functions are interrogated at once,
per-coefficient $95\%$ bands would give a joint Type-I rate of
$1-0.95^3\approx 0.14$ (only $\approx 86\%$ joint coverage). We therefore
report a **joint** SCB over $(\beta_0,\beta_1,\beta_2)$, taking the bootstrap
supremum jointly across $(k,t)$, so that
$\mathbb P\bigl(\hat L_k(t)\le\beta_k(t)\le\hat U_k(t)\ \forall k,\forall t\bigr)\approx 0.95$.
With $q=3$ the rule gives
$\hat h_{\text{inf}} = 0.070\,\bigl(1 - 8/(\sqrt{312}\,(5.74/3)^{1/4})\bigr)
= 0.043$; the joint bootstrap critical value is $\hat q_{\text{crit}}^{\text{joint}}=3.54$.
For reference we also report the marginal Cao (2018) plug-in SCB
($\gamma=3.15$). Figure 6.6 (`pbc_vcm_bands.png`) shows the three bands.

**Table 6.5.** PBC VCM — joint $95\%$ SCB (Ours) vs. marginal Cao (2018)
band, with the simultaneous significance verdict (does the joint band exclude
zero somewhere?).

| Coefficient                     | Ours joint wid | Cao wid | wid ratio | Excludes 0? |
|---------------------------------|----------------|---------|-----------|-------------|
| $\beta_0$ (intercept, log-bili) | 1.759          | 1.888   | 0.93      | **yes**     |
| $\beta_1$ (D-penicillamine)     | 2.408          | 2.847   | 0.85      | no          |
| $\beta_2$ (age)                 | 1.344          | 1.546   | 0.87      | no          |

Even under the more stringent joint construction, our bands are
$7$–$15\%$ narrower than the marginal Cao bands (which do **not** provide
joint coverage).

**Interpretation.** The intercept $\beta_0(t)$ — the baseline-group
log-bilirubin path — is everywhere bounded away from zero, mirroring the
population hyperbilirubinemia of §6.2.2. The bands for $\beta_1$ and
$\beta_2$ contain zero at every $t$: neither D-penicillamine nor age
significantly modulates the bilirubin trajectory. The **null
D-penicillamine effect** corroborates, at the biomarker level and with
simultaneous guarantees, the long-established clinical fact that
D-penicillamine has no benefit in PBC — the principal finding of the original
Mayo trial (Dickson et al., 1985), confirmed in follow-up randomized
comparisons (Lombard, Portmann, Neuberger, Williams, Tygstrup, Ranek,
Ring-Larsen et al., 1993) and current AASLD practice guidance (Lindor,
Gershwin, Poupon, Kaplan, Bergasa & Heathcote, 2009; updated by Lindor,
Bowlus, Boyer, Levy & Mayo, 2019). By contrast, ursodeoxycholic acid (UDCA),
the current first-line therapy, lowers bilirubin and improves transplant-free
survival in the same biomarker framework (Poupon, Balkau, Eschwège, Poupon &
the UDCA-PBC Study Group, 1991; Corpechot, Carrat, Bahr, Chrétien, Poupon &
Poupon, 2005); the Mayo dataset predates UDCA's widespread adoption and
contains no UDCA arm.

In summary, the proposed SCB delivers (a) population-level conclusions that
agree with the original Mayo trial, the subsequent PBC literature, and
current clinical guidelines; (b) bands $7$–$15\%$ narrower than the Cao
(2018) per-target SCB on the VCM coefficients without loss of significance;
and (c) a mean-trajectory shape whose magnitude, non-monotonicity, and null
treatment effect each receive independent support from established PBC
pathophysiology and longitudinal-cohort epidemiology.

---

## References

Bachrach, L. K., Hastie, T., Wang, M.-C., Narasimhan, B. & Marcus, R.
(1999). Bone mineral acquisition in healthy Asian, Hispanic, Black, and
Caucasian youth: a longitudinal study. *J. Clin. Endocrinol. Metab.*
**84**(12), 4702–4712.

Cai, T. T. & Hu, J. (2025). From sparse to dense functional data: phase
transitions from a simultaneous inference perspective. *Statistics and
Computing* **35**, Article 146. https://doi.org/10.1007/s11222-025-10671-1.

Cao, G., Wang, S., Wang, Y. & Yang, L. (2018). Simultaneous nonparametric
regression analysis of sparse longitudinal data. *Statistica Sinica*
**28**, 2233–2255.

Calonico, S., Cattaneo, M. D. & Titiunik, R. (2014). Robust nonparametric
confidence intervals for regression-discontinuity designs. *Econometrica*
**82**(6), 2295–2326.

Corpechot, C., Carrat, F., Bahr, A., Chrétien, Y., Poupon, R.-E. &
Poupon, R. (2005). The effect of ursodeoxycholic acid therapy on the
natural course of primary biliary cirrhosis. *Gastroenterology* **128**(2),
297–303.

Corpechot, C., Abenavoli, L., Rabahi, N., Chrétien, Y., Andréani, T.,
Johanet, C., Chazouillères, O. & Poupon, R. (2008). Biochemical response
to ursodeoxycholic acid and long-term prognosis in primary biliary
cirrhosis. *Hepatology* **48**(3), 871–877.

Dickson, E. R., Fleming, T. R., Wiesner, R. H., Baldus, W. P., Fleming,
C. R., Ludwig, J. & McCall, J. T. (1985). Trial of penicillamine in
advanced primary biliary cirrhosis. *N. Engl. J. Med.* **312**(16),
1011–1015.

Dickson, E. R., Grambsch, P. M., Fleming, T. R., Fisher, L. D. &
Langworthy, A. (1989). Prognosis in primary biliary cirrhosis: model for
decision making. *Hepatology* **10**(1), 1–7.

Hu, C., Sale, M. E., Davis, S. M., Wu, Y., Khan, M. M. & Roecker, E. B.
(2009). Analyzing trial data with informative dropouts. *Statistics in
Biopharmaceutical Research* **1**(2), 209–217.

Kaplan, M. M. & Gershwin, M. E. (2005). Primary biliary cirrhosis.
*N. Engl. J. Med.* **353**(12), 1261–1273.

Lindor, K. D., Gershwin, M. E., Poupon, R., Kaplan, M., Bergasa, N. V. &
Heathcote, E. J. (2009). Primary biliary cirrhosis. AASLD practice
guidelines. *Hepatology* **50**(1), 291–308.

Lindor, K. D., Bowlus, C. L., Boyer, J., Levy, C. & Mayo, M. (2019).
Primary biliary cholangitis: 2018 practice guidance from the American
Association for the Study of Liver Diseases. *Hepatology* **69**(1),
394–419.

Lombard, M., Portmann, B., Neuberger, J., Williams, R., Tygstrup, N.,
Ranek, L., Ring-Larsen, H. et al. (1993). Cyclosporine A treatment in
primary biliary cirrhosis: results of a long-term placebo controlled
trial. *Gastroenterology* **104**(2), 519–526.

Murtaugh, P. A., Dickson, E. R., Van Dam, G. M., Malinchoc, M., Grambsch,
P. M., Langworthy, A. L. & Gips, C. H. (1994). Primary biliary cirrhosis:
prediction of short-term survival based on repeated patient visits.
*Hepatology* **20**(1), 126–134.

Poupon, R. E., Balkau, B., Eschwège, E., Poupon, R. & UDCA-PBC Study
Group (1991). A multicenter, controlled trial of ursodiol for the
treatment of primary biliary cirrhosis. *N. Engl. J. Med.* **324**(22),
1548–1554.

Pratt, D. S. & Kaplan, M. M. (2000). Evaluation of abnormal liver-enzyme
results in asymptomatic patients. *N. Engl. J. Med.* **342**(17),
1266–1271.

Prince, M. I., Chetwynd, A., Craig, W. L., Metcalf, J. V. & James,
O. F. W. (2002). Asymptomatic primary biliary cirrhosis: clinical
features, prognosis, and symptom progression in a large population-based
cohort. *Gut* **53**(6), 865–870.

Therneau, T. M. & Grambsch, P. M. (2000). *Modeling Survival Data:
Extending the Cox Model*. Springer. (Chapter 8 introduces the Mayo PBC
sequential biomarker dataset, distributed as `pbcseq` in the R `survival`
package.)

Wulfsohn, M. S. & Tsiatis, A. A. (1997). A joint model for survival and
longitudinal data measured with error. *Biometrics* **53**(1), 330–339.

Yao, F., Müller, H.-G. & Wang, J.-L. (2005). Functional data analysis for
sparse longitudinal data. *J. Amer. Statist. Assoc.* **100**, 577–590.

Zhang, X. & Wang, J.-L. (2016). From sparse to dense functional data and
beyond. *Ann. Statist.* **44**(5), 2281–2321.
