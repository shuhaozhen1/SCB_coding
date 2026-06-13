"""Shared utilities for simulation scripts.

DGPs:
    gen_data_part1  -- single-covariate VCM (q=2): Cao 2018 design with
                       subject-level multiplicative noise zeta_i * X_i.
    gen_data_part2  -- five-coefficient VCM (q=5): FDA Part-2 design with
                       Karhunen-Loeve-like subject loadings; iid Gaussian noise
                       calibrated to a target SNR.
    gen_data_part3  -- mean function DGP (q=1): Cai-Hu style sparse functional
                       data with 4-component KL noise; uses caihu_2025 helpers.

Parallel runner:
    parallel_map(fn, args_list, workers, chunksize) -- thin wrapper around
    ProcessPoolExecutor with safe defaults.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import numpy as np


def undersmooth_h_inf(h_est, n, m_bar, q, c=8.0, floor=0.02):
    """Inference bandwidth from the estimation bandwidth via the unified rule

        h_inf = h_est * ( 1 - c / ( n^{1/2} * (m_bar / q)^{1/4} ) ),

    where n = number of subjects, m_bar = mean observations per subject, and
    q = model dimension (number of VCM coefficients). The denominator is the
    per-coefficient effective resolution: sqrt(n) carries the subject-level
    information while (m_bar/q)^{1/4} mildly weakens the undersmoothing as the
    design becomes denser. The ratio -> 1 as n -> infinity, so the band is
    asymptotically equivalent to using h_est (undersmoothing-for-inference,
    Calonico-Cattaneo-Titiunik 2014; n*m_bar effective sample size, Yao-
    Mueller-Wang 2005, Zhang-Wang 2016). c=8 is calibrated once and shared
    across all settings.
    """
    denom = np.sqrt(n) * (m_bar / q) ** 0.25
    return max(h_est * (1.0 - c / denom), floor)


def _sample_gp(times: np.ndarray, var_kind: str, rng: np.random.Generator) -> np.ndarray:
    """Mean-zero Gaussian process at irregular times; var(t) and corr fixed per var_kind."""
    m = times.size
    dt = np.abs(times.reshape(-1, 1) - times.reshape(1, -1))
    if var_kind == "exp":
        var = np.exp(times); corr = np.exp(-dt / 4.0)
    elif var_kind == "pow2":
        var = np.maximum(2.0 * times, 1e-10); corr = 2.0 ** (-dt / 4.0)
    else:
        raise ValueError(var_kind)
    cov = np.sqrt(var.reshape(-1, 1) * var.reshape(1, -1)) * corr + 1e-10 * np.eye(m)
    return rng.multivariate_normal(np.zeros(m), cov)


def _beta_one_cov(t: np.ndarray, beta1_id: int = 1) -> np.ndarray:
    """Part 1 truth: beta_0(t)=5(t-0.6)^2; beta_1 has 4 alternative shapes (1=sin)."""
    t = np.asarray(t, dtype=float)
    beta0 = 5.0 * (t - 0.6) ** 2
    if beta1_id == 1:   beta1 = 0.5 + 0.4 * np.sin(2.0 * np.pi * (t - 0.5))
    elif beta1_id == 2: beta1 = 0.5 * np.cos(2.0 * np.pi * (t - 0.3))
    elif beta1_id == 3: beta1 = 4.0 * (t - 0.4) ** 2
    elif beta1_id == 4: beta1 = 4.0 * (t - 0.5) ** 3
    else: raise ValueError(beta1_id)
    return np.vstack([beta0, beta1])


def _draw_zeta(noise: str, rng: np.random.Generator) -> float:
    """Standardized subject-level multiplier (variance 1) for Parts 1 & 3."""
    if noise == "gauss":   return float(rng.normal())
    if noise == "uniform": return float(rng.uniform(-np.sqrt(3.0), np.sqrt(3.0)))
    if noise == "laplace": return float(rng.laplace(0.0, 1.0 / np.sqrt(2.0)))
    raise ValueError(noise)


# ---------------------------------------------------------------------------
# Part 1: single-coefficient VCM (q=2)
# ---------------------------------------------------------------------------
def gen_data_part1(n: int, m: int, noise: str, seed: int, beta1_id: int = 1):
    """Cao 2018 DGP: Y_ij = beta_0(T_ij) + beta_1(T_ij)*X_ij + zeta_i*X_ij."""
    rng = np.random.default_rng(seed)
    subj = np.repeat(np.arange(n), m)
    times = rng.uniform(0.0, 1.0, n * m)
    X = np.zeros(n * m); Y = np.zeros(n * m); idx = 0
    for _ in range(n):
        ti = times[idx:idx + m]
        xi = _sample_gp(ti, "exp", rng)
        zeta = _draw_zeta(noise, rng)
        b = _beta_one_cov(ti, beta1_id)
        Y[idx:idx + m] = b[0] + b[1] * xi + zeta * xi
        X[idx:idx + m] = xi
        idx += m
    Z = np.column_stack([np.ones(n * m), X])
    return subj, times, Z, Y


def beta_true_part1(t_grid: np.ndarray, beta1_id: int = 1) -> np.ndarray:
    """(G, 2) matrix of true [beta_0(t), beta_1(t)]."""
    return _beta_one_cov(t_grid, beta1_id).T


# ---------------------------------------------------------------------------
# Part 2: five-coefficient VCM (q=5)
# ---------------------------------------------------------------------------
def beta_true_part2(t: np.ndarray, p: int = 5) -> np.ndarray:
    """(G, p) truth: beta_0=1, beta_1=2 sin(2 pi t), beta_2=2(t-0.5),
    beta_3=exp(t), beta_4=cos(5 pi t); extras (k>=5) are cos((k+1) pi t)."""
    t = np.asarray(t, dtype=float)
    out = np.zeros((t.size, p), dtype=float)
    out[:, 0] = 1.0
    if p >= 2: out[:, 1] = 2.0 * np.sin(2.0 * np.pi * t)
    if p >= 3: out[:, 2] = 2.0 * (t - 0.5)
    if p >= 4: out[:, 3] = np.exp(t)
    for k in range(4, p):
        out[:, k] = np.cos((k + 1) * np.pi * t)
    return out


def gen_data_part2(n: int, m: int, snr: float, seed: int, p: int = 5):
    """KL-like covariates + iid Gaussian noise calibrated to SNR.

    Z_{ij,k} = a^(0)_{ik} + a^(1)_{ik} sin(2 pi T_ij) + a^(2)_{ik} cos(2 pi T_ij)
              + 0.2 * eps,   each a_i ~ iid N(0,1).
    Y_ij = sum_k Z_{ij,k} beta_k(T_ij) + sigma * eps_ij,   sigma calibrated to SNR.
    Returns lists keyed by subject for compatibility with the joint inference loop.
    """
    rng = np.random.default_rng(seed)
    subj_list, times_list, Z_list, signal_list = [], [], [], []
    signal_sq = 0.0; total_pts = 0
    for i in range(n):
        m_i = int(rng.integers(low=max(1, m - 2), high=m + 3))
        t_i = np.sort(rng.uniform(0.0, 1.0, size=m_i))
        a0 = rng.normal(size=p); a1 = rng.normal(size=p); a2 = rng.normal(size=p)
        zeta = (a0[None, :]
                + a1[None, :] * np.sin(2.0 * np.pi * t_i)[:, None]
                + a2[None, :] * np.cos(2.0 * np.pi * t_i)[:, None])
        zeta += 0.2 * rng.normal(size=zeta.shape)
        beta = beta_true_part2(t_i, p=p)
        signal = np.sum(zeta * beta, axis=1)
        signal_sq += float(np.sum(signal ** 2)); total_pts += int(m_i)
        subj_list.append(np.full(m_i, i, dtype=int))
        times_list.append(t_i); Z_list.append(zeta); signal_list.append(signal)
    sigma = float(np.sqrt(signal_sq / max(1, total_pts) / snr))
    subj = np.concatenate(subj_list); times = np.concatenate(times_list)
    Z = np.concatenate(Z_list, axis=0)
    eps = rng.normal(loc=0.0, scale=sigma, size=total_pts)
    Y = np.concatenate(signal_list) + eps
    return subj, times, Z, Y, sigma


# ---------------------------------------------------------------------------
# Part 3: mean function (q=1)
# ---------------------------------------------------------------------------
def gen_data_part3(n: int, m: int, dist: str, seed: int, sigma_eps: float = 0.1):
    """Cai-Hu mean-function DGP with KL noise + iid measurement noise."""
    from comparators.caihu_2025 import m_true, sample_std_dist, sigma_x, phi_matrix
    rng = np.random.default_rng(seed)
    Xs, Ys, sids = [], [], []
    for i in range(n):
        x = rng.uniform(0.0, 1.0, size=m)
        xi = sample_std_dist(dist, size=4, rng=rng)
        eps = sample_std_dist(dist, size=m, rng=rng)
        y = m_true(x) + (phi_matrix(x) @ xi) + sigma_x(x, sigma_eps=sigma_eps, hetero=False) * eps
        Xs.append(x); Ys.append(y); sids.append(np.full(m, i, dtype=int))
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(sids)


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------
def parallel_map(fn, args_list, workers: int, chunksize: int = 4,
                 initializer=None, initargs=()):
    """Wrapper around ProcessPoolExecutor.map for embarrassingly parallel reps."""
    if workers <= 1:
        return [fn(a) for a in args_list]
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=initializer, initargs=initargs) as ex:
        return list(ex.map(fn, args_list, chunksize=chunksize))
