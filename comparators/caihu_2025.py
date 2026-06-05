"""Cai & Hu (2025) B-spline SCB for the mean function of sparse functional data.

Reference
---------
Cai, T. T. and Hu, J. (2025).
From sparse to dense functional data: phase transitions from a simultaneous
inference perspective. Statistics and Computing 35, Article 146.
DOI: 10.1007/s11222-025-10671-1   (arXiv: 2401.17646)

Method
------
* Cubic B-spline regression for the mean m(t), with BIC-selected interior
  knots J_n over [ceil(0.5 n^{1/4} (log n)^{1/2}), floor(2 n^{0.3})].
* Subject-level sandwich covariance of the B-spline coefficient vector
  Sigma_hat = V^{-1} [(n * Nbar^2)^{-1} sum_i s_i s_i^T] V^{-1},
  where s_i = sum_{j in subject i} B(T_ij) r_ij and V = (B^T B) / (n Nbar).
* Critical value Q = (1 - alpha)-quantile of sup_t |B(t)^T Z| / se(t),
  with Z ~ N(0, Sigma_hat) drawn B times.
* 95% SCB:  mhat(t) +/- (Q / sqrt(n)) * se(t).

Also exports the simulation DGP utilities m_true / phi_matrix / sigma_x
/ sample_std_dist / generate_dataset / sample_Ni used by the mean-function
Monte Carlo experiments.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


def m_true(x):
    return 1.5 * np.sin(3.0 * np.pi * (x + 0.5)) + 2.0 * x**3


def sample_std_dist(dist, size, rng):
    dist = dist.lower()
    if dist == "normal":
        return rng.normal(0.0, 1.0, size=size)
    if dist == "uniform":
        a = np.sqrt(3.0)
        return rng.uniform(-a, a, size=size)
    if dist == "laplace":
        b = 1.0 / np.sqrt(2.0)
        return rng.laplace(0.0, b, size=size)
    if dist == "t3":
        return rng.standard_t(df=3, size=size) / np.sqrt(3.0)
    if dist == "t5":
        return rng.standard_t(df=5, size=size) / np.sqrt(5.0 / 3.0)
    raise ValueError("dist must be one of: normal/uniform/laplace/t3/t5")


def sigma_x(x, sigma_eps=0.1, hetero=False):
    if not hetero:
        return sigma_eps * np.ones_like(x)
    return 1.2 * sigma_eps * (5.0 - np.exp(-x)) / (5.0 + np.exp(-x))


def phi_matrix(x):
    x = np.asarray(x)
    psi1 = np.sqrt(2.0) * np.sin(2.0 * np.pi * x)
    psi2 = np.sqrt(2.0) * np.cos(2.0 * np.pi * x)
    psi3 = np.sqrt(2.0) * np.sin(4.0 * np.pi * x)
    psi4 = np.sqrt(2.0) * np.cos(4.0 * np.pi * x)
    Psi = np.vstack([psi1, psi2, psi3, psi4]).T
    lam = np.array([2.0 ** (1 - k) for k in range(1, 5)], dtype=float)
    return Psi * np.sqrt(lam)[None, :]


def sample_Ni(n, scheme, rng):
    if scheme == 1:
        return int(rng.choice([3, 4, 5, 6]))
    if scheme == 2:
        lo = int(np.floor(2 * n**(1 / 5)))
        hi = int(np.floor(4 * n**(1 / 5)))
        return int(rng.integers(lo, hi + 1))
    if scheme == 3:
        lo = int(np.floor(n**0.5))
        hi = int(np.floor(2 * n**0.5))
        return int(rng.integers(lo, hi + 1))
    if scheme == 4:
        lo = int(np.floor(n / 4))
        hi = int(np.floor(n / 2))
        return int(rng.integers(lo, hi + 1))
    raise ValueError("scheme must be 1,2,3,4")


def generate_dataset(
    n,
    scheme,
    dist="normal",
    hetero=False,
    sigma_eps=0.1,
    seed=0,
    mean_func=m_true,
):
    rng = np.random.default_rng(seed)

    X_list, Y_list, sid_list = [], [], []
    for i in range(n):
        Ni = sample_Ni(n, scheme, rng)
        x = rng.uniform(0.0, 1.0, size=Ni)
        xi = sample_std_dist(dist, size=4, rng=rng)
        eps = sample_std_dist(dist, size=Ni, rng=rng)
        y = mean_func(x) + (phi_matrix(x) @ xi) + sigma_x(x, sigma_eps=sigma_eps, hetero=hetero) * eps

        X_list.append(x)
        Y_list.append(y)
        sid_list.append(np.full(Ni, i, dtype=int))

    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    sid = np.concatenate(sid_list)
    return X, Y, sid


def bspline_design(x, Jn, degree=3):
    x = np.asarray(x, dtype=float)
    p = degree + 1

    if Jn > 0:
        interior = np.linspace(0.0, 1.0, Jn + 2)[1:-1]
    else:
        interior = np.array([], dtype=float)

    t = np.concatenate([np.zeros(p), interior, np.ones(p)])
    K = Jn + p

    B = np.empty((x.size, K), dtype=float)
    for k in range(K):
        c = np.zeros(K, dtype=float)
        c[k] = 1.0
        spl = BSpline(t, c, degree, extrapolate=False)
        B[:, k] = spl(x)

    return np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)


def choose_Jn_by_bic(X, Y, n, degree=3):
    Ntot = X.size
    Nbar = Ntot / n

    lo = int(np.ceil(0.5 * (n**0.25) * (np.log(n) ** 0.5)))
    hi = int(np.floor(2.0 * (n**0.3)))
    lo = max(lo, 1)
    hi = max(hi, lo)

    best = None
    for Jn in range(lo, hi + 1):
        Bmat = bspline_design(X, Jn=Jn, degree=degree)
        theta, *_ = np.linalg.lstsq(Bmat, Y, rcond=None)
        resid = Y - Bmat @ theta
        mse = np.mean(resid**2)
        bic = np.log(mse) + (Jn * np.log(n * Nbar)) / (n * Nbar)

        if best is None or bic < best["bic"]:
            best = dict(Jn=Jn, bic=bic, theta=theta, resid=resid)

    return best["Jn"], best["theta"], best["resid"]


def estimate_Sigma_hat(X, resid, sid, n, Jn, degree=3):
    Ntot = X.size
    Nbar = Ntot / n

    Bmat = bspline_design(X, Jn=Jn, degree=degree)
    Vhat = (Bmat.T @ Bmat) / (n * Nbar)
    invV = np.linalg.inv(Vhat)

    S_total = np.zeros((Bmat.shape[1], Bmat.shape[1]), dtype=float)
    S2 = np.zeros((Bmat.shape[1], Bmat.shape[1]), dtype=float)

    for i in range(n):
        idx = sid == i
        Bi = Bmat[idx, :]
        Ui = resid[idx]
        si = (Bi * Ui[:, None]).sum(axis=0)
        S_total += np.outer(si, si)
        S2 += (Bi.T * (Ui**2)) @ Bi

    scale = 1.0 / (n * (Nbar**2))
    Sigma_inside_total = scale * S_total
    Sigma_inside_2 = scale * S2
    Sigma_inside_1 = Sigma_inside_total - Sigma_inside_2

    Sigma_hat = invV @ Sigma_inside_total @ invV
    Sigma_hat = 0.5 * (Sigma_hat + Sigma_hat.T)

    evals, evecs = np.linalg.eigh(Sigma_hat)
    evals_clipped = np.clip(evals, 0.0, None)
    Sigma_hat = (evecs * evals_clipped[None, :]) @ evecs.T

    return Sigma_hat, (invV, Bmat, Sigma_inside_1, Sigma_inside_2)


def scb_from_dataset(X, Y, sid, n, alpha=0.05, B=500, grid_size=2001, degree=3, seed=123):
    rng = np.random.default_rng(seed)
    Jn, theta, resid = choose_Jn_by_bic(X, Y, n, degree=degree)
    Sigma_hat, _ = estimate_Sigma_hat(X, resid, sid, n, Jn, degree=degree)

    grid = np.linspace(0.0, 1.0, grid_size)
    Bg = bspline_design(grid, Jn=Jn, degree=degree)
    se2 = np.einsum("ik,kl,il->i", Bg, Sigma_hat, Bg)
    se = np.sqrt(np.maximum(se2, 1e-30))
    mhat_g = Bg @ theta

    sups = np.empty(B, dtype=float)
    for b in range(B):
        Z = rng.multivariate_normal(mean=np.zeros(Sigma_hat.shape[0]), cov=Sigma_hat)
        zeta = (Bg @ Z) / se
        sups[b] = np.max(np.abs(zeta))

    Q = np.quantile(sups, 1.0 - alpha)
    half = (Q / np.sqrt(n)) * se
    lower = mhat_g - half
    upper = mhat_g + half
    meta = dict(Jn=Jn, Q=Q)
    return grid, mhat_g, lower, upper, meta


def mc_coverage(n=200, scheme=1, dist="normal", hetero=False, alpha=0.05, B=500, mc=200, grid_size=2001, degree=3, seed0=0):
    cover = 0
    for r in range(mc):
        X, Y, sid = generate_dataset(n, scheme, dist=dist, hetero=hetero, sigma_eps=0.1, seed=seed0 + r)
        grid, _, lo, up, _ = scb_from_dataset(
            X,
            Y,
            sid,
            n,
            alpha=alpha,
            B=B,
            grid_size=grid_size,
            degree=degree,
            seed=10_000 + r,
        )
        truth = m_true(grid)
        cover += int(np.all((truth >= lo) & (truth <= up)))
    return cover / mc
