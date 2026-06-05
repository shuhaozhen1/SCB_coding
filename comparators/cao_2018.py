"""Cao 2018 plug-in SCB for sparse longitudinal varying-coefficient regression.

Reference
---------
Cao, G., Wang, S., Wang, Y. and Yang, L. (2018).
Simultaneous nonparametric regression analysis of sparse longitudinal data.
Statistica Sinica 28, 2233-2255.

The implementation includes:
    * a plain local-linear varying-coefficient estimator (fit_beta_at_points);
    * a leave-one-subject-out CV bandwidth selector (select_bandwidth_loso);
    * Silverman-pilot plug-in components f_hat / Omega / Lambda
      (estimate_plugin_components) and the corresponding sandwich
      standard-error curve (standard_error_curve);
    * the asymptotic Gumbel-type critical value via simulation of the
      limiting Gaussian-process supremum (simulate_gamma).

A 95% SCB is then constructed as
    beta_hat(t) +/- gamma * SE(t).
"""
from __future__ import annotations

import math
import numpy as np

LAMBDA_K = 0.6   # int K^2 for Epanechnikov


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
def epanechnikov(u: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel K(u) = 0.75*(1-u^2) * 1{|u|<=1}, vectorised."""
    out = 0.75 * (1.0 - u * u)
    out[np.abs(u) > 1.0] = 0.0
    out[out < 0.0] = 0.0
    return out


# ---------------------------------------------------------------------------
# Local-linear varying-coefficient estimator
# ---------------------------------------------------------------------------
def local_linear_beta(t0: float, times: np.ndarray, Z: np.ndarray,
                      Y: np.ndarray, h: float) -> np.ndarray:
    """beta_hat(t0) for the model Y = Z^T beta(T) + eps via pooled WLS.

    Returns the first q coefficients of theta = (beta, h*beta'). The local
    design is [Z, (T - t0) * Z]. Falls back to a widened window when the
    primary kernel sum is numerically zero.
    """
    u = (times - t0) / h
    w = epanechnikov(u)
    if w.sum() <= 1e-14:
        w = epanechnikov((times - t0) / (1.5 * h))
        if w.sum() <= 1e-14:
            w = np.ones_like(times)
    q = Z.shape[1]
    dt = (times - t0).reshape(-1, 1)
    D = np.concatenate([Z, Z * dt], axis=1)
    W = w.reshape(-1, 1)
    A = D.T @ (W * D) + 1e-10 * np.eye(D.shape[1])
    b = D.T @ (w * Y)
    try:
        theta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq(A, b, rcond=None)[0]
    return theta[:q]


def fit_beta_at_points(t_eval: np.ndarray, times: np.ndarray, Z: np.ndarray,
                       Y: np.ndarray, h: float) -> np.ndarray:
    """Vectorised wrapper over a grid of evaluation points."""
    t_eval = np.asarray(t_eval, dtype=float)
    q = Z.shape[1]
    out = np.zeros((t_eval.size, q), dtype=float)
    for i, t0 in enumerate(t_eval):
        out[i, :] = local_linear_beta(float(t0), times, Z, Y, h)
    return out


# ---------------------------------------------------------------------------
# Leave-one-subject-out bandwidth selection
# ---------------------------------------------------------------------------
def select_bandwidth_loso(subject_ids: np.ndarray, times: np.ndarray,
                          Z: np.ndarray, Y: np.ndarray,
                          h_candidates: np.ndarray) -> float:
    """Leave-one-SUBJECT-out CV. Picks h minimizing pooled squared error."""
    uniq = np.unique(subject_ids)
    N = times.size
    best_h, best_cv = float(h_candidates[0]), np.inf
    for h in h_candidates:
        h = float(h); se = 0.0
        for sid in uniq:
            mask_tr = subject_ids != sid; mask_te = subject_ids == sid
            t_tr, Z_tr, Y_tr = times[mask_tr], Z[mask_tr], Y[mask_tr]
            for tt, zz, yy in zip(times[mask_te], Z[mask_te], Y[mask_te]):
                beta = local_linear_beta(float(tt), t_tr, Z_tr, Y_tr, h)
                se += (float(yy) - float(zz @ beta)) ** 2
        cv = se / N
        if cv < best_cv:
            best_cv, best_h = cv, h
    return best_h


# ---------------------------------------------------------------------------
# Plug-in components for SCB sandwich variance
# ---------------------------------------------------------------------------
def _rule_of_thumb(times: np.ndarray, power: float = 0.2) -> float:
    """Silverman normal-reference pilot bandwidth: 1.06 * sd(t) * n^{-power}."""
    s = float(np.std(times))
    n = max(2, times.size)
    return float(max(1.06 * s * (n ** (-power)), 1e-3))


def estimate_plugin_components(t_grid: np.ndarray, times: np.ndarray,
                               Z: np.ndarray, Y: np.ndarray, h: float,
                               b_n: float | None = None,
                               c_n: float | None = None,
                               d_n: float | None = None) -> dict:
    """Estimate (f_hat, Omega, Lambda) used by the Cao (2018) sandwich SE.

    The three pilot bandwidths default to the Silverman rule of thumb and
    can be overridden independently. Residuals come from a pilot
    local-linear fit at the main bandwidth h.
    """
    times = np.asarray(times, float); Z = np.asarray(Z, float)
    Y = np.asarray(Y, float).reshape(-1)
    N, q = Z.shape
    if b_n is None: b_n = _rule_of_thumb(times)
    if c_n is None: c_n = _rule_of_thumb(times)
    if d_n is None: d_n = _rule_of_thumb(times)
    beta_obs = fit_beta_at_points(times, times, Z, Y, h)
    resid = Y - np.sum(Z * beta_obs, axis=1)
    G = t_grid.size
    fhat = np.zeros(G); omega = np.zeros((G, q, q)); lamb = np.zeros((G, q, q))
    for k, t0 in enumerate(t_grid):
        wb = epanechnikov((times - t0) / b_n)
        wc = epanechnikov((times - t0) / c_n)
        wd = epanechnikov((times - t0) / d_n)
        swb, swc, swd = wb.sum(), wc.sum(), wd.sum()
        fhat[k] = max(swb / (N * b_n), 1e-12) if swb > 1e-14 else 1e-12
        omega[k] = (Z.T @ (wc.reshape(-1, 1) * Z) / swc) if swc > 1e-14 else np.eye(q)
        if swd > 1e-14:
            wz = (wd * (resid ** 2)).reshape(-1, 1) * Z
            lamb[k] = Z.T @ wz / swd
        else:
            lamb[k] = np.eye(q)
    return {"fhat": fhat, "omega": omega, "lambda": lamb,
            "resid": resid, "b_n": float(b_n), "c_n": float(c_n), "d_n": float(d_n)}


def standard_error_curve(a: np.ndarray, fhat: np.ndarray, omega: np.ndarray,
                         lamb: np.ndarray, N: int, h: float) -> np.ndarray:
    """Sandwich SE for the projection a^T beta_hat(t)."""
    a = np.asarray(a, float).reshape(-1)
    G = fhat.size; out = np.zeros(G)
    for k in range(G):
        om = omega[k] + 1e-10 * np.eye(omega.shape[1])
        la = lamb[k]
        try: om_inv = np.linalg.inv(om)
        except np.linalg.LinAlgError: om_inv = np.linalg.pinv(om)
        v = float(a @ om_inv @ la @ om_inv @ a)
        out[k] = math.sqrt(max(v, 0.0) * LAMBDA_K / (N * h * max(fhat[k], 1e-12)))
    return out


# ---------------------------------------------------------------------------
# Asymptotic Gumbel-type critical value
# ---------------------------------------------------------------------------
def simulate_gamma(alpha: float, h: float, N: int, t_grid: np.ndarray,
                   B: int, seed: int) -> float:
    """1-alpha quantile of sup_t |sum_i eta_i K((T_i-t)/h)| / sqrt(LAMBDA_K*h*N)
    where T_i ~ U(0,1), eta_i ~ N(0,1). Standard Cao (2018) construction.
    """
    rng = np.random.default_rng(seed)
    t_grid = np.asarray(t_grid, dtype=float)
    denom = np.sqrt(LAMBDA_K * h * N)
    vals = np.zeros(B)
    for b in range(B):
        T = rng.uniform(0.0, 1.0, size=N)
        eta = rng.normal(0.0, 1.0, size=N)
        U = (T.reshape(-1, 1) - t_grid.reshape(1, -1)) / h
        W = epanechnikov(U)
        S = eta.reshape(-1, 1) * W
        vals[b] = np.max(np.abs(S.sum(axis=0)) / denom)
    return float(np.quantile(vals, 1.0 - alpha))


def scb_for_projection(t_grid, times, Z, Y, a, n, h, gamma_value):
    """Convenience: full Cao 2018 SCB for the projection a^T beta(t)."""
    bh = fit_beta_at_points(t_grid, times, Z, Y, h)
    comp = estimate_plugin_components(t_grid, times, Z, Y, h)
    se = standard_error_curve(a, comp["fhat"], comp["omega"], comp["lambda"], n, h)
    est = bh @ a
    return est, est - gamma_value * se, est + gamma_value * se, se
