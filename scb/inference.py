from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from .config import BootstrapConfig, LocalLinearConfig
from .estimation import (
    _extract_arrays,
    _observation_weights,
    build_local_design,
    local_linear_vcm_fit,
)
from .kernels import epanechnikov_kernel, kernel_h


@dataclass
class SCBResult:
    """Container for VCM simultaneous confidence-band output."""

    grid: np.ndarray
    beta_hat: np.ndarray
    beta_slope_hat: np.ndarray
    sigma_hat: np.ndarray
    se_hat: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    qcrit: float
    boot_sup: np.ndarray
    psi_hat: np.ndarray
    effective_obs: np.ndarray
    effective_subjects: np.ndarray
    components: np.ndarray
    local_config: LocalLinearConfig
    bootstrap_config: BootstrapConfig


def local_linear_vcm_scb(
    data: pd.DataFrame,
    q: int,
    grid: np.ndarray,
    local_config: LocalLinearConfig,
    bootstrap_config: BootstrapConfig,
    components: Optional[Iterable[int]] = None,
    kernel: Callable[[np.ndarray], np.ndarray] = epanechnikov_kernel,
    bandwidth_inf: Optional[float] = None,
) -> SCBResult:
    """
    Estimate beta(t) and construct multiplier-bootstrap simultaneous confidence bands.

    The bootstrap is based on subject-level influence contributions

        psi_i,k(t) = L_k A_n(t)^{-1} G_i(t),

    where, under the default observation-time residual method,

        G_i(t) = (1/m_i) sum_j K_h(T_ij - t) U_ij(t) r_hat_ij,
        r_hat_ij = Y_ij - X_ij^T beta_hat(T_ij).

    That is, residuals are first constructed at the original observation times
    T_ij and then reused when evaluating the bootstrap process on the target
    grid. This matches the intended centered-data bootstrap. For comparison,
    BootstrapConfig(residual_method="local") uses target-specific local residuals
    Y_ij - U_ij(t)^T theta_hat(t).

    bandwidth_inf decouples the estimation and inference bandwidths. When None
    (default), a single bandwidth local_config.bandwidth is used everywhere. When
    set, local_config.bandwidth (h_est) is the estimation bandwidth used for the
    reported point estimate beta_hat and the plug-in residuals, while
    bandwidth_inf (h_inf) is used for the influence-function weights, the local
    matrix A_n(t), and hence the variance / bootstrap process. This mirrors the
    two-bandwidth construction in the FDA reference implementation.
    """
    grid = np.asarray(grid, dtype=float)
    G = grid.size
    p = 2 * q

    if components is None:
        components_array = np.arange(q, dtype=int)
    else:
        components_array = np.asarray(list(components), dtype=int)
    if components_array.size == 0 or np.any((components_array < 0) | (components_array >= q)):
        raise ValueError("components must be a non-empty subset of {0, ..., q-1}.")

    fit = local_linear_vcm_fit(
        data=data,
        q=q,
        grid=grid,
        config=local_config,
        kernel=kernel,
    )

    # Inference bandwidth. When bandwidth_inf is given and differs from the
    # estimation bandwidth, the influence-function machinery (local matrix A_n,
    # kernel weights, design) is rebuilt at h_inf, while the band is still
    # centered at the h_est estimate fit.beta_hat.
    h_inf = local_config.bandwidth if bandwidth_inf is None else float(bandwidth_inf)
    if bandwidth_inf is None or h_inf == local_config.bandwidth:
        fit_inf = fit
        inf_config = local_config
    else:
        inf_config = replace(local_config, bandwidth=h_inf)
        fit_inf = local_linear_vcm_fit(
            data=data,
            q=q,
            grid=grid,
            config=inf_config,
            kernel=kernel,
        )

    ids, subject_index, m_by_subject, t, y, X = _extract_arrays(data, q)
    n_subjects = len(ids)

    # Preferred residual construction for SCB:
    # first estimate beta at the original observation times T_ij, then form
    # r_hat_ij = Y_ij - X_ij^T beta_hat(T_ij). These residuals do not depend
    # on the target grid point t. They are then plugged into the subject-level
    # influence process G_i(t).
    # Plug-in residuals are computed at the INFERENCE bandwidth h_inf (not h_est).
    # Rationale: the influence / variance machinery uses h_inf, so the residuals
    # should reflect the same scale of local smoothing. With h_inf < h_est, the
    # residuals are less shrunk by in-sample smoothing -> E[r^2] is closer to
    # (and slightly above) sigma^2, giving an SE estimate that better matches
    # the actual sup distribution at finite n (especially under heavy tails).
    if bootstrap_config.residual_method == "observation":
        obs_fit = local_linear_vcm_fit(
            data=data,
            q=q,
            grid=t,
            config=inf_config,
            kernel=kernel,
        )
        fitted_y_at_obs = np.sum(X * obs_fit.beta_hat, axis=1)
        obs_residual = y - fitted_y_at_obs
    elif bootstrap_config.residual_method == "loo":
        # Leave-one-subject-out residuals at h_inf: refit beta with subject i
        # removed (using inf_config) and evaluate at subject i's observation
        # times.
        obs_residual = np.full(len(y), np.nan)
        for i in range(n_subjects):
            mask_i = subject_index == i
            train = data.iloc[~mask_i]
            t_i = t[mask_i]
            loo_fit = local_linear_vcm_fit(
                data=train,
                q=q,
                grid=t_i,
                config=inf_config,
                kernel=kernel,
            )
            obs_residual[mask_i] = y[mask_i] - np.sum(X[mask_i] * loo_fit.beta_hat, axis=1)
    else:
        obs_residual = None

    psi_hat = np.full((n_subjects, G, q), np.nan)
    sigma_hat = np.full((G, q), np.nan)
    se_hat = np.full((G, q), np.nan)

    for g, t0 in enumerate(grid):
        theta = fit_inf.theta_hat[g, :]
        A_reg = fit_inf.A_matrices[g, :, :]
        if np.any(np.isnan(theta)) or np.any(np.isnan(A_reg)):
            continue

        delta = t - t0
        K_h = kernel_h(delta, h_inf, kernel=kernel)
        Z = build_local_design(X, delta, h_inf, inf_config.scaled_design)

        if obs_residual is not None:
            residual = obs_residual
        else:
            # Backward-compatible target-specific local-linear residual.
            residual = y - Z @ theta

        weights = _observation_weights(
            K_h=K_h,
            subject_index=subject_index,
            m_by_subject=m_by_subject,
            n_subjects=n_subjects,
            weighting=local_config.weighting,
        )

        # The estimating equation is sum_obs weights * Z * residual = 0.
        # For influence representation n^{-1} sum_i G_i, multiply back by n
        # under subject weighting so G_i = m_i^{-1} sum K_h Z r.
        # For observation weighting, this remains a subject-clustered empirical
        # contribution matching the same estimating equation up to constants.
        if local_config.weighting == "subject":
            contribution_weight = K_h / m_by_subject[subject_index]
        else:
            contribution_weight = K_h * (n_subjects / K_h.size)

        G_i = np.zeros((n_subjects, p), dtype=float)
        np.add.at(G_i, subject_index, Z * (contribution_weight * residual)[:, None])

        if bootstrap_config.center_influence:
            G_i = G_i - G_i.mean(axis=0, keepdims=True)

        try:
            A_inv = np.linalg.solve(A_reg, np.eye(p))
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A_reg)

        theta_influence = G_i @ A_inv.T
        beta_influence = theta_influence[:, :q]
        psi_hat[:, g, :] = beta_influence

        sigma = np.sqrt(np.mean(beta_influence**2, axis=0))
        sigma_hat[g, :] = sigma
        se_hat[g, :] = sigma / np.sqrt(n_subjects)

    rng = np.random.default_rng(bootstrap_config.seed)
    md = bootstrap_config.multiplier_dist
    shape = (bootstrap_config.n_boot, n_subjects)
    if md == "gaussian":
        multipliers = rng.normal(size=shape)
    elif md == "laplace":
        # Variance-1 Laplace, kurtosis 6 (heavier than N(0,1) kurt 3).
        multipliers = rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=shape)
    elif md == "rademacher":
        multipliers = (rng.integers(0, 2, size=shape) * 2 - 1).astype(float)
    elif md == "mammen":
        # Mammen's two-point: E=0, E^2=1, E^3=1 (asymmetric, third-moment match).
        p = (np.sqrt(5.0) + 1.0) / (2.0 * np.sqrt(5.0))
        u = rng.uniform(size=shape) < p
        multipliers = np.where(u, (1.0 - np.sqrt(5.0)) / 2.0, (1.0 + np.sqrt(5.0)) / 2.0)
    elif md == "t3":
        # Heavy-tailed: standardized t_3 (variance 1; infinite kurtosis).
        multipliers = rng.standard_t(3.0, size=shape) / np.sqrt(3.0)
    else:
        raise ValueError(md)

    boot_process = np.einsum("bn,ngq->bgq", multipliers, psi_hat) / np.sqrt(n_subjects)
    sigma_safe = sigma_hat.copy()
    sigma_safe[sigma_safe <= 1e-14] = np.nan
    boot_process_std = boot_process / sigma_safe[None, :, :]

    selected = boot_process_std[:, :, components_array]
    boot_sup = np.nanmax(np.abs(selected), axis=(1, 2))
    qcrit = float(np.nanquantile(boot_sup, 1.0 - bootstrap_config.alpha))

    lower = fit.beta_hat - qcrit * se_hat
    upper = fit.beta_hat + qcrit * se_hat

    return SCBResult(
        grid=grid,
        beta_hat=fit.beta_hat,
        beta_slope_hat=fit.beta_slope_hat,
        sigma_hat=sigma_hat,
        se_hat=se_hat,
        lower=lower,
        upper=upper,
        qcrit=qcrit,
        boot_sup=boot_sup,
        psi_hat=psi_hat,
        effective_obs=fit.effective_obs,
        effective_subjects=fit.effective_subjects,
        components=components_array,
        local_config=local_config,
        bootstrap_config=bootstrap_config,
    )


def band_covers_truth(result: SCBResult, beta_true: np.ndarray, components: Optional[Iterable[int]] = None) -> bool:
    """Return whether the SCB covers beta_true on the grid for selected components."""
    if components is None:
        comps = result.components
    else:
        comps = np.asarray(list(components), dtype=int)
    inside = (result.lower[:, comps] <= beta_true[:, comps]) & (beta_true[:, comps] <= result.upper[:, comps])
    return bool(np.nanmin(inside))
