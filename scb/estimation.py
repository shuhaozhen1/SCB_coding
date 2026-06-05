from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .config import LocalLinearConfig
from .kernels import epanechnikov_kernel, kernel_h


@dataclass
class LocalLinearFit:
    """Container for pooled local-linear VCM estimates."""

    grid: np.ndarray
    beta_hat: np.ndarray
    beta_slope_hat: np.ndarray
    effective_obs: np.ndarray
    effective_subjects: np.ndarray
    A_matrices: np.ndarray
    theta_hat: np.ndarray
    config: LocalLinearConfig


def _extract_arrays(data: pd.DataFrame, q: int):
    required = ["id", "t", "y"] + [f"x{k}" for k in range(q)]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw_ids = data["id"].to_numpy()
    unique_ids, subject_index = np.unique(raw_ids, return_inverse=True)
    t = data["t"].to_numpy(dtype=float)
    y = data["y"].to_numpy(dtype=float)
    X = data[[f"x{k}" for k in range(q)]].to_numpy(dtype=float)
    m_by_subject = np.bincount(subject_index)

    return unique_ids, subject_index, m_by_subject, t, y, X


def _observation_weights(
    K_h: np.ndarray,
    subject_index: np.ndarray,
    m_by_subject: np.ndarray,
    n_subjects: int,
    weighting: str,
) -> np.ndarray:
    """
    Return objective weights. Common multiplicative constants do not affect
    theta_hat, but keeping them makes A_n and influence notation explicit.
    """
    if weighting == "subject":
        return K_h / m_by_subject[subject_index] / n_subjects
    if weighting == "observation":
        return K_h / K_h.size
    raise ValueError("weighting must be either 'subject' or 'observation'.")


def build_local_design(X: np.ndarray, delta: np.ndarray, bandwidth: float, scaled: bool) -> np.ndarray:
    """
    Construct local-linear design [X, v X].

    If scaled=True, v = (T - t0) / h. The second coefficient estimates h beta'(t0).
    This scaling improves numerical conditioning. If scaled=False, v = T - t0.
    """
    v = delta / bandwidth if scaled else delta
    return np.hstack([X, X * v[:, None]])


def local_linear_vcm_fit(
    data: pd.DataFrame,
    q: int,
    grid: np.ndarray,
    config: LocalLinearConfig,
    kernel: Callable[[np.ndarray], np.ndarray] = epanechnikov_kernel,
) -> LocalLinearFit:
    """
    Fit beta(t) by pooled local-linear least squares.

    For subject-level sparse weighting, the objective is

        (1/n) sum_i (1/m_i) sum_j K_h(T_ij - t)
            [Y_ij - U_ij(t)^T theta(t)]^2.

    beta_hat(t) is the first q entries of theta_hat(t).
    """
    grid = np.asarray(grid, dtype=float)
    G = grid.size
    p = 2 * q

    ids, subject_index, m_by_subject, t, y, X = _extract_arrays(data, q)
    n_subjects = len(ids)

    beta_hat = np.full((G, q), np.nan)
    beta_slope_hat = np.full((G, q), np.nan)
    theta_hat_all = np.full((G, p), np.nan)
    A_all = np.full((G, p, p), np.nan)
    effective_obs = np.zeros(G, dtype=int)
    effective_subjects = np.zeros(G, dtype=int)

    for g, t0 in enumerate(grid):
        delta = t - t0
        K_h = kernel_h(delta, config.bandwidth, kernel=kernel)
        mask = K_h > 0
        effective_obs[g] = int(mask.sum())
        effective_subjects[g] = int(np.unique(subject_index[mask]).size)

        if effective_obs[g] < p:
            continue

        Z = build_local_design(X, delta, config.bandwidth, config.scaled_design)
        weights = _observation_weights(
            K_h=K_h,
            subject_index=subject_index,
            m_by_subject=m_by_subject,
            n_subjects=n_subjects,
            weighting=config.weighting,
        )

        Zw = Z * weights[:, None]
        A = Z.T @ Zw
        b = Z.T @ (weights * y)
        A_reg = A + config.ridge * np.eye(p)

        try:
            theta = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(A_reg) @ b

        theta_hat_all[g, :] = theta
        A_all[g, :, :] = A_reg
        beta_hat[g, :] = theta[:q]
        if config.scaled_design:
            beta_slope_hat[g, :] = theta[q:] / config.bandwidth
        else:
            beta_slope_hat[g, :] = theta[q:]

    return LocalLinearFit(
        grid=grid,
        beta_hat=beta_hat,
        beta_slope_hat=beta_slope_hat,
        effective_obs=effective_obs,
        effective_subjects=effective_subjects,
        A_matrices=A_all,
        theta_hat=theta_hat_all,
        config=config,
    )
