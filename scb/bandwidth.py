from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from .config import BandwidthCVConfig, LocalLinearConfig
from .estimation import local_linear_vcm_fit
from .kernels import epanechnikov_kernel


@dataclass
class BandwidthCVResult:
    """Container for bandwidth cross-validation results."""

    bandwidth_grid: np.ndarray
    cv_scores: np.ndarray
    selected_bandwidth: float
    method: str
    n_folds: int
    fold_scores: list[np.ndarray]
    local_config_template: LocalLinearConfig
    cv_config: BandwidthCVConfig


def make_bandwidth_grid(
    h_min: float = 0.05,
    h_max: float = 0.25,
    num: int = 9,
    scale: str = "linear",
) -> np.ndarray:
    """Create a positive bandwidth grid for CV."""
    if h_min <= 0 or h_max <= 0 or h_max <= h_min:
        raise ValueError("Require 0 < h_min < h_max.")
    if num <= 1:
        raise ValueError("num must be greater than 1.")
    if scale == "linear":
        return np.linspace(h_min, h_max, num)
    if scale == "log":
        return np.exp(np.linspace(np.log(h_min), np.log(h_max), num))
    raise ValueError("scale must be either 'linear' or 'log'.")


def _subject_folds(subject_ids: np.ndarray, config: BandwidthCVConfig) -> list[np.ndarray]:
    """Return folds as arrays of subject ids."""
    subject_ids = np.asarray(subject_ids)
    n = subject_ids.size

    if config.method == "loo":
        return [np.array([sid]) for sid in subject_ids]

    n_folds = min(config.n_folds, n)
    rng = np.random.default_rng(config.seed)
    shuffled = subject_ids.copy()
    rng.shuffle(shuffled)
    return [fold for fold in np.array_split(shuffled, n_folds) if fold.size > 0]


def _prediction_loss(
    test_data: pd.DataFrame,
    beta_hat: np.ndarray,
    q: int,
    loss_weighting: str,
) -> float:
    """
    Compute prediction loss on test data.

    beta_hat must be aligned row-by-row with test_data, because the fitting grid
    is chosen as the test observation times.
    """
    X_test = test_data[[f"x{k}" for k in range(q)]].to_numpy(dtype=float)
    y_test = test_data["y"].to_numpy(dtype=float)
    pred = np.sum(X_test * beta_hat, axis=1)
    sq_err = (y_test - pred) ** 2
    valid = np.isfinite(sq_err)

    if not np.any(valid):
        return np.nan

    if loss_weighting == "observation":
        return float(np.mean(sq_err[valid]))

    tmp = pd.DataFrame({
        "id": test_data["id"].to_numpy()[valid],
        "sq_err": sq_err[valid],
    })
    return float(tmp.groupby("id")["sq_err"].mean().mean())


def select_bandwidth_cv(
    data: pd.DataFrame,
    q: int,
    bandwidth_grid: Iterable[float],
    local_config_template: Optional[LocalLinearConfig] = None,
    cv_config: Optional[BandwidthCVConfig] = None,
    kernel: Callable[[np.ndarray], np.ndarray] = epanechnikov_kernel,
) -> BandwidthCVResult:
    """
    Select bandwidth by subject-level cross-validation.

    The default split unit is the subject, not a single observation. For a given
    bandwidth h and a held-out subject set I_f, the procedure fits beta_h^{(-f)}
    using all subjects outside I_f, evaluates beta_h^{(-f)} at the observed times
    of the held-out subjects, and computes prediction error

        Y_ij - X_ij^T beta_h^{(-f)}(T_ij).

    method="loo" performs exact leave-one-subject-out CV. method="kfold" is the
    faster large-sample approximation.
    """
    if local_config_template is None:
        local_config_template = LocalLinearConfig()
    if cv_config is None:
        cv_config = BandwidthCVConfig()

    bandwidth_grid = np.asarray(list(bandwidth_grid), dtype=float)
    if bandwidth_grid.ndim != 1 or bandwidth_grid.size == 0:
        raise ValueError("bandwidth_grid must be a non-empty one-dimensional iterable.")
    if np.any(bandwidth_grid <= 0):
        raise ValueError("All bandwidths must be positive.")

    subject_ids = np.sort(data["id"].unique())
    folds = _subject_folds(subject_ids, cv_config)

    cv_scores = np.full(bandwidth_grid.size, np.nan)
    fold_scores: list[np.ndarray] = []

    for h_idx, h in enumerate(bandwidth_grid):
        local_config = replace(local_config_template, bandwidth=float(h))
        scores_h = np.full(len(folds), np.nan)

        for f_idx, heldout_ids in enumerate(folds):
            is_test = data["id"].isin(heldout_ids).to_numpy()
            train_data = data.loc[~is_test].copy()
            test_data = data.loc[is_test].copy()

            if train_data.empty or test_data.empty:
                continue

            # Evaluate beta at each held-out observation time. This makes the
            # prediction row-aligned with test_data.
            test_grid = test_data["t"].to_numpy(dtype=float)
            fit = local_linear_vcm_fit(
                data=train_data,
                q=q,
                grid=test_grid,
                config=local_config,
                kernel=kernel,
            )
            scores_h[f_idx] = _prediction_loss(
                test_data=test_data,
                beta_hat=fit.beta_hat,
                q=q,
                loss_weighting=cv_config.loss_weighting,
            )

        fold_scores.append(scores_h)
        cv_scores[h_idx] = np.nanmean(scores_h)

    if np.all(np.isnan(cv_scores)):
        raise RuntimeError("All CV scores are NaN. Check the bandwidth grid and sample size.")

    selected_idx = int(np.nanargmin(cv_scores))
    return BandwidthCVResult(
        bandwidth_grid=bandwidth_grid,
        cv_scores=cv_scores,
        selected_bandwidth=float(bandwidth_grid[selected_idx]),
        method=cv_config.method,
        n_folds=len(folds),
        fold_scores=fold_scores,
        local_config_template=local_config_template,
        cv_config=cv_config,
    )
