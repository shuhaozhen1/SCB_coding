"""Smoke tests for the three components of the repository."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scb import BootstrapConfig, LocalLinearConfig, band_covers_truth, local_linear_vcm_scb
from comparators.cao_2018 import (fit_beta_at_points, select_bandwidth_loso,
                                  simulate_gamma, estimate_plugin_components,
                                  standard_error_curve)
from comparators.caihu_2025 import m_true, scb_from_dataset as caihu_scb
from simulations._common import gen_data_part1, gen_data_part2, gen_data_part3, beta_true_part1


def test_scb_one_rep_part1():
    n, m = 40, 5
    subj, t, Z, Y = gen_data_part1(n, m, "gauss", seed=1)
    grid = np.linspace(0.1, 0.9, 21)
    df = pd.DataFrame({"id": subj.astype(int), "t": t, "y": Y, "x0": Z[:, 0], "x1": Z[:, 1]})
    lc = LocalLinearConfig(bandwidth=0.15, ridge=1e-6, weighting="subject", scaled_design=True)
    bc = BootstrapConfig(n_boot=50, alpha=0.1, seed=2, residual_method="loo")
    scb = local_linear_vcm_scb(data=df, q=2, grid=grid, local_config=lc,
                               bootstrap_config=bc, components=[1], bandwidth_inf=0.10)
    assert np.isfinite(scb.qcrit)
    truth = beta_true_part1(grid)
    cov = band_covers_truth(scb, truth, components=[1])
    assert cov in (True, False)


def test_cao_2018_smoke():
    n, m = 40, 5
    subj, t, Z, Y = gen_data_part1(n, m, "gauss", seed=3)
    h_grid = np.linspace(0.08, 0.18, 4)
    h = float(select_bandwidth_loso(subj, t, Z, Y, h_grid))
    assert h in set(map(float, h_grid))
    grid = np.linspace(0.1, 0.9, 21)
    bh = fit_beta_at_points(grid, t, Z, Y, h)
    assert bh.shape == (21, 2) and np.all(np.isfinite(bh))
    comp = estimate_plugin_components(grid, t, Z, Y, h)
    se = standard_error_curve(np.array([0.0, 1.0]), comp["fhat"], comp["omega"],
                              comp["lambda"], n, h)
    assert se.shape == (21,) and np.all(se > 0)
    gamma = simulate_gamma(0.05, h, n, grid, B=100, seed=4)
    assert 1.5 < gamma < 5.0


def test_caihu_2025_smoke():
    n, m = 40, 5
    X, Y, sid = gen_data_part3(n, m, "normal", seed=5)
    grid, mhat, lo, hi, meta = caihu_scb(X, Y, sid, n, alpha=0.1, B=50,
                                         grid_size=101, degree=3, seed=6)
    assert grid.shape == (101,)
    assert np.all(hi >= lo)
    assert meta["Jn"] >= 1
    assert np.all(np.isfinite(mhat))
    truth = m_true(grid)
    cov = bool(np.all((truth >= lo) & (truth <= hi)))
    assert cov in (True, False)


def test_part2_data_gen():
    subj, t, Z, Y, sigma = gen_data_part2(n=30, m=5, snr=5.0, seed=7, p=5)
    assert Z.shape[1] == 5
    assert sigma > 0
    assert subj.size == t.size == Y.size == Z.shape[0]
