"""Generate representative band curves (one fixed replicate per setting) for
the illustration figures. For each simulation cell we draw a single dataset
with the production base seed (rep 0), fit Ours and the matched comparator,
and store the truth, point estimate, and lower/upper SCB curves to npz.

Outputs: results/plotdata/{part1,part2,part3}.npz
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scb import (BootstrapConfig, LocalLinearConfig, local_linear_vcm_scb)
from scb.estimation import local_linear_vcm_fit
from comparators.cao_2018 import (fit_beta_at_points, select_bandwidth_loso,
                                  simulate_gamma, estimate_plugin_components,
                                  standard_error_curve)
from comparators.caihu_2025 import m_true, scb_from_dataset as caihu_scb
from simulations._common import (gen_data_part1, gen_data_part2, gen_data_part3,
                                 beta_true_part1, beta_true_part2,
                                 undersmooth_h_inf)

ALPHA = 0.05
H_GRID = np.linspace(0.04, 0.16, 13)
T_GRID = np.linspace(0.025, 0.975, 96)
N_BOOT = 300
OUT = ROOT / "results" / "plotdata"; OUT.mkdir(parents=True, exist_ok=True)


def _loso_h_q(df, n, q):
    best_h, best_cv = None, np.inf
    for h in H_GRID:
        cv = 0.0; ntot = 0
        lc = LocalLinearConfig(bandwidth=float(h), ridge=1e-6, weighting="subject", scaled_design=True)
        for i in range(n):
            test = df[df["id"] == i]; train = df[df["id"] != i]
            if test.empty or train.empty: continue
            tt = test["t"].to_numpy()
            X = test[[f"x{k}" for k in range(q)]].to_numpy()
            y = test["y"].to_numpy()
            fit = local_linear_vcm_fit(data=train, q=q, grid=tt, config=lc)
            pred = np.sum(X * fit.beta_hat, axis=1)
            cv += float(np.nansum((y - pred) ** 2)); ntot += int(np.sum(np.isfinite(y - pred)))
        c = cv / max(1, ntot)
        if c < best_cv: best_cv, best_h = c, float(h)
    return best_h


# ---------------- Part 1 ----------------
def gen_part1():
    bt = beta_true_part1(T_GRID); beta1 = bt[:, 1]
    a = np.array([0.0, 1.0])
    store = {"t": T_GRID, "truth": beta1}
    i = 0
    for n in (200, 300):
        for m in (5, 10, 15):
            for ns in ("gauss", "uniform", "laplace"):
                i += 1
                base = 10_000_000 + i * 100_000
                subj, t, Z, Y = gen_data_part1(n, m, ns, base)
                m_bar = Y.size / n
                h = float(select_bandwidth_loso(subj, t, Z, Y, H_GRID))
                h_inf = undersmooth_h_inf(h, n, m_bar, q=2)
                df = pd.DataFrame({"id": subj.astype(int), "t": t, "y": Y, "x0": Z[:, 0], "x1": Z[:, 1]})
                lc = LocalLinearConfig(bandwidth=h, ridge=1e-6, weighting="subject", scaled_design=True)
                bc = BootstrapConfig(n_boot=N_BOOT, alpha=ALPHA, seed=base + 10**6,
                                     center_influence=True, residual_method="loo")
                scb = local_linear_vcm_scb(data=df, q=2, grid=T_GRID, local_config=lc,
                                           bootstrap_config=bc, components=[1], bandwidth_inf=h_inf)
                gamma = float(simulate_gamma(ALPHA, h, n, T_GRID, B=2000, seed=12345 + int(round(h*1e6)) + 7*n))
                bh = fit_beta_at_points(T_GRID, t, Z, Y, h)
                comp = estimate_plugin_components(T_GRID, t, Z, Y, h)
                se = standard_error_curve(a, comp["fhat"], comp["omega"], comp["lambda"], n, h)
                bc_est = bh @ a
                key = f"n{n}_m{m}_{ns}"
                store[f"{key}__ours_est"] = scb.beta_hat[:, 1]
                store[f"{key}__ours_lo"] = scb.lower[:, 1]
                store[f"{key}__ours_hi"] = scb.upper[:, 1]
                store[f"{key}__cao_est"] = bc_est
                store[f"{key}__cao_lo"] = bc_est - gamma * se
                store[f"{key}__cao_hi"] = bc_est + gamma * se
                print(f"P1 {key} done", flush=True)
    np.savez(OUT / "part1.npz", **store)


# ---------------- Part 2 ----------------
def gen_part2():
    P = 5; SNR = 5.0
    bt = beta_true_part2(T_GRID, P)
    store = {"t": T_GRID}
    for k in range(P): store[f"truth_{k}"] = bt[:, k]
    for m in (5, 10, 15):
        base = 20_000_000 + m * 100_000
        subj, t, Z, Y, _ = gen_data_part2(200, m, SNR, base, p=P)
        m_bar = Y.size / 200
        df = pd.DataFrame({"id": subj.astype(int), "t": t, "y": Y})
        for k in range(P): df[f"x{k}"] = Z[:, k]
        h = _loso_h_q(df, 200, P)
        h_inf = undersmooth_h_inf(h, 200, m_bar, q=P)
        lc = LocalLinearConfig(bandwidth=h, ridge=1e-6, weighting="subject", scaled_design=True)
        bc = BootstrapConfig(n_boot=N_BOOT, alpha=ALPHA, seed=base + 10**6,
                             center_influence=True, residual_method="loo")
        scb = local_linear_vcm_scb(data=df, q=P, grid=T_GRID, local_config=lc,
                                   bootstrap_config=bc, components=list(range(P)), bandwidth_inf=h_inf)
        # Bonferroni-Cao at level alpha/P
        gamma_b = float(simulate_gamma(ALPHA / P, h, 200, T_GRID, B=2000, seed=54321 + int(round(h*1e6)) + 11*200))
        bh = fit_beta_at_points(T_GRID, t, Z, Y, h)
        comp = estimate_plugin_components(T_GRID, t, Z, Y, h)
        for k in range(P):
            a = np.zeros(P); a[k] = 1.0
            se = standard_error_curve(a, comp["fhat"], comp["omega"], comp["lambda"], 200, h)
            est = bh @ a
            key = f"m{m}_k{k}"
            store[f"{key}__ours_est"] = scb.beta_hat[:, k]
            store[f"{key}__ours_lo"] = scb.lower[:, k]
            store[f"{key}__ours_hi"] = scb.upper[:, k]
            store[f"{key}__bonf_est"] = est
            store[f"{key}__bonf_lo"] = est - gamma_b * se
            store[f"{key}__bonf_hi"] = est + gamma_b * se
        print(f"P2 m={m} done", flush=True)
    np.savez(OUT / "part2.npz", **store)


# ---------------- Part 3 ----------------
def gen_part3():
    truth = m_true(T_GRID)
    store = {"t": T_GRID, "truth": truth}
    i = 0
    for n in (200, 300):
        for m in (5, 10, 15):
            for dist in ("normal", "uniform", "laplace"):
                i += 1
                base = 30_000_000 + i * 100_000
                X, Y, sid = gen_data_part3(n, m, dist, base)
                m_bar = Y.size / n
                h = float(select_bandwidth_loso(sid, X, np.ones((X.size, 1)), Y, H_GRID))
                h_inf = undersmooth_h_inf(h, n, m_bar, q=1)
                df = pd.DataFrame({"id": sid.astype(int), "t": X, "y": Y, "x0": np.ones(X.size)})
                lc = LocalLinearConfig(bandwidth=h, ridge=1e-6, weighting="subject", scaled_design=True)
                bc = BootstrapConfig(n_boot=N_BOOT, alpha=ALPHA, seed=base + 10**6,
                                     center_influence=True, residual_method="loo")
                scb = local_linear_vcm_scb(data=df, q=1, grid=T_GRID, local_config=lc,
                                           bootstrap_config=bc, components=[0], bandwidth_inf=h_inf)
                gf, mf, lof, upf, _ = caihu_scb(X, Y, sid, n, alpha=ALPHA, B=500,
                                                grid_size=2001, degree=3, seed=base + 5*10**6)
                key = f"n{n}_m{m}_{dist}"
                store[f"{key}__ours_est"] = scb.beta_hat[:, 0]
                store[f"{key}__ours_lo"] = scb.lower[:, 0]
                store[f"{key}__ours_hi"] = scb.upper[:, 0]
                # Cai-Hu on its own grid: interpolate to T_GRID for plotting
                store[f"{key}__ch_est"] = np.interp(T_GRID, gf, mf)
                store[f"{key}__ch_lo"] = np.interp(T_GRID, gf, lof)
                store[f"{key}__ch_hi"] = np.interp(T_GRID, gf, upf)
                print(f"P3 {key} done", flush=True)
    np.savez(OUT / "part3.npz", **store)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    if 1 in args.parts: gen_part1()
    if 2 in args.parts: gen_part2()
    if 3 in args.parts: gen_part3()
    print("plot-curve generation done")
