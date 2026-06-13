"""Part 1: single-coefficient VCM (q=2). Ours vs Cao (2018) per-coef SCB.

Settings (default): n in {200, 300}, m in {5, 10, 15}, noise in
{gauss, uniform, laplace}, R = 1000, target: simultaneous 95% coverage of
beta_1(t). Inference bandwidth: h_inf = h_est*(1 - 8/(sqrt(n)*(m_bar/q)^{1/4}))
with q=2 (see simulations._common.undersmooth_h_inf).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scb import (BootstrapConfig, LocalLinearConfig,
                 band_covers_truth, local_linear_vcm_scb)
from comparators.cao_2018 import (fit_beta_at_points, select_bandwidth_loso,
                                  simulate_gamma, estimate_plugin_components,
                                  standard_error_curve)
from simulations._common import (gen_data_part1, beta_true_part1,
                                 undersmooth_h_inf)

ALPHA = 0.05
C_INF = 8.0          # h_inf = h_est*(1 - C_INF/(sqrt(n)*(m_bar/q)^{1/4})), q=2
H_GRID = np.linspace(0.04, 0.16, 13)
T_GRID = np.linspace(0.025, 0.975, 96)   # step 0.01
N_BOOT = 300
BETA_TRUE = beta_true_part1(T_GRID)            # (G, 2)
A_BETA1 = np.array([0.0, 1.0])
BETA_TRUE_BETA1 = BETA_TRUE @ A_BETA1
BT2 = np.zeros((T_GRID.size, 2)); BT2[:, 1] = BETA_TRUE[:, 1]

_GAMMAS_BY_N: dict = {}

def _init_worker(gammas):
    global _GAMMAS_BY_N
    _GAMMAS_BY_N = gammas


def _one_rep(args):
    rep, n, m, noise, base_seed = args
    subj, t, Z, Y = gen_data_part1(n, m, noise, base_seed + rep)
    h_hat = select_bandwidth_loso(subj, t, Z, Y, H_GRID)
    gamma = _GAMMAS_BY_N[n][float(h_hat)]
    # Cao
    bh = fit_beta_at_points(T_GRID, t, Z, Y, h_hat)
    comp = estimate_plugin_components(T_GRID, t, Z, Y, h_hat)
    se = standard_error_curve(A_BETA1, comp["fhat"], comp["omega"], comp["lambda"], n, h_hat)
    est = bh @ A_BETA1; lo = est - gamma * se; hi = est + gamma * se
    cao_cov = bool(np.all((BETA_TRUE_BETA1 >= lo) & (BETA_TRUE_BETA1 <= hi)))
    cao_wid = float(np.mean(hi - lo))
    # Ours
    m_bar = Y.size / n
    h_inf = undersmooth_h_inf(h_hat, n, m_bar, q=2, c=C_INF)
    df = pd.DataFrame({"id": subj.astype(int), "t": t, "y": Y,
                       "x0": Z[:, 0], "x1": Z[:, 1]})
    lc = LocalLinearConfig(bandwidth=h_hat, ridge=1e-6, weighting="subject",
                           scaled_design=True)
    bc = BootstrapConfig(n_boot=N_BOOT, alpha=ALPHA, seed=base_seed + rep + 10**6,
                         center_influence=True, residual_method="loo")
    scb = local_linear_vcm_scb(data=df, q=2, grid=T_GRID, local_config=lc,
                               bootstrap_config=bc, components=[1],
                               bandwidth_inf=h_inf)
    ours_cov = band_covers_truth(scb, BT2, components=[1])
    ours_wid = float(np.nanmean(scb.upper[:, 1] - scb.lower[:, 1]))
    return {"rep": rep, "h_hat": float(h_hat),
            "cao_cov": cao_cov, "cao_wid": cao_wid,
            "ours_cov": ours_cov, "ours_wid": ours_wid}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-list", type=int, nargs="+", default=[200, 300])
    p.add_argument("--m-list", type=int, nargs="+", default=[5, 10, 15])
    p.add_argument("--noise-list", type=str, nargs="+",
                   default=["gauss", "uniform", "laplace"])
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out-dir", type=str, default="results/part1")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Precompute Cao gammas per n
    print(f"Part 1: n in {args.n_list}, m in {args.m_list}, "
          f"noise in {args.noise_list}, R={args.reps}, workers={args.workers}")
    gammas = {}
    for n in args.n_list:
        gammas[n] = {float(h): float(simulate_gamma(ALPHA, float(h), n, T_GRID,
                                                    B=2000,
                                                    seed=12345 + int(round(float(h)*1e6)) + 7*n))
                     for h in H_GRID}
        print(f"  Precomputed Cao gammas for n={n}")

    rows = []
    settings = [(n, m, ns) for n in args.n_list for m in args.m_list
                for ns in args.noise_list]
    for i, (n, m, ns) in enumerate(settings, 1):
        base_seed = 10_000_000 + i * 100_000
        print(f"[P1 {i:2d}/{len(settings)}] n={n} m={m:>2} {ns:>8} ... ",
              end="", flush=True)
        t0 = time.perf_counter()
        args_list = [(r, n, m, ns, base_seed) for r in range(args.reps)]
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker,
                                 initargs=(gammas,)) as ex:
            results = list(ex.map(_one_rep, args_list, chunksize=8))
        el = time.perf_counter() - t0
        cao_cov = np.array([r["cao_cov"] for r in results])
        cao_wid = np.array([r["cao_wid"] for r in results])
        our_cov = np.array([r["ours_cov"] for r in results])
        our_wid = np.array([r["ours_wid"] for r in results])
        h_hat = np.array([r["h_hat"] for r in results])
        out = {"part": 1, "n": n, "m": m, "noise": ns, "n_reps": args.reps,
               "c_inf": C_INF,
               "h_hat_mean": float(h_hat.mean()),
               "h_hat_std": float(h_hat.std(ddof=1)),
               "cao_coverage": float(cao_cov.mean()),
               "cao_cov_mcse": float(np.sqrt(cao_cov.mean()*(1-cao_cov.mean())/args.reps)),
               "cao_width": float(cao_wid.mean()),
               "ours_coverage": float(our_cov.mean()),
               "ours_cov_mcse": float(np.sqrt(our_cov.mean()*(1-our_cov.mean())/args.reps)),
               "ours_width": float(np.nanmean(our_wid)),
               "elapsed_sec": float(el),
               "per_rep": results}
        with open(out_dir / f"n{n}_m{m}_{ns}.json", "w") as f:
            json.dump(out, f, indent=2)
        rows.append({k: v for k, v in out.items() if k != "per_rep"})
        print(f"Cao {out['cao_coverage']:.3f}/{out['cao_width']:.3f} | "
              f"Ours {out['ours_coverage']:.3f}/{out['ours_width']:.3f} | {el:.0f}s")
    pd.DataFrame(rows).to_csv(out_dir / "part1_summary.csv", index=False)


if __name__ == "__main__":
    main()
