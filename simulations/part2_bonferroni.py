"""Part 2 comparator: Bonferroni-corrected Cao (2018) per-coefficient SCB.

To make Cao's per-target plug-in band applicable to the JOINT problem over all
q=5 coefficients, we build for each coefficient k a Cao SCB at the Bonferroni
level alpha/q (a per-coefficient (1 - alpha/q) band) and take their
intersection. By the union bound the joint simultaneous coverage is
    P( beta_k(t) in [L_k(t), U_k(t)]  for all k, all t ) >= 1 - q*(alpha/q) = 1-alpha.
This is valid but conservative: each band uses the critical value at level
alpha/q = 0.01, so the bands are markedly wider than a genuinely joint SCB.

We report joint coverage and the per-coefficient-averaged width, using the
SAME replicate seeds as part2_joint.py so the rows align with Ours.

Cao is kept exactly as implemented elsewhere (no modification).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scb import LocalLinearConfig
from scb.estimation import local_linear_vcm_fit
from comparators.cao_2018 import (fit_beta_at_points, simulate_gamma,
                                  estimate_plugin_components, standard_error_curve)
from simulations._common import gen_data_part2, beta_true_part2

ALPHA = 0.05
P = 5
SNR = 5.0
H_GRID = np.linspace(0.04, 0.16, 13)
T_GRID = np.linspace(0.025, 0.975, 96)
BETA_TRUE = beta_true_part2(T_GRID, P)            # (G, P)
NOISE_SEED_OFFSET = {"gauss": 0, "uniform": 10_000, "laplace": 20_000}

_GAMMA_CACHE = {}


def _loso_h(df_all, n):
    best_h, best_cv = None, np.inf
    for h in H_GRID:
        cv = 0.0; ntot = 0
        lc = LocalLinearConfig(bandwidth=float(h), ridge=1e-6,
                               weighting="subject", scaled_design=True)
        for i in range(n):
            test = df_all[df_all["id"] == i]; train = df_all[df_all["id"] != i]
            if test.empty or train.empty: continue
            t_test = test["t"].to_numpy()
            X_test = test[[f"x{k}" for k in range(P)]].to_numpy()
            y_test = test["y"].to_numpy()
            fit = local_linear_vcm_fit(data=train, q=P, grid=t_test, config=lc)
            pred = np.sum(X_test * fit.beta_hat, axis=1)
            cv += float(np.nansum((y_test - pred) ** 2))
            ntot += int(np.sum(np.isfinite(y_test - pred)))
        c = cv / max(1, ntot)
        if c < best_cv: best_cv, best_h = c, float(h)
    return best_h


def _gamma_bonf(h, n, seed):
    key = (round(float(h), 4), n)
    g = _GAMMA_CACHE.get(key)
    if g is None:
        # Bonferroni: per-coefficient level alpha/P
        g = float(simulate_gamma(ALPHA / P, float(h), n, T_GRID, B=2000, seed=seed))
        _GAMMA_CACHE[key] = g
    return g


def _one_rep(args):
    rep, n, m, noise, base = args
    subj, t, Z, Y, _ = gen_data_part2(n, m, SNR, base + rep, p=P, noise=noise)
    df = pd.DataFrame({"id": subj.astype(int), "t": t, "y": Y})
    for k in range(P): df[f"x{k}"] = Z[:, k]
    h = _loso_h(df, n)
    gamma = _gamma_bonf(h, n, 54321 + int(round(h * 1e6)) + 11 * n)
    bh = fit_beta_at_points(T_GRID, t, Z, Y, h)          # (G, P)
    comp = estimate_plugin_components(T_GRID, t, Z, Y, h)
    all_ok = True
    widths = []
    for k in range(P):
        a = np.zeros(P); a[k] = 1.0
        se = standard_error_curve(a, comp["fhat"], comp["omega"], comp["lambda"], n, h)
        est = bh @ a
        lo = est - gamma * se; hi = est + gamma * se
        ok_k = bool(np.all((BETA_TRUE[:, k] >= lo) & (BETA_TRUE[:, k] <= hi)))
        all_ok = all_ok and ok_k
        widths.append(float(np.mean(hi - lo)))
    return {"rep": rep, "h_hat": float(h), "joint_cov": all_ok,
            "width_avg": float(np.mean(widths)), "gamma_bonf": gamma}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="Optional single-n shortcut; overrides --n-list.")
    ap.add_argument("--n-list", type=int, nargs="+", default=[200, 300])
    ap.add_argument("--m-list", type=int, nargs="+", default=[5, 10, 15])
    ap.add_argument("--noise-list", type=str, nargs="+",
                    default=["gauss", "uniform", "laplace"],
                    choices=["gauss", "uniform", "laplace"])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", type=str, default="results/part2_bonferroni")
    args = ap.parse_args()
    n_list = [args.n] if args.n is not None else args.n_list
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Part 2 Bonferroni-Cao: per-coef level alpha/{P}={ALPHA/P:.3f}, "
          f"n in {n_list}, m in {args.m_list}, noise in {args.noise_list}, "
          f"R={args.reps}, workers={args.workers}")
    rows = []
    total = len(n_list) * len(args.m_list) * len(args.noise_list)
    idx = 0
    for n in n_list:
        for m in args.m_list:
            for noise in args.noise_list:
                idx += 1
                base = (20_000_000 + m * 100_000 +
                        (n - 200) * 1_000_000 + NOISE_SEED_OFFSET[noise])
                print(f"[P2-Bonf {idx}/{total}] n={n}, m={m:>2}, noise={noise} ... ",
                      end="", flush=True)
                t0 = time.perf_counter()
                args_list = [(r, n, m, noise, base) for r in range(args.reps)]
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    results = list(ex.map(_one_rep, args_list, chunksize=4))
                el = time.perf_counter() - t0
                cov = np.array([r["joint_cov"] for r in results], float)
                wid = np.array([r["width_avg"] for r in results], float)
                gam = np.mean([r["gamma_bonf"] for r in results])
                out = {"part": 2, "method": "cao_bonferroni", "n": n, "m": m,
                       "noise": noise, "p": P, "n_reps": args.reps,
                       "alpha_per_coef": ALPHA / P,
                       "joint_coverage": float(cov.mean()),
                       "joint_cov_mcse": float(np.sqrt(cov.mean()*(1-cov.mean())/args.reps)),
                       "width_avg": float(np.nanmean(wid)),
                       "gamma_bonf_mean": float(gam),
                       "elapsed_sec": float(el)}
                with open(out_dir / f"n{n}_m{m}_{noise}.json", "w") as f:
                    json.dump(out, f, indent=2)
                rows.append(out)
                print(f"joint {out['joint_coverage']:.3f} wid {out['width_avg']:.3f} "
                      f"(gamma={gam:.2f}) | {el:.0f}s", flush=True)
    pd.DataFrame(rows).sort_values(["n", "m", "noise"]).to_csv(
        out_dir / "part2_bonferroni_summary.csv", index=False)


if __name__ == "__main__":
    main()
