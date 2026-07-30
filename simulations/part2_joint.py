"""Part 2: joint SCB over all q=5 coefficient functions of a VCM. Ours only.

Cao 2018's plug-in SCB is per-target by construction and is not natively
applicable to a joint multi-coefficient SCB; we therefore omit it as
comparator.

Settings (default): n in {200, 300}, m in {5, 10, 15}, observation noise in
{Gaussian, uniform, Laplace}, R = 1000. Inference bandwidth:
h_inf = h_est*(1 - 8/(sqrt(n)*(m_bar/q)^{1/4})) with q=5 (joint over all
coefficients); see simulations._common.undersmooth_h_inf.
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
from scb.estimation import local_linear_vcm_fit
from simulations._common import (gen_data_part2, beta_true_part2,
                                 undersmooth_h_inf)

ALPHA = 0.05
C_INF = 8.0          # h_inf = h_est*(1 - C_INF/(sqrt(n)*(m_bar/q)^{1/4})), q=P
P = 5
SNR = 5.0
H_GRID = np.linspace(0.04, 0.16, 13)
T_GRID = np.linspace(0.025, 0.975, 96)   # step 0.01
N_BOOT = 300
BETA_TRUE = beta_true_part2(T_GRID, P)         # (G, P)
NOISE_SEED_OFFSET = {"gauss": 0, "uniform": 10_000, "laplace": 20_000}


def _df_from_arrays(subj, t, Z, Y):
    df = pd.DataFrame({"id": subj.astype(int), "t": t, "y": Y})
    for k in range(P):
        df[f"x{k}"] = Z[:, k]
    return df


def _loso_h(subj, t, Z, Y, df_all, n):
    """Leave-one-subject-out CV on H_GRID using the scb local-linear fit."""
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


def _one_rep(args):
    rep, n, m, noise, base_seed = args
    subj, t, Z, Y, _ = gen_data_part2(n, m, SNR, base_seed + rep, p=P, noise=noise)
    df = _df_from_arrays(subj, t, Z, Y)
    h_est = _loso_h(subj, t, Z, Y, df, n)
    m_bar = Y.size / n
    h_inf = undersmooth_h_inf(h_est, n, m_bar, q=P, c=C_INF)
    lc = LocalLinearConfig(bandwidth=h_est, ridge=1e-6, weighting="subject",
                           scaled_design=True)
    bc = BootstrapConfig(n_boot=N_BOOT, alpha=ALPHA, seed=base_seed + rep + 10**6,
                         center_influence=True, residual_method="loo")
    scb = local_linear_vcm_scb(data=df, q=P, grid=T_GRID, local_config=lc,
                               bootstrap_config=bc, components=list(range(P)),
                               bandwidth_inf=h_inf)
    cov = band_covers_truth(scb, BETA_TRUE, components=list(range(P)))
    widths = [float(np.nanmean(scb.upper[:, k] - scb.lower[:, k])) for k in range(P)]
    return {"rep": rep, "h_hat": float(h_est), "joint_cov": cov,
            "widths": widths, "qcrit": float(scb.qcrit)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None,
                   help="Optional single-n shortcut; overrides --n-list.")
    p.add_argument("--n-list", type=int, nargs="+", default=[200, 300])
    p.add_argument("--m-list", type=int, nargs="+", default=[5, 10, 15])
    p.add_argument("--noise-list", type=str, nargs="+",
                   default=["gauss", "uniform", "laplace"],
                   choices=["gauss", "uniform", "laplace"])
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out-dir", type=str, default="results/part2")
    args = p.parse_args()

    n_list = [args.n] if args.n is not None else args.n_list
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Part 2: q={P} joint SCB. n in {n_list}, m in {args.m_list}, "
          f"noise in {args.noise_list}, c_inf={C_INF}, R={args.reps}, "
          f"workers={args.workers}")
    rows = []
    total = len(n_list) * len(args.m_list) * len(args.noise_list)
    idx = 0
    for n in n_list:
        for m in args.m_list:
            for noise in args.noise_list:
                idx += 1
                # Preserve the original n=200 Gaussian seeds, then offset by n and noise.
                base_seed = (20_000_000 + m * 100_000 +
                             (n - 200) * 1_000_000 + NOISE_SEED_OFFSET[noise])
                print(f"[P2 {idx}/{total}] n={n}, m={m:>2}, noise={noise} ... ",
                      end="", flush=True)
                t0 = time.perf_counter()
                args_list = [(r, n, m, noise, base_seed) for r in range(args.reps)]
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    results = list(ex.map(_one_rep, args_list, chunksize=4))
                el = time.perf_counter() - t0
                cov = np.array([r["joint_cov"] for r in results])
                widths = np.array([r["widths"] for r in results])
                h_hat = np.array([r["h_hat"] for r in results])
                out = {"part": 2, "n": n, "m": m, "noise": noise, "p": P,
                       "n_reps": args.reps, "c_inf": C_INF, "snr": SNR,
                       "h_hat_mean": float(h_hat.mean()),
                       "h_hat_std": float(h_hat.std(ddof=1)) if args.reps > 1 else 0.0,
                       "joint_coverage": float(cov.mean()),
                       "joint_cov_mcse": float(np.sqrt(cov.mean()*(1-cov.mean())/args.reps)),
                       "width_per_dim": [float(widths[:, k].mean()) for k in range(P)],
                       "width_avg": float(widths.mean()),
                       "elapsed_sec": float(el), "per_rep": results}
                with open(out_dir / f"n{n}_m{m}_{noise}.json", "w") as f:
                    json.dump(out, f, indent=2)
                rows.append({k: v for k, v in out.items() if k != "per_rep"})
                print(f"joint {out['joint_coverage']:.3f} wid {out['width_avg']:.3f} "
                      f"| {el:.0f}s")
    pd.DataFrame(rows).sort_values(["n", "m", "noise"]).to_csv(
        out_dir / "part2_summary.csv", index=False)


if __name__ == "__main__":
    main()
