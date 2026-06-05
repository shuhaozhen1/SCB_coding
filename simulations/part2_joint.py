"""Part 2: joint SCB over all q=5 coefficient functions of a VCM. Ours only.

Cao 2018's plug-in SCB is per-target by construction and is not natively
applicable to a joint multi-coefficient SCB; we therefore omit it as
comparator.

Settings (default): n=200, m in {5, 10, 15}, c_rule = 6.5, R = 1000.
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
from simulations._common import gen_data_part2, beta_true_part2

ALPHA = 0.05
C_RULE = 6.5
P = 5
SNR = 5.0
H_GRID = np.linspace(0.06, 0.20, 8)
T_GRID = np.linspace(0.025, 0.975, 191)
N_BOOT = 300
BETA_TRUE = beta_true_part2(T_GRID, P)         # (G, P)


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
    rep, n, m, base_seed = args
    subj, t, Z, Y, _ = gen_data_part2(n, m, SNR, base_seed + rep, p=P)
    df = _df_from_arrays(subj, t, Z, Y)
    h_est = _loso_h(subj, t, Z, Y, df, n)
    h_inf = max(h_est * (1.0 - C_RULE / np.sqrt(n)), 0.01)
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
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--m-list", type=int, nargs="+", default=[5, 10, 15])
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out-dir", type=str, default="results/part2")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Part 2: q={P} joint SCB. n={args.n}, m in {args.m_list}, "
          f"c={C_RULE}, R={args.reps}, workers={args.workers}")
    rows = []
    for i, m in enumerate(args.m_list, 1):
        base_seed = 20_000_000 + m * 100_000
        print(f"[P2 {i}/{len(args.m_list)}] m={m:>2} ... ", end="", flush=True)
        t0 = time.perf_counter()
        args_list = [(r, args.n, m, base_seed) for r in range(args.reps)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_one_rep, args_list, chunksize=4))
        el = time.perf_counter() - t0
        cov = np.array([r["joint_cov"] for r in results])
        widths = np.array([r["widths"] for r in results])
        h_hat = np.array([r["h_hat"] for r in results])
        out = {"part": 2, "n": args.n, "m": m, "p": P, "n_reps": args.reps,
               "c_rule": C_RULE, "snr": SNR,
               "h_hat_mean": float(h_hat.mean()),
               "h_hat_std": float(h_hat.std(ddof=1)),
               "joint_coverage": float(cov.mean()),
               "joint_cov_mcse": float(np.sqrt(cov.mean()*(1-cov.mean())/args.reps)),
               "width_per_dim": [float(widths[:, k].mean()) for k in range(P)],
               "width_avg": float(widths.mean()),
               "elapsed_sec": float(el), "per_rep": results}
        with open(out_dir / f"n{args.n}_m{m}.json", "w") as f:
            json.dump(out, f, indent=2)
        rows.append({k: v for k, v in out.items() if k != "per_rep"})
        print(f"joint {out['joint_coverage']:.3f} wid {out['width_avg']:.3f} "
              f"| {el:.0f}s")
    pd.DataFrame(rows).to_csv(out_dir / "part2_summary.csv", index=False)


if __name__ == "__main__":
    main()
