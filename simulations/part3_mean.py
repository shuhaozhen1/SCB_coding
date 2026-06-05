"""Part 3: mean function SCB. Ours vs Cai & Hu (2025) B-spline.

Settings (default): n=200, m in {5, 10, 15}, dist in {normal, uniform,
laplace}, c_rule = 4.5, R = 1000.

Cao 2018's plug-in is *not* a designed comparator for the mean-function
problem (its formulation targets local-linear varying-coefficient regression)
and is therefore omitted; the natural comparator is the Cai-Hu (2025)
B-spline SCB.
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
from comparators.cao_2018 import select_bandwidth_loso
from comparators.caihu_2025 import (m_true, scb_from_dataset as caihu_scb)
from simulations._common import gen_data_part3

ALPHA = 0.05
C_RULE = 4.5
H_GRID = np.linspace(0.06, 0.20, 8)
T_GRID = np.linspace(0.05, 0.95, 181)
N_BOOT = 300
BETA_TRUE = m_true(T_GRID)
BT2 = BETA_TRUE.reshape(-1, 1)


def _one_rep(args):
    rep, n, m, dist, base_seed = args
    X, Y, sid = gen_data_part3(n, m, dist, base_seed + rep)
    Z = np.ones((X.size, 1))
    h_hat = float(select_bandwidth_loso(sid, X, Z, Y, H_GRID))
    # Ours
    h_inf = max(h_hat * (1.0 - C_RULE / np.sqrt(n)), 0.01)
    df = pd.DataFrame({"id": sid.astype(int), "t": X, "y": Y,
                       "x0": np.ones(X.size)})
    lc = LocalLinearConfig(bandwidth=h_hat, ridge=1e-6, weighting="subject",
                           scaled_design=True)
    bc = BootstrapConfig(n_boot=N_BOOT, alpha=ALPHA,
                         seed=base_seed + rep + 10**6,
                         center_influence=True, residual_method="loo")
    scb = local_linear_vcm_scb(data=df, q=1, grid=T_GRID, local_config=lc,
                               bootstrap_config=bc, components=[0],
                               bandwidth_inf=h_inf)
    ours_cov = band_covers_truth(scb, BT2, components=[0])
    ours_wid = float(np.nanmean(scb.upper[:, 0] - scb.lower[:, 0]))
    # Cai-Hu B-spline
    grid_f, mhat_f, lo_f, up_f, meta = caihu_scb(
        X, Y, sid, n, alpha=ALPHA, B=500, grid_size=2001, degree=3,
        seed=base_seed + rep + 5*10**6,
    )
    truth_f = m_true(grid_f)
    caihu_cov = bool(np.all((truth_f >= lo_f) & (truth_f <= up_f)))
    caihu_wid = float(np.mean(up_f - lo_f))
    return {"rep": rep, "h_hat": float(h_hat),
            "ours_cov": ours_cov, "ours_wid": ours_wid,
            "caihu_cov": caihu_cov, "caihu_wid": caihu_wid}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--m-list", type=int, nargs="+", default=[5, 10, 15])
    p.add_argument("--dist-list", type=str, nargs="+",
                   default=["normal", "uniform", "laplace"])
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out-dir", type=str, default="results/part3")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Part 3: mean function. n={args.n}, m in {args.m_list}, "
          f"dist in {args.dist_list}, c={C_RULE}, R={args.reps}, "
          f"workers={args.workers}")
    rows = []
    settings = [(args.n, m, d) for m in args.m_list for d in args.dist_list]
    for i, (n, m, d) in enumerate(settings, 1):
        base_seed = 30_000_000 + i * 100_000
        print(f"[P3 {i}/{len(settings)}] n={n} m={m:>2} {d:>8} ... ",
              end="", flush=True)
        t0 = time.perf_counter()
        args_list = [(r, n, m, d, base_seed) for r in range(args.reps)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_one_rep, args_list, chunksize=8))
        el = time.perf_counter() - t0
        oc = np.array([r["ours_cov"] for r in results])
        ow = np.array([r["ours_wid"] for r in results])
        cc = np.array([r["caihu_cov"] for r in results])
        cw = np.array([r["caihu_wid"] for r in results])
        h_hat = np.array([r["h_hat"] for r in results])
        out = {"part": 3, "n": n, "m": m, "dist": d, "n_reps": args.reps,
               "c_rule": C_RULE,
               "h_hat_mean": float(h_hat.mean()),
               "ours_coverage": float(oc.mean()),
               "ours_cov_mcse": float(np.sqrt(oc.mean()*(1-oc.mean())/args.reps)),
               "ours_width": float(np.nanmean(ow)),
               "caihu_coverage": float(cc.mean()),
               "caihu_cov_mcse": float(np.sqrt(cc.mean()*(1-cc.mean())/args.reps)),
               "caihu_width": float(cw.mean()),
               "elapsed_sec": float(el), "per_rep": results}
        with open(out_dir / f"n{n}_m{m}_{d}.json", "w") as f:
            json.dump(out, f, indent=2)
        rows.append({k: v for k, v in out.items() if k != "per_rep"})
        print(f"Ours {out['ours_coverage']:.3f}/{out['ours_width']:.3f} | "
              f"Cai-Hu {out['caihu_coverage']:.3f}/{out['caihu_width']:.3f} "
              f"| {el:.0f}s")
    pd.DataFrame(rows).to_csv(out_dir / "part3_summary.csv", index=False)


if __name__ == "__main__":
    main()
