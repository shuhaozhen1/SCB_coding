"""PBC VCM application: joint 95% SCB over (beta_0, beta_1, beta_2) for
the log-bilirubin trajectory with covariates D-penicillamine treatment and
standardized baseline age.

Model:
    log(bili_ij) = beta_0(t_ij) + beta_1(t_ij) trt_i + beta_2(t_ij) age_i + eps_ij,
where t = day / 3000 (normalized to [0, 1]).

Outputs (under results/realdata/):
    pbc_vcm_summary.json
    pbc_vcm_bands.png
    pbc_vcm_band_data.csv
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scb import (BootstrapConfig, LocalLinearConfig, local_linear_vcm_scb)
from comparators.cao_2018 import (fit_beta_at_points, select_bandwidth_loso,
                                  simulate_gamma, estimate_plugin_components,
                                  standard_error_curve)
from simulations._common import undersmooth_h_inf

DATA_PATH = ROOT / "data" / "pbcseq.csv"
OUT_DIR = ROOT / "results" / "realdata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAP_DAYS = 3000.0
ALPHA = 0.05
C_INF = 8.0          # h_inf = h_est*(1 - C_INF/(sqrt(n)*(m_bar/q)^{1/4})), q=3
H_GRID = np.linspace(0.04, 0.16, 13)
T_GRID = np.linspace(0.025, 0.975, 96)   # step 0.01
N_BOOT = 3000
COEF_NAMES = ["beta0_intercept", "beta1_treatment", "beta2_age"]


def load_pbc():
    df = pd.read_csv(DATA_PATH)
    df = df[["id", "day", "trt", "age", "bili"]].dropna()
    df = df[df["day"] <= CAP_DAYS].copy()
    df["t"] = df["day"] / CAP_DAYS
    df["logbili"] = np.log(df["bili"].clip(lower=1e-3))
    df["age_std"] = (df["age"] - df["age"].mean()) / df["age"].std()
    df["trt01"] = (df["trt"] == df["trt"].max()).astype(float)
    return df


def main():
    df = load_pbc()
    uniq = np.unique(df["id"].to_numpy())
    df["pid"] = df["id"].map({v: k for k, v in enumerate(uniq)}).astype(int)
    n = len(uniq); sid = df["pid"].to_numpy()
    t = df["t"].to_numpy(); Y = df["logbili"].to_numpy()
    Z = np.column_stack([np.ones(len(df)), df["trt01"].to_numpy(),
                         df["age_std"].to_numpy()])
    print(f"PBC VCM: n={n} patients, {len(df)} obs, p=3 covariates")

    m_bar = len(df) / n
    h_est = float(select_bandwidth_loso(sid, t, Z, Y, H_GRID))
    h_inf = undersmooth_h_inf(h_est, n, m_bar, q=3, c=C_INF)
    print(f"LOSO h_est={h_est:.4f}, h_inf={h_inf:.4f}")

    gamma_cao = float(simulate_gamma(ALPHA, h_est, n, T_GRID, B=2000,
                                     seed=12345 + int(round(h_est * 1e6))))
    cao_beta_grid = fit_beta_at_points(T_GRID, t, Z, Y, h_est)
    comp_cao = estimate_plugin_components(T_GRID, t, Z, Y, h_est)

    # Joint Ours SCB over all 3 coefficients
    df_o = pd.DataFrame({"id": sid.astype(int), "t": t, "y": Y,
                         "x0": Z[:, 0], "x1": Z[:, 1], "x2": Z[:, 2]})
    lc = LocalLinearConfig(bandwidth=h_est, ridge=1e-6, weighting="subject",
                           scaled_design=True)
    bc = BootstrapConfig(n_boot=N_BOOT, alpha=ALPHA, seed=2026,
                         center_influence=True, residual_method="loo")
    scb_joint = local_linear_vcm_scb(data=df_o, q=3, grid=T_GRID,
                                     local_config=lc, bootstrap_config=bc,
                                     components=[0, 1, 2], bandwidth_inf=h_inf)
    qc_joint = float(scb_joint.qcrit)
    print(f"Ours joint qcrit = {qc_joint:.3f}; Cao gamma = {gamma_cao:.3f}")

    summary = {
        "setting": {"n_patients": n, "n_obs": int(len(df)), "p": 3,
                    "cap_days": CAP_DAYS, "alpha": ALPHA, "c_inf": C_INF,
                    "h_est": h_est, "h_inf": h_inf, "n_boot": N_BOOT,
                    "joint_qcrit": qc_joint, "cao_gamma": gamma_cao},
        "coefficients": {},
    }
    band_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for k, name in enumerate(COEF_NAMES):
        b_o = scb_joint.beta_hat[:, k]
        lo_o = scb_joint.lower[:, k]; hi_o = scb_joint.upper[:, k]
        a_k = np.zeros(3); a_k[k] = 1.0
        b_c = cao_beta_grid @ a_k
        se_c = standard_error_curve(a_k, comp_cao["fhat"], comp_cao["omega"],
                                    comp_cao["lambda"], n, h_est)
        lo_c = b_c - gamma_cao * se_c; hi_c = b_c + gamma_cao * se_c
        len_o = float(np.nanmean(hi_o - lo_o))
        len_c = float(np.mean(hi_c - lo_c))
        excl_o = bool(np.any((lo_o > 0) | (hi_o < 0)))
        excl_c = bool(np.any((lo_c > 0) | (hi_c < 0)))
        summary["coefficients"][name] = {
            "ours_joint_avg_band_len": len_o,
            "cao_marginal_avg_band_len": len_c,
            "len_ratio_ours_over_cao": len_o / len_c if len_c > 0 else None,
            "ours_excludes_zero": excl_o, "cao_excludes_zero": excl_c,
        }
        for j in range(T_GRID.size):
            band_rows.append({"coef": name, "t": float(T_GRID[j]),
                              "ours_beta": float(b_o[j]),
                              "ours_lower": float(lo_o[j]),
                              "ours_upper": float(hi_o[j]),
                              "cao_beta": float(b_c[j]),
                              "cao_lower": float(lo_c[j]),
                              "cao_upper": float(hi_c[j])})
        ax = axes[k]
        ax.plot(T_GRID, b_o, "C0-", lw=2, label="Ours joint")
        ax.fill_between(T_GRID, lo_o, hi_o, color="C0", alpha=0.25,
                        label="Ours 95% joint SCB")
        ax.plot(T_GRID, b_c, "C1--", lw=1.5, label="Cao marginal")
        ax.plot(T_GRID, lo_c, "C1:", lw=1.0)
        ax.plot(T_GRID, hi_c, "C1:", lw=1.0, label="Cao 95% marginal")
        ax.axhline(0.0, color="gray", lw=0.6)
        ax.set_xlabel("t = day / 3000"); ax.set_title(name)
        if k == 0: ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"PBC VCM (n={n}): Ours JOINT 95% SCB vs Cao marginal 95% SCB")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / "pbc_vcm_bands.png", dpi=150, bbox_inches="tight")

    with open(OUT_DIR / "pbc_vcm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(band_rows).to_csv(OUT_DIR / "pbc_vcm_band_data.csv", index=False)

    print()
    print(f"{'coefficient':>22} | {'Ours wid':>9} {'Cao wid':>8} {'ratio':>6} "
          f"{'Ours sig':>9} {'Cao sig':>8}")
    for k, name in enumerate(COEF_NAMES):
        s = summary["coefficients"][name]
        print(f"{name:>22} | {s['ours_joint_avg_band_len']:>9.4f} "
              f"{s['cao_marginal_avg_band_len']:>8.4f} "
              f"{s['len_ratio_ours_over_cao']:>6.3f} "
              f"{str(s['ours_excludes_zero']):>9} "
              f"{str(s['cao_excludes_zero']):>8}")


if __name__ == "__main__":
    main()
