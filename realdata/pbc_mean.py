"""PBC mean function application: 95% SCB for the population mean
log-bilirubin trajectory.

Model: log(bili_ij) = m(t_ij) + eps_ij,   t_ij = day_ij / 3000.

Methods compared:
    Ours        : local-linear SCB with q=1, LOSO h_est,
                  h_inf from the common undersmoothing rule, loo residuals,
                  Gaussian multipliers (B = 3000).
    Cai-Hu 2025 : B-spline SCB (Statistics and Computing 35:146).

Outputs (under results/realdata/):
    pbc_mean_summary.json
    pbc_mean_bands.png
    pbc_mean_band_data.csv
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
from scipy.interpolate import interp1d

from scb import (BootstrapConfig, LocalLinearConfig, local_linear_vcm_scb)
from comparators.cao_2018 import select_bandwidth_loso
from comparators.caihu_2025 import scb_from_dataset as caihu_scb
from simulations._common import undersmooth_h_inf

DATA_PATH = ROOT / "data" / "pbcseq.csv"
OUT_DIR = ROOT / "results" / "realdata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAP_DAYS = 3000.0
ALPHA = 0.05
C_INF = 8.0          # h_inf = h_est*(1 - C_INF/(sqrt(n)*(m_bar/q)^{1/4})), q=1
H_GRID = np.linspace(0.10, 0.22, 13)
T_GRID = np.linspace(0.025, 0.975, 96)   # step 0.01
N_BOOT = 3000


def load_pbc():
    df = pd.read_csv(DATA_PATH)
    df = df[["id", "day", "bili"]].dropna()
    df = df[df["day"] <= CAP_DAYS].copy()
    df["t"] = df["day"] / CAP_DAYS
    df["logbili"] = np.log(df["bili"].clip(lower=1e-3))
    return df


def main():
    df = load_pbc()
    uniq = np.unique(df["id"].to_numpy())
    df["pid"] = df["id"].map({v: k for k, v in enumerate(uniq)}).astype(int)
    n = len(uniq); sid = df["pid"].to_numpy()
    t = df["t"].to_numpy(); Y = df["logbili"].to_numpy()
    print(f"PBC mean: n={n} patients, {len(df)} obs "
          f"(median obs/patient = {int(df.groupby('pid').size().median())})")

    # Ours
    Z_int = np.ones((t.size, 1))
    m_bar = t.size / n
    h_est = float(select_bandwidth_loso(sid, t, Z_int, Y, H_GRID))
    h_inf = undersmooth_h_inf(h_est, n, m_bar, q=1, c=C_INF)
    print(f"Ours: LOSO h_est={h_est:.4f}, h_inf={h_inf:.4f}")

    df_o = pd.DataFrame({"id": sid.astype(int), "t": t, "y": Y,
                         "x0": np.ones(t.size)})
    lc = LocalLinearConfig(bandwidth=h_est, ridge=1e-6, weighting="subject",
                           scaled_design=True)
    bc = BootstrapConfig(n_boot=N_BOOT, alpha=ALPHA, seed=2026,
                         center_influence=True, residual_method="loo")
    scb = local_linear_vcm_scb(data=df_o, q=1, grid=T_GRID, local_config=lc,
                               bootstrap_config=bc, components=[0],
                               bandwidth_inf=h_inf)
    mhat_o = scb.beta_hat[:, 0]
    lo_o = scb.lower[:, 0]; hi_o = scb.upper[:, 0]
    qcrit_o = float(scb.qcrit)
    wid_o = float(np.nanmean(hi_o - lo_o))

    # Cai-Hu B-spline
    grid_f, mhat_f, lo_f, hi_f, meta = caihu_scb(
        t, Y, sid, n, alpha=ALPHA, B=2000, grid_size=2001, degree=3, seed=2026,
    )
    mhat_fg = interp1d(grid_f, mhat_f, kind="linear")(T_GRID)
    lo_fg = interp1d(grid_f, lo_f, kind="linear")(T_GRID)
    hi_fg = interp1d(grid_f, hi_f, kind="linear")(T_GRID)
    wid_f = float(np.mean(hi_fg - lo_fg))
    print(f"Ours qcrit={qcrit_o:.3f} wid={wid_o:.4f} | "
          f"Cai-Hu J_n={meta['Jn']} Q={meta['Q']:.3f} wid={wid_f:.4f}")

    summary = {
        "setting": {"n_patients": n, "n_obs": int(len(df)), "m_bar": m_bar,
                    "cap_days": CAP_DAYS, "alpha": ALPHA, "c_inf": C_INF,
                    "h_est": h_est, "h_inf": h_inf, "n_boot": N_BOOT},
        "ours": {"qcrit": qcrit_o, "mean_band_width": wid_o,
                 "mhat_at_t01": float(mhat_o[0]),
                 "mhat_at_t05": float(mhat_o[T_GRID.size // 2]),
                 "mhat_at_t09": float(mhat_o[-1])},
        "caihu_bspline": {"Jn": int(meta["Jn"]), "Q": float(meta["Q"]),
                          "mean_band_width": wid_f,
                          "mhat_at_t01": float(mhat_fg[0]),
                          "mhat_at_t05": float(mhat_fg[T_GRID.size // 2]),
                          "mhat_at_t09": float(mhat_fg[-1])},
        "width_ratio_ours_over_caihu": wid_o / wid_f,
    }
    band_rows = []
    for j in range(T_GRID.size):
        band_rows.append({"t": float(T_GRID[j]),
                          "day": float(T_GRID[j] * CAP_DAYS),
                          "ours_mhat": float(mhat_o[j]),
                          "ours_lower": float(lo_o[j]),
                          "ours_upper": float(hi_o[j]),
                          "caihu_mhat": float(mhat_fg[j]),
                          "caihu_lower": float(lo_fg[j]),
                          "caihu_upper": float(hi_fg[j]),
                          "ours_mhat_bili": float(np.exp(mhat_o[j])),
                          "caihu_mhat_bili": float(np.exp(mhat_fg[j]))})
    with open(OUT_DIR / "pbc_mean_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(band_rows).to_csv(OUT_DIR / "pbc_mean_band_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.plot(T_GRID, mhat_o, "C0-", lw=2, label="Ours estimate")
    ax.fill_between(T_GRID, lo_o, hi_o, color="C0", alpha=0.22,
                    label="Ours 95% SCB")
    ax.plot(T_GRID, mhat_fg, "C2--", lw=1.6, label="Cai-Hu estimate")
    ax.plot(T_GRID, lo_fg, "C2:", lw=1.0)
    ax.plot(T_GRID, hi_fg, "C2:", lw=1.0, label="Cai-Hu 95% SCB")
    ax.set_xlabel("t = day / 3000"); ax.set_ylabel("log(bilirubin)")
    ax.set_title("log scale"); ax.legend(fontsize=9, loc="upper left")
    ax = axes[1]
    ax.plot(T_GRID, np.exp(mhat_o), "C0-", lw=2)
    ax.fill_between(T_GRID, np.exp(lo_o), np.exp(hi_o), color="C0", alpha=0.22)
    ax.plot(T_GRID, np.exp(mhat_fg), "C2--", lw=1.6)
    ax.plot(T_GRID, np.exp(lo_fg), "C2:", lw=1.0)
    ax.plot(T_GRID, np.exp(hi_fg), "C2:", lw=1.0)
    ax.axhline(1.0, color="gray", lw=0.8, ls=":", alpha=0.6,
               label="normal upper (1 mg/dL)")
    ax.axhline(2.0, color="orange", lw=0.8, ls=":", alpha=0.6,
               label="mild (2 mg/dL)")
    ax.axhline(5.0, color="red", lw=0.8, ls=":", alpha=0.6,
               label="severe (5 mg/dL)")
    ax.set_xlabel("t = day / 3000"); ax.set_ylabel("bilirubin (mg/dL)")
    ax.set_title("clinical scale (back-transformed)")
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"PBC mean log-bilirubin (n={n}): Ours vs Cai-Hu B-spline")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / "pbc_mean_bands.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
