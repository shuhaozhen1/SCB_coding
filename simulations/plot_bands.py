"""Illustration figures: representative SCB per setting, Ours vs comparator.

Reads results/plotdata/{part1,part2,part3}.npz (one fixed replicate per cell)
and writes panel grids to results/figures/.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PD = ROOT / "results" / "plotdata"
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

OURS_C = "#1f77b4"; CAO_C = "#d62728"; CH_C = "#ff7f0e"; TRUTH_C = "black"


def _panel(ax, t, truth, o_lo, o_hi, o_est, c_lo, c_hi, comp_name, comp_color, title):
    ax.fill_between(t, o_lo, o_hi, color=OURS_C, alpha=0.20, lw=0)
    ax.plot(t, o_lo, color=OURS_C, lw=0.8); ax.plot(t, o_hi, color=OURS_C, lw=0.8)
    ax.plot(t, o_est, color=OURS_C, lw=1.4, label="Ours est.")
    ax.plot(t, c_lo, color=comp_color, lw=0.9, ls="--")
    ax.plot(t, c_hi, color=comp_color, lw=0.9, ls="--", label=f"{comp_name} SCB")
    ax.plot(t, truth, color=TRUTH_C, lw=1.3, ls=":", label="truth")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def plot_part1():
    d = dict(np.load(PD / "part1.npz"))
    t = d["t"]; truth = d["truth"]
    for n in (200, 300):
        fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
        for r, m in enumerate((5, 10, 15)):
            for c, ns in enumerate(("gauss", "uniform", "laplace")):
                k = f"n{n}_m{m}_{ns}"; ax = axes[r, c]
                _panel(ax, t, truth, d[f"{k}__ours_lo"], d[f"{k}__ours_hi"], d[f"{k}__ours_est"],
                       d[f"{k}__cao_lo"], d[f"{k}__cao_hi"], "Cao 2018", CAO_C,
                       f"m={m}, {ns}")
                if r == 0 and c == 0: ax.legend(fontsize=7, loc="upper left")
        fig.suptitle(f"Part 1 (q=2 VCM): 95% SCB for $\\beta_1(t)$, n={n}", fontsize=12)
        fig.supxlabel("t"); fig.supylabel("$\\beta_1(t)$")
        fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
        fp = FIG / f"part1_bands_n{n}.png"; fig.savefig(fp, dpi=140); plt.close(fig)
        print(f"wrote {fp}")


def plot_part2():
    d = dict(np.load(PD / "part2.npz"))
    t = d["t"]; P = 5
    fig, axes = plt.subplots(P, 3, figsize=(12, 14), sharex=True)
    for k in range(P):
        for c, m in enumerate((5, 10, 15)):
            key = f"m{m}_k{k}"; ax = axes[k, c]
            ax.fill_between(t, d[f"{key}__ours_lo"], d[f"{key}__ours_hi"], color=OURS_C, alpha=0.20, lw=0)
            ax.plot(t, d[f"{key}__ours_lo"], color=OURS_C, lw=0.8)
            ax.plot(t, d[f"{key}__ours_hi"], color=OURS_C, lw=0.8)
            ax.plot(t, d[f"{key}__ours_est"], color=OURS_C, lw=1.3, label="Ours joint")
            ax.plot(t, d[f"{key}__bonf_lo"], color=CAO_C, lw=0.9, ls="--")
            ax.plot(t, d[f"{key}__bonf_hi"], color=CAO_C, lw=0.9, ls="--", label="Bonferroni-Cao")
            ax.plot(t, d[f"truth_{k}"], color=TRUTH_C, lw=1.2, ls=":", label="truth")
            ax.set_title(f"$\\beta_{k}$, m={m}", fontsize=9); ax.tick_params(labelsize=7)
            if k == 0 and c == 0: ax.legend(fontsize=7, loc="best")
    fig.suptitle("Part 2 (q=5 VCM): joint 95% SCB vs Bonferroni-Cao, n=200", fontsize=12)
    fig.supxlabel("t"); fig.tight_layout(rect=(0.01, 0.01, 1, 0.98))
    fp = FIG / "part2_bands_joint.png"; fig.savefig(fp, dpi=140); plt.close(fig)
    print(f"wrote {fp}")


def plot_part3():
    d = dict(np.load(PD / "part3.npz"))
    t = d["t"]; truth = d["truth"]
    for n in (200, 300):
        fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
        for r, m in enumerate((5, 10, 15)):
            for c, dist in enumerate(("normal", "uniform", "laplace")):
                k = f"n{n}_m{m}_{dist}"; ax = axes[r, c]
                _panel(ax, t, truth, d[f"{k}__ours_lo"], d[f"{k}__ours_hi"], d[f"{k}__ours_est"],
                       d[f"{k}__ch_lo"], d[f"{k}__ch_hi"], "Cai-Hu 2025", CH_C,
                       f"m={m}, {dist}")
                if r == 0 and c == 0: ax.legend(fontsize=7, loc="upper left")
        fig.suptitle(f"Part 3 (q=1 mean function): 95% SCB, n={n}", fontsize=12)
        fig.supxlabel("t"); fig.supylabel("$m(t)$")
        fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
        fp = FIG / f"part3_bands_n{n}.png"; fig.savefig(fp, dpi=140); plt.close(fig)
        print(f"wrote {fp}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    if 1 in args.parts and (PD / "part1.npz").exists(): plot_part1()
    if 2 in args.parts and (PD / "part2.npz").exists(): plot_part2()
    if 3 in args.parts and (PD / "part3.npz").exists(): plot_part3()
    print("plotting done")
