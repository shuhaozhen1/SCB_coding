"""PBC data illustration figure + descriptive summary.

Panels:
  A  spaghetti plot of individual log-bilirubin trajectories (sparse design)
  B  histogram of observations per patient (sparsity of the design)
  C  raw vs log bilirubin distribution (motivates the log transform)

Outputs (results/realdata/):
  pbc_data_illustration.png
  pbc_data_summary.json
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

DATA_PATH = ROOT / "data" / "pbcseq.csv"
OUT_DIR = ROOT / "results" / "realdata"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CAP_DAYS = 3000.0


def main():
    raw = pd.read_csv(DATA_PATH)
    df = raw[["id", "day", "trt", "age", "sex", "bili"]].dropna(subset=["bili"]).copy()
    df = df[df["day"] <= CAP_DAYS].copy()
    df["t"] = df["day"] / CAP_DAYS
    df["logbili"] = np.log(df["bili"].clip(lower=1e-3))
    n = df["id"].nunique()
    counts = df.groupby("id").size()

    summary = {
        "n_patients": int(n),
        "n_obs": int(len(df)),
        "obs_per_patient": {"median": float(counts.median()),
                            "min": int(counts.min()), "max": int(counts.max()),
                            "mean": float(counts.mean())},
        "followup_years_cap": CAP_DAYS / 365.25,
        "bili_raw": {"median": float(df["bili"].median()),
                     "mean": float(df["bili"].mean()),
                     "skewness": float(df["bili"].skew()),
                     "pct_above_1": float((df["bili"] > 1.0).mean()),
                     "pct_above_2": float((df["bili"] > 2.0).mean())},
        "treatment_split": {str(k): int(v) for k, v in
                            raw.dropna(subset=["trt"]).groupby("id")["trt"].first().value_counts().items()},
        "age_at_entry": {"median": float(raw.groupby("id")["age"].first().median()),
                         "min": float(raw.groupby("id")["age"].first().min()),
                         "max": float(raw.groupby("id")["age"].first().max())},
    }
    (OUT_DIR / "pbc_data_summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # Panel A: spaghetti of log-bilirubin
    ax = axes[0]
    rng = np.random.default_rng(2026)
    ids = df["id"].unique()
    show = rng.choice(ids, size=min(60, len(ids)), replace=False)
    for pid in show:
        g = df[df["id"] == pid].sort_values("t")
        ax.plot(g["t"], g["logbili"], "-", color="0.6", lw=0.5, alpha=0.6)
        ax.plot(g["t"], g["logbili"], ".", color="0.35", ms=2.5)
    # population mean (binned)
    bins = np.linspace(0, 1, 21); mids = 0.5 * (bins[:-1] + bins[1:])
    bidx = np.digitize(df["t"], bins) - 1
    mean_b = [df["logbili"][bidx == k].mean() for k in range(len(mids))]
    ax.plot(mids, mean_b, "C3-", lw=2.2, label="binned mean")
    ax.axhline(0.0, color="C0", ls=":", lw=1, alpha=0.7, label="log(1)=normal")
    ax.set_xlabel("t = day / 3000"); ax.set_ylabel("log(bilirubin)")
    ax.set_title(f"A. Individual log-bilirubin trajectories\n(60 of {n} patients)")
    ax.legend(fontsize=8, loc="upper left")

    # Panel B: observations per patient
    ax = axes[1]
    ax.hist(counts.values, bins=np.arange(0.5, counts.max() + 1.5, 1.0),
            color="C0", edgecolor="white")
    ax.axvline(counts.median(), color="C3", lw=2,
               label=f"median = {int(counts.median())}")
    ax.set_xlabel("observations per patient"); ax.set_ylabel("number of patients")
    ax.set_title("B. Sparse longitudinal design")
    ax.legend(fontsize=9)

    # Panel C: raw vs log bili
    ax = axes[2]
    ax.hist(df["bili"].values, bins=40, color="0.7", edgecolor="white",
            label=f"raw bilirubin (skew {df['bili'].skew():.1f})")
    ax.set_xlabel("bilirubin (mg/dL)"); ax.set_ylabel("observations")
    ax.set_title("C. Right-skewed marker -> log scale")
    ax2 = ax.twiny()
    ax2.hist(df["logbili"].values, bins=40, histtype="step", color="C0", lw=1.6,
             label="log(bilirubin)")
    ax2.set_xlabel("log(bilirubin)", color="C0"); ax2.tick_params(axis="x", colors="C0")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    fig.suptitle(f"Mayo PBC sequential bilirubin: n={n} patients, {len(df)} observations, "
                 f"median {int(counts.median())} obs/patient", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "pbc_data_illustration.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_DIR / 'pbc_data_illustration.png'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
