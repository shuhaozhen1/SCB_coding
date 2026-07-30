"""Consolidate all simulation results into a documented Excel workbook.

Sheets:
  Notes       - settings, the undersmoothing rule, column glossary
  Part1       - q=2 VCM marginal SCB (Ours vs Cao 2018), 18 cells
  Part2       - q=5 joint SCB (Ours vs Bonferroni-Cao), 18 cells
  Part3       - q=1 mean SCB (Ours vs Cai-Hu 2025), 18 cells
  Aggregates  - per-part summary metrics

Output: results/simulation_results.xlsx
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "simulation_results.xlsx"


def _mcse(p, R):
    return float(np.sqrt(p * (1 - p) / R))


def load_part1():
    df = pd.read_csv(RES / "part1" / "part1_summary.csv")
    out = pd.DataFrame({
        "n": df["n"], "m": df["m"], "noise": df["noise"],
        "ours_coverage": df["ours_coverage"].round(3),
        "ours_cov_mcse": df["ours_cov_mcse"].round(4),
        "ours_width": df["ours_width"].round(3),
        "cao_coverage": df["cao_coverage"].round(3),
        "cao_width": df["cao_width"].round(3),
        "width_ratio_ours_over_cao": (df["ours_width"] / df["cao_width"]).round(3),
        "h_est_mean": df["h_hat_mean"].round(4),
        "n_reps": df["n_reps"],
    })
    return out.sort_values(["n", "m", "noise"]).reset_index(drop=True)


def load_part3():
    df = pd.read_csv(RES / "part3" / "part3_summary.csv")
    out = pd.DataFrame({
        "n": df["n"], "m": df["m"], "dist": df["dist"],
        "ours_coverage": df["ours_coverage"].round(3),
        "ours_cov_mcse": df["ours_cov_mcse"].round(4),
        "ours_width": df["ours_width"].round(3),
        "caihu_coverage": df["caihu_coverage"].round(3),
        "caihu_width": df["caihu_width"].round(3),
        "width_ratio_ours_over_caihu": (df["ours_width"] / df["caihu_width"]).round(3),
        "h_est_mean": df["h_hat_mean"].round(4),
        "n_reps": df["n_reps"],
    })
    return out.sort_values(["n", "m", "dist"]).reset_index(drop=True)


def load_part2():
    ours = pd.read_csv(RES / "part2" / "part2_summary.csv")
    if "noise" not in ours.columns:
        ours["noise"] = "gauss"
    bf_path = RES / "part2_bonferroni" / "part2_bonferroni_summary.csv"
    bf = pd.read_csv(bf_path) if bf_path.exists() else None
    if bf is not None and "noise" not in bf.columns:
        bf["noise"] = "gauss"
    rows = []
    for _, o in ours.sort_values(["n", "m", "noise"]).iterrows():
        row = {"n": int(o["n"]), "m": int(o["m"]), "noise": str(o["noise"]),
               "ours_joint_coverage": round(float(o["joint_coverage"]), 3),
               "ours_cov_mcse": round(float(o["joint_cov_mcse"]), 4),
               "ours_width": round(float(o["width_avg"]), 3),
               "h_est_mean": round(float(o["h_hat_mean"]), 4),
               "n_reps": int(o["n_reps"])}
        if bf is not None:
            match = bf[(bf["n"] == o["n"]) &
                       (bf["m"] == o["m"]) &
                       (bf["noise"] == o["noise"])]
            if not match.empty:
                b = match.iloc[0]
                row["bonferroni_cao_coverage"] = round(float(b["joint_coverage"]), 3)
                row["bonferroni_cao_width"] = round(float(b["width_avg"]), 3)
                row["width_ratio_ours_over_bonf"] = round(
                    float(o["width_avg"]) / float(b["width_avg"]), 3)
            else:
                row["bonferroni_cao_coverage"] = np.nan
                row["bonferroni_cao_width"] = np.nan
                row["width_ratio_ours_over_bonf"] = np.nan
        else:
            row["bonferroni_cao_coverage"] = np.nan
            row["bonferroni_cao_width"] = np.nan
            row["width_ratio_ours_over_bonf"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n", "m", "noise"]).reset_index(drop=True)


def aggregates(p1, p2, p3):
    rows = []
    def row(part, method, cov, wid):
        return {"part": part, "method": method,
                "mean_cov": round(cov.mean(), 3),
                "mean_abs_dev_0.95": round(float(np.mean(np.abs(cov - 0.95))), 3),
                "min_cov": round(cov.min(), 3), "max_cov": round(cov.max(), 3),
                "mean_width": round(wid.mean(), 3)}
    rows.append(row("Part 1 (q=2)", "Ours", p1["ours_coverage"], p1["ours_width"]))
    rows.append(row("Part 1 (q=2)", "Cao 2018", p1["cao_coverage"], p1["cao_width"]))
    rows.append(row("Part 2 (q=5 joint)", "Ours", p2["ours_joint_coverage"], p2["ours_width"]))
    if p2["bonferroni_cao_coverage"].notna().all():
        rows.append(row("Part 2 (q=5 joint)", "Bonferroni-Cao", p2["bonferroni_cao_coverage"], p2["bonferroni_cao_width"]))
    rows.append(row("Part 3 (q=1)", "Ours", p3["ours_coverage"], p3["ours_width"]))
    rows.append(row("Part 3 (q=1)", "Cai-Hu 2025", p3["caihu_coverage"], p3["caihu_width"]))
    return pd.DataFrame(rows)


def notes_sheet():
    lines = [
        ("Simultaneous Confidence Bands - Simulation Results", ""),
        ("", ""),
        ("Generated", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")),
        ("Replicates per cell (R)", "1000"),
        ("Nominal level (1 - alpha)", "0.95"),
        ("Evaluation grid", "96 points, linspace(0.025, 0.975)"),
        ("Bootstrap (Ours)", "300 Gaussian multipliers, leave-one-subject-out residuals"),
        ("Estimation bandwidth h_est", "leave-one-subject-out CV on linspace(0.04,0.16,13)"),
        ("Inference bandwidth h_inf", "h_est * (1 - 8 / ( n^(1/2) * (m_bar/q)^(1/4) ))"),
        ("   q (model dimension)", "Part 1: q=2 ; Part 2: q=5 ; Part 3: q=1"),
        ("   n", "number of subjects ; m_bar = mean observations per subject"),
        ("", ""),
        ("Comparators", ""),
        ("   Part 1", "Cao et al. (2018) plug-in SCB (Gumbel critical value)"),
        ("   Part 2", "Bonferroni-Cao: per-coefficient Cao SCB at level alpha/q=0.01,"),
        ("           ", "                intersected -> joint coverage >= 1-alpha (conservative)"),
        ("   Part 3", "Cai & Hu (2025) B-spline sparse-to-dense SCB"),
        ("", ""),
        ("Column glossary", ""),
        ("   *_coverage", "empirical simultaneous coverage (fraction of R replicates"),
        ("             ", "whose band contains the target at every grid point)"),
        ("   *_cov_mcse", "Monte-Carlo standard error of the coverage, sqrt(p(1-p)/R)"),
        ("   *_width", "average band width (over grid; for joint/Part2 averaged over coefficients)"),
        ("   width_ratio_ours_over_*", "Ours width divided by comparator width (<1 => Ours narrower)"),
        ("   h_est_mean", "mean LOSO-CV estimation bandwidth across replicates"),
        ("", ""),
        ("Notes", "Part 2 is a genuinely joint band over all 5 coefficients; the"),
        ("     ", "Bonferroni-Cao comparator attains joint coverage only by inflating"),
        ("     ", "each per-coefficient band to the alpha/q level, hence its larger width."),
        ("     ", "Cao (2018) is a sparse-design method; its over-coverage at large m is"),
        ("     ", "expected and reported faithfully (no tuning applied to the comparator)."),
    ]
    return pd.DataFrame(lines, columns=["item", "value"])


def main():
    p1 = load_part1(); p2 = load_part2(); p3 = load_part3()
    notes = notes_sheet(); agg = aggregates(p1, p2, p3)
    with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
        notes.to_excel(xl, sheet_name="Notes", index=False)
        agg.to_excel(xl, sheet_name="Aggregates", index=False)
        p1.to_excel(xl, sheet_name="Part1_VCM_q2", index=False)
        p2.to_excel(xl, sheet_name="Part2_joint_q5", index=False)
        p3.to_excel(xl, sheet_name="Part3_mean_q1", index=False)
        # widen columns
        for sh in xl.sheets.values():
            for col in sh.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                sh.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 60)
    print(f"wrote {OUT}")
    print("\nAggregates:")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
