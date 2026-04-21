"""Emit Table II (LaTeX, booktabs) from per-mode part1_summary.csv files.

Generates:
    results/table2a_synthetic.tex
    results/table2b_real.tex
"""

from pathlib import Path

import numpy as np

from part1.run_part1 import RESULTS


def _load_rows(path):
    rows = []
    with open(path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            rows.append(dict(zip(header, line.strip().split(","))))
    return rows


CAPTIONS = {
    "synthetic": (r"\textbf{Table IIa — Synthetic positive control.} "
                  r"Measurements are $y_t = \mathrm{gt}_t + \mathcal{N}(0, R_t)$ "
                  r"with $R_t$ matching the filter's internal $R$. Pooled "
                  r"across 20 episodes of 150 steps."),
    "real":      (r"\textbf{Table IIb — Real (HSV) detector.} "
                  r"Measurements come from the OpenCV pipeline on rendered "
                  r"MuJoCo frames with per-frame pixel corruption. "
                  r"$R$ scales with injected pixel noise on $(x,y)$; "
                  r"a 1\,px floor prevents a singular gain at $\sigma{=}0$."),
}

LABEL = {"synthetic": "tab:part1_synth", "real": "tab:part1_real"}


def emit_table(mode):
    csv_path = RESULTS / mode / "part1_summary.csv"
    rows = _load_rows(csv_path)
    sigmas = sorted({int(r["sigma"]) for r in rows})
    display = {"raw": "raw",
               "ekf_cart": "EKF-Cartesian",
               "riekf": "RI-EKF"}
    order = ["raw", "ekf_cart", "riekf"]

    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{" + CAPTIONS[mode] + r" Position RMSE in cm, "
             r"orientation RMSE in degrees; $\bar{\mathrm{NIS}}$ mean, "
             r"$f_{\mathrm{CI}}$ = fraction of pooled NIS inside "
             r"$[\chi^2_{3,.025}, \chi^2_{3,.975}]$, $p_{\mathrm{KS}}$ = "
             r"K--S $p$-value against $\chi^2(3)$.}",
             r"\label{" + LABEL[mode] + r"}",
             r"\begin{tabular}{ll" + "ccccc" + "}",
             r"\toprule",
             r"$\sigma$ (px) & Filter & pos (cm) & ori (deg) & "
             r"$\bar{\mathrm{NIS}}$ & $f_{\mathrm{CI}}$ & $p_{\mathrm{KS}}$ \\",
             r"\midrule"]

    def fmt_stat(val, digits):
        if val == "" or val == "nan":
            return "--"
        try:
            v = float(val)
            if np.isnan(v):
                return "--"
            return f"{v:.{digits}f}"
        except ValueError:
            return "--"

    for s in sigmas:
        for flt in order:
            r = next(r for r in rows
                     if int(r["sigma"]) == s and r["filter"] == flt)
            pos_cm = float(r["pos_rmse"]) * 100
            ori_deg = np.degrees(float(r["ori_rmse"]))
            cells = [str(s) if flt == "raw" else "",
                     display[flt],
                     f"{pos_cm:.3f}",
                     f"{ori_deg:.3f}",
                     fmt_stat(r["mean_nis"], 3),
                     fmt_stat(r["frac_in_CI"], 3),
                     fmt_stat(r["ks_pvalue"], 3)]
            lines.append(" & ".join(cells) + r" \\")
        if s != sigmas[-1]:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    suffix = "a_synthetic" if mode == "synthetic" else "b_real"
    out = RESULTS / f"table2{suffix}.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"[table] wrote {out}")


def main():
    emit_table("synthetic")
    emit_table("real")


if __name__ == "__main__":
    main()
