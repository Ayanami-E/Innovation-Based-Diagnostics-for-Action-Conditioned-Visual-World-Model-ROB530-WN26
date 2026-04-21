"""§V.B analyses: correlation matrix, discriminative power, metric scatter.

Inputs: `results/part2/summary_5b.csv`, `results/part2/summary_5a.csv`
       (the latter is re-summarized from per-ep NIS for consistency).

Outputs:
  - `results/part2/corr_matrix.csv` (Pearson + Spearman)
  - `results/part2/discriminative.csv` (AUC, Cohen's d per metric)
  - `results/part2/figures/fig_corr_heatmap.pdf`
  - `results/part2/figures/fig_discriminative.pdf`
  - `results/part2/figures/fig_scatter.pdf`

AUC and Mann-Whitney U are rolled by hand (no sklearn in venv).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "part2"
FIG = RESULTS / "figures"

METRIC_COLS = [
    "nis_mean", "nis_ks_pvalue",
    "sod_mean", "sod_ks_pvalue",
    "mean_phi", "anomaly_rate",
    "mean_mse", "mean_ncc",
    "pos_drift_vs_gt", "ori_drift_vs_gt",
]

# Nicer labels for axes/columns.
LABEL = {
    "nis_mean":        "NIS (mean)",
    "nis_ks_pvalue":   "NIS KS p",
    "sod_mean":        "SOD-chi2 (mean)",
    "sod_ks_pvalue":   "SOD-chi2 KS p",
    "mean_phi":        "Flow cons phi",
    "anomaly_rate":    "Anomaly rate",
    "mean_mse":        "Frame MSE",
    "mean_ncc":        "Frame NCC",
    "pos_drift_vs_gt": "Pos drift (px)",
    "ori_drift_vs_gt": "Ori drift (rad)",
}


def _load_5b() -> list[dict]:
    with (RESULTS / "summary_5b.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_array(rows, key):
    return np.array([float(r[key]) for r in rows], dtype=float)


def _auc_roc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC where positive class ranks *higher*. Ties = 0.5."""
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    count = 0.0
    for p in pos:
        count += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return count / (pos.size * neg.size)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    sa = a.std(ddof=1); sb = b.std(ddof=1)
    pooled = np.sqrt(((a.size - 1) * sa**2 + (b.size - 1) * sb**2) /
                     (a.size + b.size - 2))
    if pooled < 1e-12:
        return float("nan" if abs(a.mean() - b.mean()) < 1e-12 else np.sign(
            a.mean() - b.mean()) * np.inf)
    return float((a.mean() - b.mean()) / pooled)


# ---- Analysis A: correlation matrix ----------------------------------------

def analysis_correlation(rows: list[dict]):
    """Pool all 10 (source x episode) rows; compute Pearson + Spearman."""
    M = np.column_stack([_as_array(rows, c) for c in METRIC_COLS])
    # Many metrics have log-heavy tails (NIS, MSE); use log1p before Pearson
    # where sensible. Spearman is rank-based so it doesn't need this.
    # We report Pearson on the raw values for transparency.
    n = len(METRIC_COLS)
    P = np.full((n, n), np.nan)
    S = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            xi, xj = M[:, i], M[:, j]
            valid = (~np.isnan(xi)) & (~np.isnan(xj))
            if valid.sum() < 3:
                continue
            if np.std(xi[valid]) < 1e-12 or np.std(xj[valid]) < 1e-12:
                continue
            P[i, j] = float(np.corrcoef(xi[valid], xj[valid])[0, 1])
            S[i, j] = float(sstats.spearmanr(xi[valid], xj[valid]).statistic)

    # Dump CSV (long format: i, j, pearson, spearman)
    csv_path = RESULTS / "corr_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric_i", "metric_j", "pearson", "spearman"])
        for i in range(n):
            for j in range(n):
                w.writerow([METRIC_COLS[i], METRIC_COLS[j],
                            f"{P[i, j]:.4f}" if not np.isnan(P[i, j]) else "",
                            f"{S[i, j]:.4f}" if not np.isnan(S[i, j]) else ""])
    print(f"Wrote {csv_path}")

    # Heatmap figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, mat, title in [(axes[0], P, "Pearson"), (axes[1], S, "Spearman")]:
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(n))
        ax.set_xticklabels([LABEL[c] for c in METRIC_COLS], rotation=45,
                           ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels([LABEL[c] for c in METRIC_COLS])
        ax.set_title(f"{title} (n={M.shape[0]})")
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(v) > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.045)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig_path = FIG / "fig_corr_heatmap.pdf"
    fig.savefig(fig_path); fig.savefig(fig_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Wrote {fig_path}")

    return P, S


# ---- Analysis B: discriminative power --------------------------------------

TRIVIAL_BY_CONSTRUCTION = {"mean_mse", "mean_ncc",
                           "pos_drift_vs_gt", "ori_drift_vs_gt"}


def analysis_discriminative(rows: list[dict]):
    gt = [r for r in rows if r["source"] == "gt"]
    ro = [r for r in rows if r["source"] == "rollout"]

    out = []
    for col in METRIC_COLS:
        pos = _as_array(ro, col)   # rollout = positive class
        neg = _as_array(gt, col)
        # Some metrics (ks p, NCC, anomaly_rate) move in the direction where
        # rollout is LOWER. Auto-detect via means and invert if so.
        direction = 1.0 if pos.mean() >= neg.mean() else -1.0
        auc = _auc_roc(direction * pos, direction * neg)
        d = direction * _cohens_d(pos, neg)
        out.append(dict(
            metric=col, label=LABEL[col],
            direction=int(direction),
            gt_mean=float(neg.mean()), gt_std=float(neg.std(ddof=1)),
            rollout_mean=float(pos.mean()), rollout_std=float(pos.std(ddof=1)),
            cohens_d=float(d), auc_roc=float(auc),
            trivial=int(col in TRIVIAL_BY_CONSTRUCTION),
        ))

    csv_path = RESULTS / "discriminative.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"Wrote {csv_path}")
    for r in out:
        tag = " (trivial)" if r["trivial"] else ""
        print(f"  {r['label']:18s}: gt={r['gt_mean']:8.3f}+/-{r['gt_std']:6.3f} "
              f"ro={r['rollout_mean']:8.3f}+/-{r['rollout_std']:6.3f} "
              f"d={r['cohens_d']:+6.2f} AUC={r['auc_roc']:.2f}{tag}")

    # Bar chart of |AUC - 0.5|*2 (i.e., how much info the metric carries).
    fig, ax = plt.subplots(figsize=(9, 4))
    metrics = [r["label"] for r in out]
    aucs = np.array([r["auc_roc"] for r in out])
    colors = ["#c0392b" if r["metric"] in ("nis_mean", "sod_mean")
              else ("#999" if r["trivial"] else "#3498db")
              for r in out]
    bars = ax.bar(range(len(metrics)), aucs, color=colors)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8,
               label="Chance")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=35, ha="right")
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("AUC-ROC (rollout positive; direction auto-flipped)")
    ax.set_title("GT vs open-loop rollout discrimination (n=5+5)")
    ax.legend(loc="lower right")
    for b, r in zip(bars, out):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{r['auc_roc']:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig_path = FIG / "fig_discriminative.pdf"
    fig.savefig(fig_path); fig.savefig(fig_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Wrote {fig_path}")

    return out


# ---- Analysis C: metric-metric scatter -------------------------------------

def analysis_scatter(rows: list[dict]):
    target = "nis_mean"
    other = [c for c in METRIC_COLS if c != target]
    ncols = 3
    nrows = int(np.ceil(len(other) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = axes.ravel()
    x_gt = _as_array([r for r in rows if r["source"] == "gt"], target)
    x_ro = _as_array([r for r in rows if r["source"] == "rollout"], target)
    for ax, col in zip(axes, other):
        y_gt = _as_array([r for r in rows if r["source"] == "gt"], col)
        y_ro = _as_array([r for r in rows if r["source"] == "rollout"], col)
        ax.scatter(x_gt, y_gt, color="#3498db", label="gt", s=40)
        ax.scatter(x_ro, y_ro, color="#c0392b", label="rollout", s=40,
                   marker="^")
        for x, y, r in zip(np.r_[x_gt, x_ro], np.r_[y_gt, y_ro],
                           rows):
            pass  # labels would clutter; skip
        ax.set_xlabel(LABEL[target]); ax.set_ylabel(LABEL[col])
        ax.grid(alpha=0.3)
    for ax in axes[len(other):]:
        ax.set_visible(False)
    axes[0].legend(loc="best")
    fig.suptitle(f"{LABEL[target]} vs other metrics, per episode", y=1.00)
    fig.tight_layout()
    fig_path = FIG / "fig_scatter.pdf"
    fig.savefig(fig_path); fig.savefig(fig_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Wrote {fig_path}")


def main():
    rows = _load_5b()
    print(f"Loaded {len(rows)} rows from summary_5b.csv")
    FIG.mkdir(parents=True, exist_ok=True)
    analysis_correlation(rows)
    analysis_discriminative(rows)
    analysis_scatter(rows)


if __name__ == "__main__":
    main()
