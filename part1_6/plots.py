"""§V.D figures: blind-spot panel + P4 trajectory example.

Outputs:
  results/part1_6/fig_blind_spot.pdf, .png   (2x2 panel, NIS/SOD vs pos err)
  results/part1_6/fig_p4_example.pdf, .png   (one episode, GT vs P4 overlay)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from part1_6.perturb import apply_perturbation

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "part1_6"
CACHE = ROOT / "results" / "part1_cache"

PERTURBATIONS = ["P1", "P2", "P3", "P4"]
PERTURB_TITLE = {
    "P1": "P1 — constant-velocity drift",
    "P2": "P2 — constant angular bias",
    "P3": "P3 — constant-acceleration drift",
    "P4": "P4 — sine bump (smooth, bounded)",
}


def _load_summary() -> list[dict]:
    with (RESULTS / "summary_1_6.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _by_kind(rows: list[dict], kind: str) -> dict:
    sel = [r for r in rows if r["kind"] == kind]
    sel.sort(key=lambda r: float(r["magnitude"]))
    return dict(
        magnitude=np.array([float(r["magnitude"]) for r in sel]),
        mean_nis=np.array([float(r["mean_nis"]) for r in sel]),
        sod_mean=np.array([float(r["sod_mean"]) for r in sel]),
        flow_phi=np.array([float(r["flow_phi_mean"]) for r in sel]),
        pos_err_mean=np.array([float(r["pos_err_mean"]) for r in sel]),
        pos_err_max=np.array([float(r["pos_err_max"]) for r in sel]),
        ori_err_max=np.array([float(r["ori_err_max"]) for r in sel]),
    )


def fig_blind_spot(rows: list[dict]):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    chi2_mean = 3.0  # expected mean for chi^2(3) under the null

    for ax, kind in zip(axes.ravel(), PERTURBATIONS):
        d = _by_kind(rows, kind)
        # x-axis: mean *physical error* (use ori_err for P2, pos_err else)
        if kind == "P2":
            x = d["ori_err_max"]
            xlabel = "Max angular error (rad)"
        else:
            x = d["pos_err_mean"]
            xlabel = "Mean positional error (m)"

        nis = np.maximum(d["mean_nis"], 1e-3)  # log-axis floor
        sod = np.maximum(d["sod_mean"], 1e-3)

        ax.plot(x, nis, marker="o", color="#c0392b", label="NIS (mean)")
        ax.plot(x, sod, marker="s", color="#2980b9", label="SOD-chi2 (mean)")
        ax.axhline(chi2_mean, color="black", linestyle="--", linewidth=0.8,
                   label="chi^2(3) expected")
        ax.axhline(5.0, color="gray", linestyle=":", linewidth=0.8,
                   label="acceptance gate (5)")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Metric value (log scale)")
        ax.set_title(PERTURB_TITLE[kind])
        ax.grid(alpha=0.3, which="both")
        # annotate magnitude at each point
        for xi, nv, mg in zip(x, nis, d["magnitude"]):
            ax.annotate(f"m={mg:g}", (xi, nv), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Part 1.6 — kinematic-metric blind spot to "
                 "smooth-but-wrong perturbations")
    fig.tight_layout()
    out = RESULTS / "fig_blind_spot.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def fig_p4_example():
    """Trajectory overlay: clean GT vs P4 perturbed at m=0.05, episode 0."""
    cache = CACHE / "episode_00.npz"
    if not cache.exists():
        print(f"  [skip] {cache} not present")
        return
    gt = np.load(cache)["gt_poses"].astype(float)
    perturbed = apply_perturbation(gt, "P4", magnitude=0.05)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    t = np.arange(gt.shape[0])
    sc1 = ax.scatter(gt[:, 0], gt[:, 1], c=t, cmap="Blues",
                     s=22, label="GT", marker="o", edgecolors="none")
    sc2 = ax.scatter(perturbed[:, 0], perturbed[:, 1], c=t, cmap="Reds",
                     s=22, label="P4 perturbed (m=0.05)",
                     marker="^", edgecolors="none")
    ax.plot(gt[:, 0], gt[:, 1], color="#3498db", linewidth=0.7, alpha=0.5)
    ax.plot(perturbed[:, 0], perturbed[:, 1], color="#c0392b",
            linewidth=0.7, alpha=0.5)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.set_title("P4 sine-bump perturbation (episode 0): "
                 "smooth, bounded, kinematically plausible — "
                 "but physically wrong")
    fig.colorbar(sc2, ax=ax, label="step t (red = perturbed)")
    fig.colorbar(sc1, ax=ax, label="step t (blue = GT)")
    ax.legend(loc="best")
    fig.tight_layout()
    out = RESULTS / "fig_p4_example.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    rows = _load_summary()
    fig_blind_spot(rows)
    fig_p4_example()


if __name__ == "__main__":
    main()
