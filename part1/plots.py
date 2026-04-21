"""Figures for Part 1 report.

Per mode (real / synthetic) in its own subdir:
    fig2_nis.{pdf,png}            NIS vs time, one ep at sigma=10
    fig3_rmse_vs_sigma.{pdf,png}  RMSE vs sigma, filters vs raw

Cross-mode (top-level results/):
    fig2b_nis_histograms.{pdf,png}  NIS histograms vs chi^2(3) pdf.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2

from part1.scene import DT
from part1.perception import detect_tblock
from part1.corruption import apply_corruption
from part1.ekf_se2_cartesian import EKF_SE2_Cartesian
from part1.riekf_se2 import RIEKF_SE2
from part1.se2 import pose_to_SE2
from part1.run_part1 import (
    RESULTS, CACHE, Q_EKF, Q_XI, Q_V, P0_EKF, P0_RI,
    compute_R_real, compute_R_synthetic,
    synthetic_episode,
    BASE_SEED, FOV_HALF,
)


def _load_csv(path):
    rows = []
    with open(path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            vals = line.strip().split(",")
            rows.append(dict(zip(header, vals)))
    return rows


def _run_one_episode(ep_idx, sigma, obs_source):
    cache = CACHE / f"episode_{ep_idx:02d}.npz"
    d = np.load(cache)
    gt = d["gt_poses"]
    frames = d["frames"]
    T = len(frames)

    if obs_source == "real":
        R = compute_R_real(sigma)
        rng = np.random.default_rng(BASE_SEED + ep_idx + 1000 * (sigma + 1))
        dets = np.zeros((T, 3))
        valid = np.zeros(T, dtype=bool)
        prev = None
        for t, f in enumerate(frames):
            img = apply_corruption(f, sigma_pixel=sigma, rng=rng) \
                if sigma > 0 else f
            p = detect_tblock(img, prev_theta=prev, fov_half_width=FOV_HALF)
            if p is None:
                dets[t] = dets[t - 1] if t > 0 else gt[0]
                valid[t] = False
            else:
                dets[t] = p
                valid[t] = True
                prev = p[2]
    else:
        R = compute_R_synthetic(sigma)
        sim_seed = BASE_SEED + ep_idx + 1000 * (sigma + 1)
        dets, valid = synthetic_episode(gt, sigma, R, sim_seed)

    init = next((dets[i] for i in range(T) if valid[i]), gt[0])
    x0 = np.array([init[0], init[1], init[2], 0, 0, 0])
    ekf = EKF_SE2_Cartesian(DT, Q_EKF, R, x0, P0_EKF)
    ri = RIEKF_SE2(DT, Q_XI, Q_V, R, pose_to_SE2(*init),
                   np.zeros(3), P0_RI)
    for t in range(1, T):
        ekf.predict(); ri.predict()
        if valid[t]:
            ekf.update(dets[t]); ri.update(dets[t])
    return ekf.nis_log, ri.nis_log


def fig2_nis(mode, sigma=10, ep_idx=0):
    nis_ekf, nis_ri = _run_one_episode(ep_idx, sigma, mode)
    lo = chi2.ppf(0.025, 3)
    hi = chi2.ppf(0.975, 3)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, nis, title in zip(axes, [nis_ekf, nis_ri],
                              ["EKF-Cartesian", "RI-EKF"]):
        t = np.arange(len(nis))
        ax.plot(t, nis, lw=1.0, color="tab:blue")
        ax.axhline(lo, ls="--", color="tab:red", lw=1,
                   label=r"$\chi^2_{3,.025}$")
        ax.axhline(hi, ls="--", color="tab:red", lw=1,
                   label=r"$\chi^2_{3,.975}$")
        ax.axhline(3.0, ls=":", color="k", lw=1, label="nominal = 3")
        ax.set_title(f"{title}  (sigma={sigma}px, ep={ep_idx}, {mode})")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("NIS")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(f"Fig 2 — NIS over time ({mode})", fontweight="bold")
    fig.tight_layout()
    out = RESULTS / mode
    for ext in ("pdf", "png"):
        p = out / f"fig2_nis.{ext}"
        fig.savefig(p, dpi=150)
        print(f"  wrote {p}")
    plt.close(fig)


def fig3_rmse_vs_sigma(mode):
    out = RESULTS / mode
    rows = _load_csv(out / "part1_summary.csv")
    sigmas = sorted({int(r["sigma"]) for r in rows})
    filters = ["raw", "ekf_cart", "riekf"]
    styles = {"raw": ("tab:gray", "o", "raw"),
              "ekf_cart": ("tab:blue", "s", "EKF-Cartesian"),
              "riekf": ("tab:green", "^", "RI-EKF")}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for flt in filters:
        ys_pos, ys_ori = [], []
        for s in sigmas:
            r = next(r for r in rows
                     if int(r["sigma"]) == s and r["filter"] == flt)
            ys_pos.append(float(r["pos_rmse"]) * 100)
            ys_ori.append(np.degrees(float(r["ori_rmse"])))
        c, m, lbl = styles[flt]
        axes[0].plot(sigmas, ys_pos, color=c, marker=m, label=lbl)
        axes[1].plot(sigmas, ys_ori, color=c, marker=m, label=lbl)
    for ax, title, ylab in zip(axes,
                               ["Position RMSE", "Orientation RMSE"],
                               ["RMSE (cm)", "RMSE (deg)"]):
        ax.set_title(title)
        ax.set_xlabel(r"$\sigma$ (px)")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"Fig 3 — RMSE vs. corruption sigma ({mode})",
                 fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = out / f"fig3_rmse_vs_sigma.{ext}"
        fig.savefig(p, dpi=150)
        print(f"  wrote {p}")
    plt.close(fig)


def fig2b_nis_histograms():
    syn = np.load(RESULTS / "synthetic" / "part1_nis_pooled.npz")
    real = np.load(RESULTS / "real" / "part1_nis_pooled.npz")
    # Show EKF-Cartesian NIS (the canonical chi^2 test target). RI-EKF looks
    # similar at sigma>=5; showing one filter keeps the panel uncluttered.
    panels = [
        ("(a) synthetic, $\\sigma$=0 px",  syn["0_ekf_cart"]),
        ("(b) synthetic, $\\sigma$=10 px", syn["10_ekf_cart"]),
        ("(c) real (HSV), $\\sigma$=0 px", real["0_ekf_cart"]),
    ]

    x = np.linspace(0.01, 20, 400)
    chi2_pdf = chi2.pdf(x, 3)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)
    for ax, (label, data) in zip(axes, panels):
        # Clip to visible range; report what was clipped.
        clipped = (data > 20).sum()
        data_show = np.clip(data, 0, 20)
        ax.hist(data_show, bins=np.linspace(0, 20, 41), density=True,
                color="tab:blue", alpha=0.55, edgecolor="k",
                linewidth=0.3, label="empirical NIS")
        ax.plot(x, chi2_pdf, "r-", lw=1.8, label=r"$\chi^2(3)$ pdf")
        ax.axvline(chi2.ppf(0.025, 3), ls="--", color="gray", lw=0.8)
        ax.axvline(chi2.ppf(0.975, 3), ls="--", color="gray", lw=0.8)
        ax.set_xlim(0, 20)
        ax.set_title(label)
        ax.set_xlabel("NIS")
        ax.set_ylabel("density")
        note = (f"n={len(data)}, mean={data.mean():.2f}, "
                f"clip>20: {clipped}")
        ax.text(0.98, 0.98, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7",
                          alpha=0.85))
        ax.legend(loc="upper center", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fig 2b — NIS distributions vs. $\\chi^2(3)$ "
                 "(EKF-Cartesian; 20 eps × 150 steps)",
                 fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = RESULTS / f"fig2b_nis_histograms.{ext}"
        fig.savefig(p, dpi=150)
        print(f"  wrote {p}")
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    for mode in ("synthetic", "real"):
        (RESULTS / mode).mkdir(parents=True, exist_ok=True)
        print(f"[plots] {mode}  Fig 2 (NIS vs time)")
        fig2_nis(mode, sigma=10, ep_idx=0)
        print(f"[plots] {mode}  Fig 3 (RMSE vs sigma)")
        fig3_rmse_vs_sigma(mode)
    print("[plots] Fig 2b (NIS histograms)")
    fig2b_nis_histograms()


if __name__ == "__main__":
    main()
