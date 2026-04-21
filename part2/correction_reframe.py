"""§V.C: reframe Part 3 correction methods as NIS validation.

For each cached correction method on episode 3 (`output/ep3_results/`):
  - Load both halves of the MP4 (rollout = pred half; GT is identical
    across methods since the source episode is the same).
  - Run the IWS HSV detector on both halves -> dense pose sequences.
  - Run EKF-Cartesian on the rollout-side detection sequence and pool NIS.
  - Run SOD-chi^2 with Sigma_a estimated from the GT-side pose pool.
  - Compute flow consistency, anomaly rate, MSE, NCC vs the GT half.

Outputs:
  - results/part2/summary_5c.csv
  - results/part2/figures/fig_correction_response.pdf
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from part1.ekf_se2_cartesian import EKF_SE2_Cartesian
from part1.nis_analysis import nis_stats

from part2 import iws_io, perception_iws
from part2.cross_metrics import (
    calibrate_anomaly_threshold,
    compute_anomaly_rate,
    compute_flow_consistency,
    compute_frame_mse,
    compute_ncc,
    compute_sod_chi2,
    estimate_sod_covariance,
)
from part2.run_iws_nis import (
    DT, P0_EKF_PX, Q_EKF_PX, R_PX,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "part2"
FIG = RESULTS / "figures"

METHODS = [
    "open-loop",
    "periodic_reset_k10",
    "periodic_reset_k20",
    "latent_smoother",
    "selective_correction",
    "best-of-3",
    "best-of-5",
]


def _run_ekf(raw_dets):
    initialized = False
    f = None
    n_drops = 0
    for det in raw_dets:
        if not initialized:
            if det is None:
                n_drops += 1; continue
            x0 = np.zeros(6); x0[:3] = det
            f = EKF_SE2_Cartesian(dt=DT, Q=Q_EKF_PX, R=R_PX, x0=x0,
                                  P0=P0_EKF_PX)
            initialized = True; continue
        f.predict()
        if det is None:
            n_drops += 1; continue
        f.update(det)
    if f is None:
        return np.array([]), n_drops
    return np.array(f.nis_log, dtype=float), n_drops


def main():
    available = iws_io.available_correction_methods()
    print(f"Available cached methods: {available}")
    methods = [m for m in METHODS if m in available]
    print(f"Running on: {methods}")

    # GT for ep3 is the right half of any of these MP4s; pull from open-loop.
    gt_ep = iws_io.load_episode("open-loop", "gt")
    raw_gt, dense_gt = perception_iws.detect_episode(gt_ep.frames)
    print(f"GT detection success: "
          f"{sum(d is not None for d in raw_gt)}/{len(raw_gt)}")

    # Sigma_a from GT pose only (single episode here)
    Sigma = estimate_sod_covariance([dense_gt])
    Sigma_inv = np.linalg.inv(Sigma)

    # Anomaly threshold from GT flow consistency (single episode pool)
    phi_gt = compute_flow_consistency(gt_ep.frames)
    tau = calibrate_anomaly_threshold(phi_gt)

    # GT-side baseline so we can compute deltas/ratios in the figure
    gt_nis, _ = _run_ekf(raw_gt)
    gt_nis_stats = nis_stats(gt_nis, dim=3, alpha=0.05)
    gt_sod = compute_sod_chi2(dense_gt, Sigma_inv)
    gt_sod_stats = nis_stats(gt_sod[~np.isnan(gt_sod)], dim=3, alpha=0.05)
    gt_phi_mean = float(phi_gt.mean())

    rows = []
    for method in methods:
        ep = iws_io.load_episode(method, "rollout")
        raw, dense = perception_iws.detect_episode(ep.frames)
        n_succ = sum(d is not None for d in raw)
        nis_arr, drops = _run_ekf(raw)
        ns = nis_stats(nis_arr, dim=3, alpha=0.05)
        sod_v = compute_sod_chi2(dense, Sigma_inv)
        sod_v = sod_v[~np.isnan(sod_v)]
        ss = nis_stats(sod_v, dim=3, alpha=0.05)
        phi = compute_flow_consistency(ep.frames)
        anomaly = compute_anomaly_rate(phi, tau)
        mse = float(compute_frame_mse(ep.frames, gt_ep.frames).mean())
        ncc = float(compute_ncc(ep.frames, gt_ep.frames).mean())

        rows.append(dict(
            task="pusht_iws",
            method=method,
            n_frames=len(ep.frames),
            det_success=n_succ,
            mean_nis=ns["mean_nis"],
            ks_pvalue_nis=ns["ks_pvalue"],
            sod_mean=ss["mean_nis"],
            ks_pvalue_sod=ss["ks_pvalue"],
            mean_flow_cons=float(phi.mean()),
            anomaly_rate=anomaly,
            mean_mse=mse,
            mean_ncc=ncc,
        ))
        print(f"  {method:22s} nis={ns['mean_nis']:6.2f} "
              f"sod={ss['mean_nis']:6.2f} "
              f"phi={phi.mean():.3f} mse={mse:6.0f} ncc={ncc:.3f}")

    # Insert a GT row at top for reference
    rows.insert(0, dict(
        task="pusht_iws", method="GT_reference",
        n_frames=len(gt_ep.frames), det_success=sum(d is not None for d in raw_gt),
        mean_nis=gt_nis_stats["mean_nis"],
        ks_pvalue_nis=gt_nis_stats["ks_pvalue"],
        sod_mean=gt_sod_stats["mean_nis"],
        ks_pvalue_sod=gt_sod_stats["ks_pvalue"],
        mean_flow_cons=gt_phi_mean,
        anomaly_rate=compute_anomaly_rate(phi_gt, tau),
        mean_mse=0.0,
        mean_ncc=1.0,
    ))

    csv_path = RESULTS / "summary_5c.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    _plot_correction_response(rows)


def _plot_correction_response(rows):
    """Grouped bar chart: methods x metrics, each metric normalized by the
    open-loop baseline for that metric so all bars sit on a common scale.
    """
    base = {r["method"]: r for r in rows}["open-loop"]
    metric_specs = [
        ("mean_nis", "NIS", "lower=better"),
        ("sod_mean", "SOD-chi2", "lower=better"),
        ("mean_flow_cons", "Flow phi", "higher=better"),
        ("anomaly_rate", "Anomaly", "lower=better"),
        ("mean_mse", "Frame MSE", "lower=better"),
        ("mean_ncc", "Frame NCC", "higher=better"),
    ]

    methods = [r["method"] for r in rows if r["method"] != "GT_reference"]
    n_methods = len(methods)
    n_metrics = len(metric_specs)

    width = 0.12
    x = np.arange(n_methods)
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, (col, label, dirn) in enumerate(metric_specs):
        vals = []
        for m in methods:
            r = next(rr for rr in rows if rr["method"] == m)
            v = float(r[col])
            base_v = float(base[col]) if float(base[col]) != 0 else 1.0
            vals.append(v / base_v if base_v != 0 else np.nan)
        ax.bar(x + (j - n_metrics / 2) * width, vals, width,
               label=f"{label} ({dirn})")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Ratio vs open-loop baseline (1.0 = same)")
    ax.set_title("§V.C correction-method response, all metrics "
                 "(normalized to open-loop)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    out = FIG / "fig_correction_response.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
