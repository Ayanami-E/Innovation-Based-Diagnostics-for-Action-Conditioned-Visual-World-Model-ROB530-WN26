"""§V.D driver: synthetic perturbations through the Part 1 EKF + SOD-chi^2.

Reuses Part 1's cached MuJoCo episodes (`results/part1_cache/episode_*.npz`)
and Part 1 EKF settings at sigma=10 (Regime A fit). Runs each
(perturbation, magnitude, episode) cell, pools NIS / SOD across
episodes, and writes `results/part1_6/summary_1_6.csv`.

Flow-phi note: the perturbations operate on the *pose* sequence, not on
images, so a faithful pixel-domain Flow-phi is not defined here. The
plan permits skipping it -- we report a structural argument in
PART1_6_NOTES.md instead. A toy "delta-coherence" proxy is included in
the CSV for completeness but should not be over-interpreted.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from part1.ekf_se2_cartesian import EKF_SE2_Cartesian
from part1.nis_analysis import nis_stats
from part1.se2 import wrap_angle
from part2.cross_metrics import (
    compute_sod_chi2,
    estimate_sod_covariance,
)

from part1_6.perturb import apply_perturbation, perturbation_severity

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "part1_cache"
RESULTS = ROOT / "results" / "part1_6"

# ---- Part 1 Regime A (synthetic, sigma=10) settings, copied verbatim --------
DT = 0.05
FOV_HALF = 0.5 * np.tan(np.radians(22.5))
IMG_SIZE = 256
M_PER_PX = (2 * FOV_HALF) / IMG_SIZE

Q_EKF = np.diag([1e-7, 1e-7, 1e-6, 1e-4, 1e-4, 1e-3])
R_POS = (10.0 * M_PER_PX) ** 2
R_ANG = 5e-4
R = np.diag([R_POS, R_POS, R_ANG])
P0 = np.diag([1e-6, 1e-6, 1e-6, 1e-2, 1e-2, 1e-1])

PERTURBATIONS = ["P1", "P2", "P3", "P4"]
MAGNITUDES = [0.0, 0.005, 0.01, 0.02, 0.05]


def _load_episodes() -> list[np.ndarray]:
    """Return GT pose arrays for every cached episode (no frames)."""
    eps = []
    for p in sorted(CACHE.glob("episode_*.npz")):
        d = np.load(p)
        eps.append(d["gt_poses"].astype(float))
    if not eps:
        raise FileNotFoundError(f"no cached episodes in {CACHE}")
    return eps


def _run_ekf(obs_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """EKF on a (T, 3) observation sequence; init from first pose."""
    x0 = np.zeros(6); x0[:3] = obs_seq[0]
    f = EKF_SE2_Cartesian(dt=DT, Q=Q_EKF, R=R, x0=x0, P0=P0)
    est = np.zeros_like(obs_seq)
    est[0] = f.pose()
    for t in range(1, obs_seq.shape[0]):
        f.predict()
        f.update(obs_seq[t])
        est[t] = f.pose()
    return np.array(f.nis_log, dtype=float), est


def _delta_coherence(obs_seq: np.ndarray) -> float:
    """Toy proxy for Flow-phi on a pose sequence: cosine similarity of
    consecutive (dx, dy) deltas to their mean direction. 1.0 = perfectly
    rigid translation. Documented as a proxy, not a substitute.
    """
    d = np.diff(obs_seq[:, :2], axis=0)
    if d.shape[0] == 0:
        return 1.0
    mag = np.linalg.norm(d, axis=1)
    keep = mag > 1e-6
    if keep.sum() < 2:
        return 1.0
    dn = d[keep] / mag[keep, None]
    mean_dir = dn.mean(axis=0)
    mn = np.linalg.norm(mean_dir)
    if mn < 1e-6:
        return 0.0
    return float((dn @ mean_dir / mn).mean())


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    episodes = _load_episodes()
    print(f"Loaded {len(episodes)} cached episodes; T={episodes[0].shape[0]}")

    # Sigma_a for SOD-chi^2 calibrated on the *clean* GT pose pool only.
    Sigma = estimate_sod_covariance(episodes)
    Sigma_inv = np.linalg.inv(Sigma)

    rows = []
    for kind in PERTURBATIONS:
        for m in MAGNITUDES:
            nis_pool = []
            sod_pool = []
            phi_pool = []
            pos_mean_pool = []
            pos_max_pool = []
            ori_mean_pool = []
            ori_max_pool = []
            for ep_gt in episodes:
                obs = apply_perturbation(ep_gt, kind, m)
                sev = perturbation_severity(obs, ep_gt)
                nis_arr, _ = _run_ekf(obs)
                sod_arr = compute_sod_chi2(obs, Sigma_inv)
                sod_arr = sod_arr[~np.isnan(sod_arr)]
                phi_pool.append(_delta_coherence(obs))
                nis_pool.append(nis_arr)
                sod_pool.append(sod_arr)
                pos_mean_pool.append(sev["pos_err_mean"])
                pos_max_pool.append(sev["pos_err_max"])
                ori_mean_pool.append(sev["ori_err_mean"])
                ori_max_pool.append(sev["ori_err_max"])
            nis_concat = np.concatenate(nis_pool)
            sod_concat = np.concatenate(sod_pool)
            ns = nis_stats(nis_concat, dim=3, alpha=0.05)
            ss = nis_stats(sod_concat, dim=3, alpha=0.05)
            row = dict(
                kind=kind, magnitude=m, n_ep=len(episodes),
                mean_nis=ns["mean_nis"],
                ks_pvalue=ns["ks_pvalue"],
                frac_in_CI_nis=ns["frac_in_CI"],
                sod_mean=ss["mean_nis"],
                sod_ks_pvalue=ss["ks_pvalue"],
                flow_phi_mean=float(np.mean(phi_pool)),
                pos_err_mean=float(np.mean(pos_mean_pool)),
                pos_err_max=float(np.max(pos_max_pool)),
                ori_err_mean=float(np.mean(ori_mean_pool)),
                ori_err_max=float(np.max(ori_max_pool)),
            )
            rows.append(row)
            print(f"  {kind} m={m:6.4f} nis={ns['mean_nis']:6.2f} "
                  f"sod={ss['mean_nis']:6.2f} "
                  f"phi={row['flow_phi_mean']:.3f} "
                  f"pos_max={row['pos_err_max']:.4f}m "
                  f"pos_mean={row['pos_err_mean']:.4f}m")

    csv_path = RESULTS / "summary_1_6.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")
    return rows


if __name__ == "__main__":
    main()
