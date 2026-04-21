"""§V.A driver: NIS on IWS PushT for {gt, rollout} x {ekf_cart, riekf}.

Outputs:
  results/part2/nis/{task}__{source}__{filter}.npz
  results/part2/nis/{task}__{source}__{filter}__traces.npz
  results/part2/summary_5a.csv

Q and R are inherited from Part 1 (`run_part1.py`) at sigma=10, rescaled
from MuJoCo metric units (m, m/s) to IWS image-pixel units (px, px/s) by
the conversion factor `1/m_per_px` derived from MuJoCo's top-down camera.
This is the only allowed deviation per the plan and is documented in
PART2_NOTES.md (deviation 3).

`pos_rmse` and `ori_rmse` columns mean "filter estimate vs GT-side
detection" (per the deviation in PART2_NOTES.md), since true low-dim qpos
is not on disk. For source=gt, these measure the smoothing residual; for
source=rollout, they measure the world-model drift relative to physical
ground truth.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from part1.ekf_se2_cartesian import EKF_SE2_Cartesian
from part1.nis_analysis import nis_stats
from part1.riekf_se2 import RIEKF_SE2
from part1.se2 import pose_to_SE2, wrap_angle

from part2 import iws_io, perception_iws

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "part2"
NIS_DIR = RESULTS / "nis"

DT = 1.0 / 15.0  # MP4 fps; one filter step per IWS prediction.

# ---- Q, R derivation ---------------------------------------------------------
# MuJoCo Part 1 had a top-down camera with metric homography:
#   m_per_px_mujoco = 2 * 0.5 * tan(22.5 deg) / 256 = 0.001618 m/px
# Pixel-space scale factor: 1 / m_per_px_mujoco.
M_PER_PX_MUJOCO = 2 * 0.5 * np.tan(np.radians(22.5)) / 256.0
PX_SCALE2 = 1.0 / (M_PER_PX_MUJOCO ** 2)  # multiply m^2 entries by this

# Part 1 EKF Q (per step): [pos x3, theta, vel x3, omega] in m, m/s, rad, rad/s.
# Position and linear-velocity entries get the px-scale; angular entries stay.
Q_EKF_PART1 = np.diag([1e-7, 1e-7, 1e-6, 1e-4, 1e-4, 1e-3])
Q_EKF_PX = np.diag([
    1e-7 * PX_SCALE2,         # x  (px^2/step)
    1e-7 * PX_SCALE2,         # y
    1e-6,                     # theta (rad^2/step)
    1e-4 * PX_SCALE2,         # vx (px^2/s^2/step)
    1e-4 * PX_SCALE2,         # vy
    1e-3,                     # omega
])

# Part 1 RIEKF Q rates (per second). Same scaling logic.
Q_XI_PART1 = np.diag([2e-6, 2e-6, 2e-5])
Q_V_PART1 = np.diag([1e-3, 1e-3, 1e-2])
Q_XI_PX = np.diag([2e-6 * PX_SCALE2, 2e-6 * PX_SCALE2, 2e-5])
Q_V_PX = np.diag([1e-3 * PX_SCALE2, 1e-3 * PX_SCALE2, 1e-2])

# R at sigma=10 (Part 1): pos = (10 px * m_per_px)^2, angle = 5e-4 rad^2.
# In pixel space: pos = (10 px)^2 = 100 px^2.
R_PX = np.diag([100.0, 100.0, 5e-4])

# Initial covariances scaled the same way.
P0_EKF_PX = np.diag([1.0, 1.0, 1e-2, 100.0, 100.0, 1.0])
P0_RI_PX = np.diag([1.0, 1.0, 1e-2, 100.0, 100.0, 1.0])

# ---- Episode setup -----------------------------------------------------------

OPENLOOP_EP_IDS = [f"openloop_ep{i}" for i in range(5)]
TASK = "pusht_iws"


def _init_ekf(first_pose):
    x0 = np.zeros(6)
    x0[:3] = first_pose
    return EKF_SE2_Cartesian(dt=DT, Q=Q_EKF_PX, R=R_PX, x0=x0, P0=P0_EKF_PX)


def _init_riekf(first_pose):
    X0 = pose_to_SE2(first_pose[0], first_pose[1], first_pose[2])
    v0 = np.zeros(3)
    return RIEKF_SE2(dt=DT, Q_xi=Q_XI_PX, Q_v=Q_V_PX, R_meas=R_PX,
                     X0=X0, v0=v0, P0=P0_RI_PX)


def _run_filter(filter_obj, raw_dets):
    """Run predict/update across the episode. Returns:
        nis_arr, est_traj (T,3 with NaN where pre-init), drop_count.
    """
    T = len(raw_dets)
    est = np.full((T, 3), np.nan)
    nis_log = []
    initialized = False
    drops = 0

    for t, det in enumerate(raw_dets):
        if not initialized:
            if det is None:
                drops += 1
                continue
            # First successful detection: replace filter init with this pose.
            # The factory was called with a placeholder; rebuild now.
            filter_obj = (
                _init_ekf(det) if isinstance(filter_obj, EKF_SE2_Cartesian)
                else _init_riekf(det)
            )
            est[t] = filter_obj.pose()
            initialized = True
            continue

        filter_obj.predict()
        if det is None:
            drops += 1
            est[t] = filter_obj.pose()
            continue
        filter_obj.update(det)
        est[t] = filter_obj.pose()

    nis_arr = np.array(filter_obj.nis_log, dtype=float)
    return nis_arr, est, drops


def _drift_vs_gt(est_traj, gt_dense):
    """RMSE between filter estimate and GT-side detection (frame-by-frame).

    Both arrays are (T, 3). NaN positions are excluded. Returns
    (pos_rmse, ori_rmse) in (px, rad).
    """
    valid = (~np.any(np.isnan(est_traj), axis=1)) & \
            (~np.any(np.isnan(gt_dense), axis=1))
    if valid.sum() == 0:
        return float("nan"), float("nan")
    e = est_traj[valid]
    g = gt_dense[valid]
    pos_err = np.sqrt((e[:, 0] - g[:, 0]) ** 2 + (e[:, 1] - g[:, 1]) ** 2)
    ori_err = np.array([wrap_angle(a - b) for a, b in zip(e[:, 2], g[:, 2])])
    return (float(np.sqrt(np.mean(pos_err ** 2))),
            float(np.sqrt(np.mean(ori_err ** 2))))


def run_one(task: str, source: str, filter_name: str,
            episode_ids: list[str]):
    """Run one (task, source, filter) cell across episode_ids. Returns dict."""
    pooled_nis = []
    per_ep_nis = {}
    n_steps = 0
    n_drop_total = 0
    n_frames_total = 0
    pos_rmse_vals = []
    ori_rmse_vals = []
    all_traces: dict[str, np.ndarray] = {}

    for ep_id in episode_ids:
        ep_src = iws_io.load_episode(ep_id, source)
        # GT-side detection for the drift comparison (always run on GT frames)
        ep_gt = iws_io.load_episode(ep_id, "gt")
        _, gt_dense = perception_iws.detect_episode(ep_gt.frames)
        raw_dets, raw_dense = perception_iws.detect_episode(ep_src.frames)
        n_frames_total += len(raw_dets)

        # Build a placeholder filter so isinstance check inside _run_filter
        # routes to the right re-init at the first successful detection.
        first = next((d for d in raw_dets if d is not None), None)
        if first is None:
            print(f"  [skip] {ep_id}/{source}: no detections in entire episode")
            continue
        if filter_name == "ekf_cart":
            filt = _init_ekf(first)
        elif filter_name == "riekf":
            filt = _init_riekf(first)
        else:
            raise ValueError(f"unknown filter {filter_name!r}")

        nis_arr, est, drops = _run_filter(filt, raw_dets)
        pooled_nis.append(nis_arr)
        per_ep_nis[ep_id] = nis_arr
        n_steps += nis_arr.size
        n_drop_total += drops

        pos_rmse, ori_rmse = _drift_vs_gt(est, gt_dense)
        pos_rmse_vals.append(pos_rmse)
        ori_rmse_vals.append(ori_rmse)

        idx = ep_id.replace("openloop_ep", "ep")
        all_traces[f"raw_{idx}"] = raw_dense
        all_traces[f"est_{idx}"] = est
        all_traces[f"gt_{idx}"] = gt_dense
        all_traces[f"nis_{idx}"] = nis_arr

        print(f"  {ep_id}/{source}/{filter_name}: "
              f"mean_nis={nis_arr.mean():.2f} "
              f"drops={drops}/{len(raw_dets)} "
              f"pos_rmse={pos_rmse:.2f}px ori_rmse={np.degrees(ori_rmse):+.1f}deg")

    if not pooled_nis:
        return None
    nis_concat = np.concatenate(pooled_nis)
    stats = nis_stats(nis_concat, dim=3, alpha=0.05)
    drop_rate = n_drop_total / max(n_frames_total, 1)

    NIS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        NIS_DIR / f"{task}__{source}__{filter_name}.npz",
        nis_pooled=nis_concat,
        **{f"nis_{ep}": v for ep, v in per_ep_nis.items()},
    )
    if all_traces:
        np.savez_compressed(
            NIS_DIR / f"{task}__{source}__{filter_name}__traces.npz",
            episode_ids=np.array(list(per_ep_nis.keys())),
            **all_traces,
        )

    return dict(
        task=task,
        source=source,
        filter=filter_name,
        mean_nis=stats["mean_nis"],
        frac_in_CI=stats["frac_in_CI"],
        ks_pvalue=stats["ks_pvalue"],
        ks_stat=stats["ks_stat"],
        pos_rmse_vs_gt_det=float(np.mean(pos_rmse_vals)),
        ori_rmse_vs_gt_det=float(np.mean(ori_rmse_vals)),
        drop_rate=drop_rate,
        n_episodes=len(per_ep_nis),
        n_steps=n_steps,
    )


def main():
    rows = []
    for source in ("gt", "rollout"):
        for filter_name in ("ekf_cart", "riekf"):
            print(f"--- {TASK} / source={source} / filter={filter_name}")
            row = run_one(TASK, source, filter_name, OPENLOOP_EP_IDS)
            if row is not None:
                rows.append(row)

    csv_path = RESULTS / "summary_5a.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote {csv_path}")
    for r in rows:
        print(f"  {r['source']:8s} {r['filter']:10s} "
              f"mean_nis={r['mean_nis']:8.3f} "
              f"frac_in_CI={r['frac_in_CI']:.2f} "
              f"ks_p={r['ks_pvalue']:.2e} "
              f"drift_pos={r['pos_rmse_vs_gt_det']:5.1f}px "
              f"drift_ori={np.degrees(r['ori_rmse_vs_gt_det']):+6.1f}deg")


if __name__ == "__main__":
    main()
