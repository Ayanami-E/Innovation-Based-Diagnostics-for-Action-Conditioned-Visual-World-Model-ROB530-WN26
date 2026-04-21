"""Experiment driver for Part 1: sweep sigma x filter, emit CSVs + NIS logs.

Two observation sources:

  --obs-source real       (default)  pixel-corruption → OpenCV detector
  --obs-source synthetic             y_t = gt_t + N(0, R_target)

Results land in `results/real/` or `results/synthetic/` respectively so the
two sweeps don't clobber each other.

Usage:
    python -m part1.run_part1                     # real, default
    python -m part1.run_part1 --obs-source real
    python -m part1.run_part1 --obs-source synthetic
"""

from pathlib import Path
import argparse
import csv
import json

import numpy as np

from part1.scene import generate_episode, DT, IMG_SIZE
from part1.perception import detect_tblock, pixel_to_world
from part1.corruption import apply_corruption
from part1.ekf_se2_cartesian import EKF_SE2_Cartesian
from part1.riekf_se2 import RIEKF_SE2
from part1.nis_analysis import nis_stats, rmse_pose
from part1.se2 import pose_to_SE2

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CACHE = RESULTS / "part1_cache"

N_EPISODES = 20
N_STEPS = 150
SIGMAS = [0, 5, 10, 20]
BASE_SEED = 20250417

FOV_HALF = 0.5 * np.tan(np.radians(22.5))
METERS_PER_PIXEL = (2 * FOV_HALF) / IMG_SIZE

# Real-mode R: pixel-noise floor prevents a singular gain at sigma=0 when the
# OpenCV detector is producing measurements. Angle R is fixed per plan.
SIGMA_FLOOR_PX_REAL = 1.0
R_ANGLE = 5e-4  # rad^2; fixed per plan

# Synthetic-mode R at sigma=0: we inject *exactly* N(0, 0) on (x, y), so
# R_pos=0 makes the chi-squared NIS well-defined via S = H P H^T (driven by
# prediction covariance). A tiny numeric floor keeps linalg operations
# stable without materially changing statistics.
R_POS_NUMERIC_FLOOR = 1e-14
R_ANGLE_NUMERIC_FLOOR = 1e-14

# Process-noise choices (shared defaults for both filters).
# NOTE: Per the user's directive for this pass, Q is frozen — do not tune.
Q_EKF = np.diag([1e-7, 1e-7, 1e-6, 1e-4, 1e-4, 1e-3])
Q_XI = np.diag([2e-6, 2e-6, 2e-5])    # world-frame twist cov / sec
Q_V = np.diag([1e-3, 1e-3, 1e-2])     # body-frame velocity cov / sec

P0_EKF = np.diag([1e-6, 1e-6, 1e-6, 1e-2, 1e-2, 1e-1])
P0_RI = np.diag([1e-6, 1e-6, 1e-6, 1e-2, 1e-2, 1e-1])


def compute_R_real(sigma_px):
    sigma_eff = max(sigma_px, SIGMA_FLOOR_PX_REAL)
    sigma_m = sigma_eff * METERS_PER_PIXEL
    return np.diag([sigma_m ** 2, sigma_m ** 2, R_ANGLE])


def compute_R_synthetic(sigma_px):
    """Strict R per user spec: diag((σ·m/px)², (σ·m/px)², 5e-4).

    A numeric floor is added to keep matrix inversion stable at σ=0 without
    changing the statistical model in any practically relevant way.
    """
    sigma_m = sigma_px * METERS_PER_PIXEL
    return np.diag([sigma_m ** 2 + R_POS_NUMERIC_FLOOR,
                    sigma_m ** 2 + R_POS_NUMERIC_FLOOR,
                    R_ANGLE + R_ANGLE_NUMERIC_FLOOR])


def _cache_path(ep_idx):
    return CACHE / f"episode_{ep_idx:02d}.npz"


def load_or_generate_episodes(need_frames=True):
    CACHE.mkdir(parents=True, exist_ok=True)
    episodes = []
    for i in range(N_EPISODES):
        p = _cache_path(i)
        if p.exists():
            d = np.load(p)
            gt = d["gt_poses"]
            frames = d["frames"] if need_frames else None
        else:
            gt, frames_list = generate_episode(n_steps=N_STEPS,
                                               seed=BASE_SEED + i)
            frames = np.stack(frames_list, axis=0).astype(np.uint8)
            np.savez_compressed(p, gt_poses=gt, frames=frames)
            print(f"  cached {p.name} ({frames.shape})")
            if not need_frames:
                frames = None
        episodes.append((gt, frames))
    return episodes


def detect_episode_real(frames, sigma_px, corrupt_seed):
    """Real mode: corrupt frames + run OpenCV detector."""
    rng = np.random.default_rng(corrupt_seed)
    T = len(frames)
    dets = np.zeros((T, 3))
    valid = np.zeros(T, dtype=bool)
    prev_theta = None
    for t, frame in enumerate(frames):
        img = apply_corruption(frame, sigma_pixel=sigma_px,
                               rng=rng) if sigma_px > 0 else frame
        pose = detect_tblock(img, prev_theta=prev_theta,
                             fov_half_width=FOV_HALF)
        if pose is None:
            dets[t] = dets[t - 1] if t > 0 else 0.0
            valid[t] = False
        else:
            dets[t] = pose
            valid[t] = True
            prev_theta = pose[2]
    return dets, valid


def synthetic_episode(gt, sigma_px, R_target, sim_seed):
    """Synthetic mode: y_t = gt_t + N(0, R_target). Angle is wrapped.

    Injected noise covariance matches R_target *exactly*, guaranteeing a
    correctly-specified observation model for the filter.
    """
    rng = np.random.default_rng(sim_seed)
    T = len(gt)
    # Sample innovation noise with covariance R_target (diagonal).
    sigma_pos = np.sqrt(R_target[0, 0])  # equal on x & y
    sigma_ang = np.sqrt(R_target[2, 2])
    noise = np.zeros((T, 3))
    noise[:, 0] = rng.standard_normal(T) * sigma_pos
    noise[:, 1] = rng.standard_normal(T) * sigma_pos
    noise[:, 2] = rng.standard_normal(T) * sigma_ang

    dets = gt + noise
    dets[:, 2] = (dets[:, 2] + np.pi) % (2 * np.pi) - np.pi
    valid = np.ones(T, dtype=bool)
    return dets, valid


def _initial_pose(detections, valid, gt0):
    for t in range(len(detections)):
        if valid[t]:
            return detections[t]
    return gt0


def run_ekf_cart(detections, valid, gt0, R):
    init_pose = _initial_pose(detections, valid, gt0)
    x0 = np.array([init_pose[0], init_pose[1], init_pose[2],
                   0.0, 0.0, 0.0])
    kf = EKF_SE2_Cartesian(dt=DT, Q=Q_EKF, R=R, x0=x0, P0=P0_EKF)
    T = len(detections)
    est = np.zeros((T, 3))
    est[0] = kf.pose()
    for t in range(1, T):
        kf.predict()
        if valid[t]:
            kf.update(detections[t])
        est[t] = kf.pose()
    return est, np.array(kf.nis_log), np.stack(kf.innov_log) if kf.innov_log else np.zeros((0, 3))


def run_riekf(detections, valid, gt0, R):
    init_pose = _initial_pose(detections, valid, gt0)
    X0 = pose_to_SE2(init_pose[0], init_pose[1], init_pose[2])
    v0 = np.zeros(3)
    kf = RIEKF_SE2(dt=DT, Q_xi=Q_XI, Q_v=Q_V, R_meas=R,
                   X0=X0, v0=v0, P0=P0_RI)
    T = len(detections)
    est = np.zeros((T, 3))
    est[0] = kf.pose()
    for t in range(1, T):
        kf.predict()
        if valid[t]:
            kf.update(detections[t])
        est[t] = kf.pose()
    return est, np.array(kf.nis_log), np.stack(kf.innov_log) if kf.innov_log else np.zeros((0, 3))


def run_raw_baseline(detections, valid):
    T = len(detections)
    est = detections.copy()
    last_good = None
    for t in range(T):
        if valid[t]:
            last_good = detections[t]
        elif last_good is not None:
            est[t] = last_good
    return est


def main(obs_source="real"):
    assert obs_source in ("real", "synthetic")

    out_dir = RESULTS / obs_source
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[part1] MuJoCo sweep  obs-source={obs_source}  "
          f"N_ep={N_EPISODES}  T={N_STEPS}  sigmas={SIGMAS}")
    print(f"[part1] meters/pixel = {METERS_PER_PIXEL:.6f}")

    print("[part1] Loading / generating episodes...")
    # Synthetic mode only needs GT, but we still cache frames for the real
    # pass; loading them is fine.
    episodes = load_or_generate_episodes(need_frames=(obs_source == "real"))

    rows = []
    pooled = {}

    for sigma in SIGMAS:
        if obs_source == "real":
            R = compute_R_real(sigma)
        else:
            R = compute_R_synthetic(sigma)
        print(f"\n[part1] sigma={sigma} px   R=diag({np.diag(R)})")

        raw_pos, raw_ori = [], []
        ekf_pos, ekf_ori = [], []
        ri_pos, ri_ori = [], []
        ekf_nis_pool, ri_nis_pool = [], []
        drop_count = total_count = 0

        for ep in range(N_EPISODES):
            gt, frames = episodes[ep]
            sim_seed = BASE_SEED + ep + 1000 * (sigma + 1)
            if obs_source == "real":
                dets, valid = detect_episode_real(frames, sigma, sim_seed)
            else:
                dets, valid = synthetic_episode(gt, sigma, R, sim_seed)
            total_count += len(valid)
            drop_count += int((~valid).sum())

            raw_est = run_raw_baseline(dets, valid)
            pr, orr = rmse_pose(raw_est, gt)
            raw_pos.append(pr); raw_ori.append(orr)

            ekf_est, ekf_nis, _ = run_ekf_cart(dets, valid, gt[0], R)
            pe, oe = rmse_pose(ekf_est, gt)
            ekf_pos.append(pe); ekf_ori.append(oe)
            ekf_nis_pool.append(ekf_nis)

            ri_est, ri_nis, _ = run_riekf(dets, valid, gt[0], R)
            pr_, or_ = rmse_pose(ri_est, gt)
            ri_pos.append(pr_); ri_ori.append(or_)
            ri_nis_pool.append(ri_nis)

        drop_rate = drop_count / total_count if total_count else 0.0
        ekf_nis_flat = np.concatenate(ekf_nis_pool) if ekf_nis_pool else np.array([])
        ri_nis_flat = np.concatenate(ri_nis_pool) if ri_nis_pool else np.array([])

        pooled[(sigma, "ekf_cart")] = ekf_nis_flat
        pooled[(sigma, "riekf")] = ri_nis_flat

        ekf_stats = nis_stats(ekf_nis_flat, dim=3)
        ri_stats = nis_stats(ri_nis_flat, dim=3)

        rows.append(dict(sigma=sigma, filter="raw",
                         pos_rmse=float(np.mean(raw_pos)),
                         ori_rmse=float(np.mean(raw_ori)),
                         mean_nis=float("nan"),
                         frac_in_CI=float("nan"),
                         ks_pvalue=float("nan"),
                         drop_rate=drop_rate))
        rows.append(dict(sigma=sigma, filter="ekf_cart",
                         pos_rmse=float(np.mean(ekf_pos)),
                         ori_rmse=float(np.mean(ekf_ori)),
                         mean_nis=ekf_stats["mean_nis"],
                         frac_in_CI=ekf_stats["frac_in_CI"],
                         ks_pvalue=ekf_stats["ks_pvalue"],
                         drop_rate=drop_rate))
        rows.append(dict(sigma=sigma, filter="riekf",
                         pos_rmse=float(np.mean(ri_pos)),
                         ori_rmse=float(np.mean(ri_ori)),
                         mean_nis=ri_stats["mean_nis"],
                         frac_in_CI=ri_stats["frac_in_CI"],
                         ks_pvalue=ri_stats["ks_pvalue"],
                         drop_rate=drop_rate))

        print(f"  raw      pos={np.mean(raw_pos):.5f} m  "
              f"ori={np.degrees(np.mean(raw_ori)):.3f} deg  "
              f"drop={drop_rate:.3f}")
        print(f"  ekf_cart pos={np.mean(ekf_pos):.5f} m  "
              f"ori={np.degrees(np.mean(ekf_ori)):.3f} deg  "
              f"NIS mean={ekf_stats['mean_nis']:.2f}  "
              f"CI={ekf_stats['frac_in_CI']:.2f}  "
              f"KS p={ekf_stats['ks_pvalue']:.3f}")
        print(f"  riekf    pos={np.mean(ri_pos):.5f} m  "
              f"ori={np.degrees(np.mean(ri_ori)):.3f} deg  "
              f"NIS mean={ri_stats['mean_nis']:.2f}  "
              f"CI={ri_stats['frac_in_CI']:.2f}  "
              f"KS p={ri_stats['ks_pvalue']:.3f}")

    csv_path = out_dir / "part1_summary.csv"
    with open(csv_path, "w", newline="") as f:
        fields = ["sigma", "filter", "pos_rmse", "ori_rmse",
                  "mean_nis", "frac_in_CI", "ks_pvalue", "drop_rate"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[part1] wrote {csv_path}")

    nis_path = out_dir / "part1_nis_pooled.npz"
    np.savez(nis_path, **{f"{sig}_{flt}": arr
                          for (sig, flt), arr in pooled.items()})
    print(f"[part1] wrote {nis_path}")

    (out_dir / "part1_config.json").write_text(json.dumps(dict(
        obs_source=obs_source,
        N_EPISODES=N_EPISODES, N_STEPS=N_STEPS, SIGMAS=SIGMAS,
        BASE_SEED=BASE_SEED, FOV_HALF=float(FOV_HALF),
        METERS_PER_PIXEL=float(METERS_PER_PIXEL),
        R_ANGLE=R_ANGLE, SIGMA_FLOOR_PX_REAL=SIGMA_FLOOR_PX_REAL,
        Q_EKF=np.diag(Q_EKF).tolist(),
        Q_XI=np.diag(Q_XI).tolist(),
        Q_V=np.diag(Q_V).tolist(),
    ), indent=2))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-source", choices=["real", "synthetic"],
                    default="real",
                    help="Measurement source: 'real' (detector) or "
                         "'synthetic' (gt + N(0, R_target)).")
    args = ap.parse_args()
    main(obs_source=args.obs_source)
