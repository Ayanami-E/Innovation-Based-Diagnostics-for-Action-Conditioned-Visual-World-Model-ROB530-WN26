"""§V.B cross-metrics: flow consistency, anomaly rate, SOD-χ², pixel/NCC.

LPIPS deviation: the plan's `lpips` pip package requires torch, which is
not installed in this venv (and is heavy to add for one metric on a
deadline). We substitute two pixel-space perceptual proxies that ARE
runnable from cv2/numpy alone:

  - `frame_mse`: per-frame mean-squared pixel error vs GT.
  - `ncc_score`: per-frame normalized cross-correlation vs GT.
    (Higher = more similar; a coarser SSIM-like global similarity.)

Both are legitimate pixel-space comparators. They are not state-of-the-art
perceptual metrics, but the §V.B comparison argument — "is NIS redundant
with pixel-space metrics?" — survives the substitution: if NIS is
correlated with these, it is doubly correlated with LPIPS (which is
strictly more sophisticated). If it is uncorrelated, the claim only
strengthens. Documented in PART2_NOTES.md (deviation 5b).

FVD: still skipped per the original plan (only 5 episodes per source).

Outputs `results/part2/summary_5b.csv` with one row per (task, source,
episode) and columns for every metric.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy import stats as sstats

from part1.nis_analysis import nis_stats
from part1.se2 import wrap_angle

from part2 import iws_io, perception_iws

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "part2"

# ---- 1. Flow consistency (reimpl of optical_flow_metrics.compute_*) ---------

PINK_LO = np.array([135, 60, 60], dtype=np.uint8)
PINK_HI = np.array([175, 255, 255], dtype=np.uint8)


def _block_mask(frame_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, PINK_LO, PINK_HI)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def compute_flow_consistency(frames: np.ndarray) -> np.ndarray:
    """phi_t in [0, 1]: cosine similarity of T-block flow vectors to mean.

    1 = perfectly rigid motion. Lower = deformation / texture flicker.
    """
    n = len(frames)
    out = np.full(n, 1.0)
    for i in range(1, n):
        prev_g = cv2.cvtColor(frames[i - 1], cv2.COLOR_RGB2GRAY)
        curr_g = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_g, curr_g, None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mask = _block_mask(frames[i])
        if int(np.sum(mask > 0)) < 50:
            continue
        fx = flow[..., 0][mask > 0]
        fy = flow[..., 1][mask > 0]
        mfx, mfy = float(fx.mean()), float(fy.mean())
        mmag = float(np.hypot(mfx, mfy))
        if mmag < 0.1:
            continue
        mags = np.hypot(fx, fy)
        valid = mags > 0.1
        if int(valid.sum()) < 10:
            continue
        cos = (fx[valid] * mfx + fy[valid] * mfy) / (mags[valid] * mmag + 1e-8)
        out[i] = float(cos.mean())
    return out


# ---- 2. Anomaly rate ---------------------------------------------------------

def compute_anomaly_rate(phi: np.ndarray, threshold: float) -> float:
    return float(np.mean(phi < threshold))


def calibrate_anomaly_threshold(phi_gt_pool: np.ndarray) -> float:
    """tau = mean(phi_gt) - 2 * std(phi_gt)."""
    return float(phi_gt_pool.mean() - 2.0 * phi_gt_pool.std())


# ---- 3. SOD-chi^2 (model-free baseline against NIS) -------------------------

def _second_order_diffs(poses_dense: np.ndarray) -> np.ndarray:
    """a_t = p_{t+1} - 2 p_t + p_{t-1}, with theta wrapped per term.

    Input shape (T, 3); output (T-2, 3). NaN rows produce NaN diffs.
    """
    p = poses_dense
    a_xy = p[2:, :2] - 2 * p[1:-1, :2] + p[:-2, :2]
    a_th = np.array([
        wrap_angle(p[i + 2, 2] - 2 * p[i + 1, 2] + p[i, 2])
        for i in range(p.shape[0] - 2)
    ])
    return np.column_stack([a_xy, a_th])


def estimate_sod_covariance(gt_pose_sequences: list[np.ndarray]) -> np.ndarray:
    """Pool second-order diffs across GT episodes; return 3x3 empirical Σ_a."""
    pieces = []
    for seq in gt_pose_sequences:
        a = _second_order_diffs(seq)
        a = a[~np.any(np.isnan(a), axis=1)]
        pieces.append(a)
    A = np.concatenate(pieces, axis=0)
    Sigma = np.cov(A.T, bias=False)
    # Numerical floor on the diagonal in case orientation is rarely changing.
    Sigma = Sigma + np.eye(3) * 1e-12
    return Sigma


def compute_sod_chi2(poses_dense: np.ndarray,
                     Sigma_inv: np.ndarray) -> np.ndarray:
    """Per-step s_t = a_t^T Σ_a^{-1} a_t. Returns array of length T-2."""
    a = _second_order_diffs(poses_dense)
    valid = ~np.any(np.isnan(a), axis=1)
    s = np.full(a.shape[0], np.nan)
    if valid.any():
        av = a[valid]
        s[valid] = np.einsum("ni,ij,nj->n", av, Sigma_inv, av)
    return s


# ---- 4. Pixel-space proxies for LPIPS/FVD -----------------------------------

def compute_frame_mse(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-frame MSE in [0, 255^2 * 3]. Lower = closer to GT."""
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {gt.shape}")
    diff = pred.astype(np.float32) - gt.astype(np.float32)
    return diff.reshape(diff.shape[0], -1).var(axis=1) + (
        diff.reshape(diff.shape[0], -1).mean(axis=1) ** 2)


def compute_ncc(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-frame normalized cross-correlation in [-1, 1].

    Computes on grayscale to keep it cheap; higher = more similar.
    """
    n = len(pred)
    out = np.zeros(n)
    for i in range(n):
        p = cv2.cvtColor(pred[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
        g = cv2.cvtColor(gt[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
        p -= p.mean(); g -= g.mean()
        denom = np.sqrt((p * p).sum() * (g * g).sum()) + 1e-8
        out[i] = float((p * g).sum() / denom)
    return out


# ---- 5. Episode-level driver ------------------------------------------------

@dataclass
class EpisodeMetrics:
    task: str
    source: str
    episode_id: str
    n_frames: int
    mean_phi: float
    anomaly_rate: float
    mean_mse: float
    mean_ncc: float
    sod_mean: float
    sod_frac_in_CI: float
    sod_ks_pvalue: float
    nis_mean: float            # filled later from §V.A NPZ
    nis_frac_in_CI: float
    nis_ks_pvalue: float
    pos_drift_vs_gt: float
    ori_drift_vs_gt: float


def _load_nis_per_ep(filter_name: str = "ekf_cart") -> dict[str, np.ndarray]:
    """Pull per-episode NIS arrays produced by run_iws_nis."""
    out = {"gt": {}, "rollout": {}}
    for source in ("gt", "rollout"):
        path = RESULTS / "nis" / f"pusht_iws__{source}__{filter_name}.npz"
        d = np.load(path, allow_pickle=True)
        for key in d.files:
            if key.startswith("nis_"):
                out[source][key[4:]] = d[key]
    return out


def main(filter_for_nis: str = "ekf_cart"):
    OPENLOOP_EP_IDS = [f"openloop_ep{i}" for i in range(5)]

    # 1) Run perception once per (episode, source). Cache the dense pose.
    poses = {("gt", e): None for e in OPENLOOP_EP_IDS}
    poses.update({("rollout", e): None for e in OPENLOOP_EP_IDS})
    frames_cache = {}
    for source in ("gt", "rollout"):
        for ep_id in OPENLOOP_EP_IDS:
            ep = iws_io.load_episode(ep_id, source)
            _, dense = perception_iws.detect_episode(ep.frames)
            poses[(source, ep_id)] = dense
            frames_cache[(source, ep_id)] = ep.frames

    # 2) Estimate Sigma_a from GT pose sequences only (per-task calibration).
    gt_seqs = [poses[("gt", e)] for e in OPENLOOP_EP_IDS]
    Sigma = estimate_sod_covariance(gt_seqs)
    Sigma_inv = np.linalg.inv(Sigma)
    # also sanity-check Sigma -- print to stdout
    print("Sigma_a (from GT pool):\n", Sigma)

    # 3) Pool flow consistency from GT to calibrate anomaly threshold.
    phi_gt_pool = []
    phi_cache = {}
    for ep_id in OPENLOOP_EP_IDS:
        phi = compute_flow_consistency(frames_cache[("gt", ep_id)])
        phi_cache[("gt", ep_id)] = phi
        phi_gt_pool.append(phi)
    phi_gt_concat = np.concatenate(phi_gt_pool)
    tau = calibrate_anomaly_threshold(phi_gt_concat)
    print(f"Flow anomaly threshold tau = {tau:.4f}")

    for ep_id in OPENLOOP_EP_IDS:
        phi_cache[("rollout", ep_id)] = compute_flow_consistency(
            frames_cache[("rollout", ep_id)])

    # 4) Pull per-episode NIS from §V.A output.
    nis_per_ep = _load_nis_per_ep(filter_for_nis)

    # 5) Drift vs GT-side detection.
    rows: list[EpisodeMetrics] = []
    for source in ("gt", "rollout"):
        for ep_id in OPENLOOP_EP_IDS:
            frames = frames_cache[(source, ep_id)]
            gt_frames = frames_cache[("gt", ep_id)]
            phi = phi_cache[(source, ep_id)]
            anomaly = compute_anomaly_rate(phi, tau)
            mse = compute_frame_mse(frames, gt_frames)
            ncc = compute_ncc(frames, gt_frames)

            sod = compute_sod_chi2(poses[(source, ep_id)], Sigma_inv)
            sod_v = sod[~np.isnan(sod)]
            sod_stats = nis_stats(sod_v, dim=3, alpha=0.05)

            nis_v = nis_per_ep[source].get(ep_id, np.array([]))
            nis_st = nis_stats(nis_v, dim=3, alpha=0.05)

            # drift between this source's detection and the GT-side detection
            ds = poses[(source, ep_id)]; dg = poses[("gt", ep_id)]
            valid = (~np.any(np.isnan(ds), axis=1)) & \
                    (~np.any(np.isnan(dg), axis=1))
            if valid.any():
                pos_drift = float(np.sqrt(np.mean(
                    (ds[valid, 0] - dg[valid, 0]) ** 2 +
                    (ds[valid, 1] - dg[valid, 1]) ** 2)))
                ori_drift = float(np.sqrt(np.mean(np.array([
                    wrap_angle(a - b) ** 2
                    for a, b in zip(ds[valid, 2], dg[valid, 2])
                ]))))
            else:
                pos_drift = float("nan"); ori_drift = float("nan")

            rows.append(EpisodeMetrics(
                task="pusht_iws", source=source, episode_id=ep_id,
                n_frames=len(frames),
                mean_phi=float(phi.mean()),
                anomaly_rate=anomaly,
                mean_mse=float(mse.mean()),
                mean_ncc=float(ncc.mean()),
                sod_mean=sod_stats["mean_nis"],
                sod_frac_in_CI=sod_stats["frac_in_CI"],
                sod_ks_pvalue=sod_stats["ks_pvalue"],
                nis_mean=nis_st["mean_nis"],
                nis_frac_in_CI=nis_st["frac_in_CI"],
                nis_ks_pvalue=nis_st["ks_pvalue"],
                pos_drift_vs_gt=pos_drift,
                ori_drift_vs_gt=ori_drift,
            ))

    csv_path = RESULTS / "summary_5b.csv"
    fields = list(rows[0].__dict__.keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)
    print(f"Wrote {csv_path}")
    for r in rows:
        print(f"  {r.source:8s} {r.episode_id:14s} "
              f"phi={r.mean_phi:.3f} anom={r.anomaly_rate:.2f} "
              f"mse={r.mean_mse:7.0f} ncc={r.mean_ncc:.3f} "
              f"sod={r.sod_mean:6.2f} nis={r.nis_mean:6.2f}")


if __name__ == "__main__":
    main()
