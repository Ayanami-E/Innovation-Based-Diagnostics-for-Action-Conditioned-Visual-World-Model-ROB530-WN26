"""Legacy entry point — now a thin wrapper over the part1/ package.

Runs a single MuJoCo PushT episode, applies the OpenCV-based detector, and
compares raw detections vs. the EKF-Cartesian filter. Kept for backwards
compatibility (the full sweep lives in `python -m part1.run_part1`).
"""

from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from part1.scene import generate_episode, DT, IMG_SIZE
from part1.perception import detect_tblock, world_to_pixel
from part1.ekf_se2_cartesian import EKF_SE2_Cartesian
from part1.se2 import wrap_angle


FOV_HALF = 0.5 * np.tan(np.radians(22.5))

Q = np.diag([1e-7, 1e-7, 1e-6, 1e-4, 1e-4, 1e-3])
R = np.diag([(1.0 * (2 * FOV_HALF / IMG_SIZE)) ** 2,
             (1.0 * (2 * FOV_HALF / IMG_SIZE)) ** 2,
             5e-4])
P0 = np.diag([1e-6, 1e-6, 1e-6, 1e-2, 1e-2, 1e-1])


def pos_rmse(est, gt):
    return np.sqrt(np.mean((est[:, 0] - gt[:, 0]) ** 2 +
                           (est[:, 1] - gt[:, 1]) ** 2))


def ang_rmse(est, gt):
    return np.sqrt(np.mean(
        np.array([wrap_angle(e - g) for e, g in zip(est[:, 2], gt[:, 2])]) ** 2
    ))


def run_part1(output_dir=None, n_steps=300, seed=42):
    if output_dir is None:
        output_dir = Path(__file__).parent / "output" / "part1"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Part 1: MuJoCo PushT - EKF-Cartesian (legacy wrapper)")
    print("=" * 60)

    print("\n[1/4] Generating MuJoCo PushT episode...")
    gt_states, images = generate_episode(n_steps=n_steps, seed=seed)
    print(f"  Episode: {len(images)} frames")

    print("\n[2/4] Running OpenCV detection...")
    opencv_world = np.zeros((len(images), 3))
    prev = None
    failures = 0
    for i, img in enumerate(images):
        p = detect_tblock(img, prev_theta=prev, fov_half_width=FOV_HALF)
        if p is None:
            opencv_world[i] = opencv_world[i - 1] if i > 0 else gt_states[0]
            failures += 1
        else:
            opencv_world[i] = p
            prev = p[2]
    print(f"  Detection failures: {failures}/{len(images)}")

    print("\n[3/4] Running EKF-Cartesian...")
    init = opencv_world[0]
    x0 = np.array([init[0], init[1], init[2], 0.0, 0.0, 0.0])
    kf = EKF_SE2_Cartesian(DT, Q, R, x0, P0)
    kf_states = np.zeros((len(images), 3))
    kf_states[0] = kf.pose()
    for i in range(1, len(images)):
        kf.predict()
        kf.update(opencv_world[i])
        kf_states[i] = kf.pose()

    print("\n[4/4] Evaluating...")
    r_cv_p = pos_rmse(opencv_world, gt_states)
    r_cv_a = ang_rmse(opencv_world, gt_states)
    r_kf_p = pos_rmse(kf_states, gt_states)
    r_kf_a = ang_rmse(kf_states, gt_states)

    print("\n" + "=" * 55)
    print(f"{'Method':<12} {'Pos RMSE (m)':>14} {'Angle RMSE (rad)':>18}")
    print("-" * 55)
    print(f"{'OpenCV':<12} {r_cv_p:>14.5f} {r_cv_a:>18.4f}")
    print(f"{'KF':<12} {r_kf_p:>14.5f} {r_kf_a:>18.4f}")
    print("-" * 55)
    if r_cv_p > 1e-9:
        print(f"{'Improve':<12} {(1 - r_kf_p / r_cv_p) * 100:>13.1f}% "
              f"{(1 - r_kf_a / r_cv_a) * 100:>17.1f}%")
    print("=" * 55)

    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t = np.arange(len(gt_states))
    ax = axes[0, 0]
    ax.plot(t, gt_states[:, 0] * 100, "k-", lw=2, label="GT")
    ax.plot(t, opencv_world[:, 0] * 100, "r-", alpha=0.5, label="OpenCV")
    ax.plot(t, kf_states[:, 0] * 100, "b-", alpha=0.8, label="KF")
    ax.set_title("X Position (cm)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, gt_states[:, 1] * 100, "k-", lw=2, label="GT")
    ax.plot(t, opencv_world[:, 1] * 100, "r-", alpha=0.5, label="OpenCV")
    ax.plot(t, kf_states[:, 1] * 100, "b-", alpha=0.8, label="KF")
    ax.set_title("Y Position (cm)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, np.degrees(gt_states[:, 2]), "k-", lw=2, label="GT")
    ax.plot(t, np.degrees(opencv_world[:, 2]), "r-", alpha=0.5, label="OpenCV")
    ax.plot(t, np.degrees(kf_states[:, 2]), "b-", alpha=0.8, label="KF")
    ax.set_title("Orientation (deg)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(gt_states[:, 0] * 100, gt_states[:, 1] * 100, "k-", lw=2, label="GT")
    ax.plot(opencv_world[:, 0] * 100, opencv_world[:, 1] * 100, "r-",
            alpha=0.5, label="OpenCV")
    ax.plot(kf_states[:, 0] * 100, kf_states[:, 1] * 100, "b-",
            alpha=0.8, label="KF")
    ax.set_title("2D Trajectory (cm)")
    ax.set_aspect("equal"); ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle("Part 1 (legacy wrapper): MuJoCo PushT + EKF-Cartesian",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    png = output_dir / "part1_comparison.png"
    plt.savefig(str(png), dpi=150)
    print(f"  Saved: {png}")
    plt.close(fig)

    np.savez(str(output_dir / "part1_data.npz"),
             gt_states=gt_states,
             opencv_world=opencv_world,
             kf_states=kf_states)
    print(f"  Saved: {output_dir / 'part1_data.npz'}")

    # Simple video: overlay GT, OpenCV, KF markers
    print("\nSaving video...")
    video = cv2.VideoWriter(
        str(output_dir / "part1_demo.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), 20, (IMG_SIZE, IMG_SIZE))
    for i, img in enumerate(images):
        vis = img.copy()
        gx, gy = world_to_pixel(gt_states[i, 0], gt_states[i, 1],
                                fov_half_width=FOV_HALF)
        ox, oy = world_to_pixel(opencv_world[i, 0], opencv_world[i, 1],
                                fov_half_width=FOV_HALF)
        kx, ky = world_to_pixel(kf_states[i, 0], kf_states[i, 1],
                                fov_half_width=FOV_HALF)
        cv2.drawMarker(vis, (int(gx), int(gy)), (0, 200, 0),
                       cv2.MARKER_CROSS, 15, 2)
        cv2.circle(vis, (int(ox), int(oy)), 5, (255, 50, 50), 2)
        cv2.circle(vis, (int(kx), int(ky)), 5, (50, 50, 255), 2)
        cv2.putText(vis, f"frame {i}", (5, IMG_SIZE - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        video.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    video.release()
    print(f"  Saved: {output_dir / 'part1_demo.mp4'}")

    print("\nPart 1 legacy wrapper complete.")


if __name__ == "__main__":
    run_part1()
