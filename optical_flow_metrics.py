# my_project/flow_consistency_metrics.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'my_project')
from physics_metrics import extract_state

def get_block_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.inRange(hsv, np.array([130,40,80]), np.array([175,255,255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def compute_flow_consistency(pred_frames):
    """
    For each pair of adjacent frames:
    1. Compute dense optical flow.
    2. Compute directional consistency of the flow vectors inside the T-block mask.
    3. Low consistency = deformation.
    """
    consistency_scores = [1.0]  # first frame has no predecessor; set to 1

    for i in range(1, len(pred_frames)):
        prev = pred_frames[i-1]
        curr = pred_frames[i]

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)

        # optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # T-block mask
        mask = get_block_mask(curr)
        block_pixels = np.sum(mask > 0)

        if block_pixels < 50:
            consistency_scores.append(1.0)
            continue

        # flow vectors inside the mask
        fx = flow[..., 0][mask > 0]
        fy = flow[..., 1][mask > 0]

        # compute the mean direction
        mean_fx = fx.mean()
        mean_fy = fy.mean()
        mean_mag = np.sqrt(mean_fx**2 + mean_fy**2)

        if mean_mag < 0.1:
            # T-block barely moved; set consistency to 1
            consistency_scores.append(1.0)
            continue

        # cosine similarity of each pixel's flow direction against the mean direction
        mags = np.sqrt(fx**2 + fy**2)
        valid = mags > 0.1
        if valid.sum() < 10:
            consistency_scores.append(1.0)
            continue

        cos_sim = (fx[valid] * mean_fx + fy[valid] * mean_fy) / (
            mags[valid] * mean_mag + 1e-8)
        consistency = cos_sim.mean()  # 1 = fully consistent, 0 = fully random
        consistency_scores.append(float(consistency))

    return np.array(consistency_scores)


if __name__ == "__main__":
    from yixuan_utilities.hdf5_utils import load_dict_from_hdf5
    from yixuan_utilities.draw_utils import center_crop

    pred_frames = np.load("my_project/output/openloop_ep3_pred.npy")
    epi_data, _ = load_dict_from_hdf5("data/mini/pusht/val/episode_3.hdf5")
    gt_images = epi_data["obs"]["images"]["camera_1_color"][()]

    gt_frames = []
    for img in gt_images:
        img = center_crop(img, (128, 128))
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        gt_frames.append(img)
    gt_frames = np.array(gt_frames)

    pred_consistency = compute_flow_consistency(pred_frames)
    gt_consistency = compute_flow_consistency(gt_frames)

    print("=== Flow Consistency Stats ===")
    print(f"Pred: mean={pred_consistency.mean():.3f}  min={pred_consistency.min():.3f}")
    print(f"GT:   mean={gt_consistency.mean():.3f}  min={gt_consistency.min():.3f}")

    # Define the threshold from the GT distribution
    threshold = gt_consistency.mean() - 2 * gt_consistency.std()
    print(f"\nAnomaly threshold (GT mean-2std): {threshold:.3f}")

    pred_anomalies = np.where(pred_consistency < threshold)[0]
    gt_anomalies = np.where(gt_consistency < threshold)[0]
    print(f"Pred anomaly frames: {len(pred_anomalies)} ({len(pred_anomalies)/len(pred_consistency):.1%})")
    print(f"GT anomaly frames:   {len(gt_anomalies)} ({len(gt_anomalies)/len(gt_consistency):.1%})")

    print("\nTop 10 worst consistency frames (pred):")
    worst = np.argsort(pred_consistency)[:10]
    for t in worst:
        print(f"  frame {t}: consistency={pred_consistency[t]:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pred_consistency, label='Pred', color='red', alpha=0.8)
    ax.plot(gt_consistency, label='GT', color='blue', alpha=0.8)
    ax.axhline(y=threshold, color='gray', linestyle='--', label='Anomaly threshold')
    ax.set_title("Flow Consistency within T-block (1=rigid motion, low=deformation)")
    ax.set_ylabel("Cosine consistency")
    ax.set_xlabel("Timestep")
    ax.legend()
    plt.tight_layout()
    plt.savefig("my_project/output/flow_consistency.png", dpi=150)
    print("Saved to my_project/output/flow_consistency.png")