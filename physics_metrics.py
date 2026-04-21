# my_project/physics_metrics.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5
from yixuan_utilities.draw_utils import center_crop


def extract_state(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    kernel = np.ones((5, 5), np.uint8)

    lower_block = np.array([130, 40, 80])
    upper_block = np.array([175, 255, 255])
    block_mask = cv2.inRange(hsv, lower_block, upper_block)
    block_mask = cv2.morphologyEx(block_mask, cv2.MORPH_CLOSE, kernel)
    block_mask = cv2.morphologyEx(block_mask, cv2.MORPH_OPEN, kernel)

    lower_robot = np.array([0, 80, 80])
    upper_robot = np.array([25, 255, 255])
    robot_mask = cv2.inRange(hsv, lower_robot, upper_robot)
    robot_mask = cv2.morphologyEx(robot_mask, cv2.MORPH_CLOSE, kernel)

    state = {}
    block_contours, _ = cv2.findContours(
        block_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if block_contours:
        largest = max(block_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > 50:
            M = cv2.moments(largest)
            if M['m00'] > 0:
                state['block_cx'] = M['m10'] / M['m00']
                state['block_cy'] = M['m01'] / M['m00']
            rect = cv2.minAreaRect(largest)
            state['block_angle'] = rect[2]
            state['block_area'] = area
        else:
            state['block_cx'] = state['block_cy'] = 0
            state['block_angle'] = 0
            state['block_area'] = 0
    else:
        state['block_cx'] = state['block_cy'] = 0
        state['block_angle'] = 0
        state['block_area'] = 0

    penetration = cv2.bitwise_and(block_mask, robot_mask)
    state['penetration_area'] = int(np.sum(penetration > 0))
    return state


def compute_physics_metrics(pred_frames, angle_jump_thresh=10.0):
    """
    Core metrics:
    - angle_jump: per-frame orientation change (>10 deg counts as an anomaly)
    - area_change: per-frame area change
    - penetration: T-block / robot overlap
    - anomaly_count: number of angle-anomalous frames
    """
    penetration_list = []
    centroid_jump_list = []
    angle_jump_list = []
    area_list = []
    anomaly_frames = []

    prev_state = None
    for i, frame in enumerate(pred_frames):
        state = extract_state(frame)
        penetration_list.append(state['penetration_area'])
        area_list.append(state['block_area'])

        if prev_state is not None:
            dx = state['block_cx'] - prev_state['block_cx']
            dy = state['block_cy'] - prev_state['block_cy']
            jump = np.sqrt(dx**2 + dy**2)
            centroid_jump_list.append(jump)

            da = abs(state['block_angle'] - prev_state['block_angle'])
            da = min(da, 90 - da) if da > 45 else da
            angle_jump_list.append(da)

            if da > angle_jump_thresh:
                anomaly_frames.append(i)
        else:
            centroid_jump_list.append(0)
            angle_jump_list.append(0)

        prev_state = state

    return {
        'penetration': np.array(penetration_list),
        'centroid_jump': np.array(centroid_jump_list),
        'angle_jump': np.array(angle_jump_list),
        'area': np.array(area_list),
        'anomaly_frames': anomaly_frames,
        'anomaly_rate': len(anomaly_frames) / len(pred_frames),
    }


def compare_methods(method_results, episode_id, save_path):
    """
    Compare metric curves across several methods.
    method_results: dict mapping method name -> return value of compute_physics_metrics.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    colors = {'open_loop': 'red', 'periodic_reset': 'orange',
              'selective_correction': 'green'}

    for method, results in method_results.items():
        color = colors.get(method, 'blue')
        T = len(results['angle_jump'])
        axes[0].plot(results['angle_jump'], label=method, color=color, alpha=0.8)
        axes[1].plot(results['centroid_jump'], label=method, color=color, alpha=0.8)
        axes[2].plot(results['area'], label=method, color=color, alpha=0.8)

    axes[0].axhline(y=10, color='gray', linestyle='--', label='anomaly threshold')
    axes[0].set_title(f"Episode {episode_id}: Orientation Jump (>10° = anomaly)")
    axes[0].set_ylabel("Degrees")
    axes[0].legend()

    axes[1].set_title("Centroid Jump (frame-to-frame)")
    axes[1].set_ylabel("Pixels")
    axes[1].legend()

    axes[2].set_title("T-block Area over Time")
    axes[2].set_ylabel("Pixels")
    axes[2].set_xlabel("Timestep")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    print("=== Open-loop Physics Metrics ===")
    for i in range(5):
        frames = np.load(f"my_project/output/openloop_ep{i}_pred.npy")
        results = compute_physics_metrics(frames, angle_jump_thresh=20.0)
        print(f"ep{i}: "
              f"angle_jump_avg={results['angle_jump'].mean():.2f}deg  "
              f"max_angle_jump={results['angle_jump'].max():.1f}deg  "
              f"anomaly_rate={results['anomaly_rate']:.1%}  "
              f"anomaly_frames={len(results['anomaly_frames'])}")