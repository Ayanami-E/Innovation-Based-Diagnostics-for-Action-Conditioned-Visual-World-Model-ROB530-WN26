# my_project/check_gt_metrics.py
import numpy as np
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5
from yixuan_utilities.draw_utils import center_crop
import cv2
import sys
sys.path.insert(0, 'my_project')
from physics_metrics import compute_physics_metrics

print("=== GT Physics Metrics ===")
for i in range(5):
    epi_data, _ = load_dict_from_hdf5(f"data/mini/pusht/val/episode_{i}.hdf5")
    gt_images = epi_data["obs"]["images"]["camera_1_color"][()]

    gt_frames = []
    for img in gt_images:
        img = center_crop(img, (128, 128))
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        gt_frames.append(img)
    gt_frames = np.array(gt_frames)

    results = compute_physics_metrics(gt_frames)
    print(f"GT ep{i}: "
          f"angle_jump_avg={results['angle_jump'].mean():.2f}deg  "
          f"max_angle_jump={results['angle_jump'].max():.1f}deg  "
          f"anomaly_rate={results['anomaly_rate']:.1%}  "
          f"anomaly_frames={len(results['anomaly_frames'])}")