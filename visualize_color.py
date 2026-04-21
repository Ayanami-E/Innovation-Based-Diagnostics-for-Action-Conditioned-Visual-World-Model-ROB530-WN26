# my_project/visualize_colors.py
import numpy as np
import cv2
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5
from yixuan_utilities.draw_utils import center_crop

# Load one predicted frame and one ground-truth frame
pred_frames = np.load("my_project/output/openloop_ep0_pred.npy")
epi_data, _ = load_dict_from_hdf5("data/mini/pusht/val/episode_0.hdf5")
gt_images = epi_data["obs"]["images"]["camera_1_color"][()]

# Take frame 50 (a mid-rollout frame, more representative)
pred = pred_frames[50]
gt = gt_images[51]
gt = center_crop(gt, (128, 128))
gt = cv2.resize(gt, (128, 128), interpolation=cv2.INTER_AREA)

# Save both frames so colors can be inspected visually
cv2.imwrite("my_project/output/sample_pred.png",
            cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
cv2.imwrite("my_project/output/sample_gt.png",
            cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))

# Print the most frequent colors in the image (HSV space)
def top_colors(img, n=10):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, w, _ = hsv.shape
    pixels = hsv.reshape(-1, 3)
    # Quantize the H channel
    h_vals = pixels[:, 0]
    unique, counts = np.unique(h_vals, return_counts=True)
    idx = np.argsort(-counts)[:n]
    print("Top H values (0-179):", unique[idx], "counts:", counts[idx])

print("=== Predicted frame colors ===")
top_colors(pred)
print("=== GT frame colors ===")
top_colors(gt)