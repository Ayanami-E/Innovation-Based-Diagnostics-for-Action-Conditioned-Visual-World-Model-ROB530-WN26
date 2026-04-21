# my_project/check_solidity.py
import numpy as np
import cv2
import sys
sys.path.insert(0, 'my_project')
from physics_metrics import extract_state

def block_shape_score(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    kernel = np.ones((5,5), np.uint8)
    lower_block = np.array([130, 40, 80])
    upper_block = np.array([175, 255, 255])
    block_mask = cv2.inRange(hsv, lower_block, upper_block)
    block_mask = cv2.morphologyEx(block_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        block_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0
    largest = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest)
    if contour_area < 50:
        return 0.0, 0.0
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = contour_area / (hull_area + 1e-6)
    return solidity, contour_area

pred_frames = np.load("my_project/output/openloop_ep2_pred.npy")

print("Frame | Solidity | Area")
for i in range(0, len(pred_frames), 5):
    s, a = block_shape_score(pred_frames[i])
    print(f"  {i:3d}  |  {s:.3f}   | {a:.0f}")