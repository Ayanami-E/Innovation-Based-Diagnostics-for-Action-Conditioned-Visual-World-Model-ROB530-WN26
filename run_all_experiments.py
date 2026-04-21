# my_project/run_all_experiments.py
"""
Full experiment script: runs every method on ep3, evaluates them with a
unified flow-consistency metric, and prints a comparison table.

Methods:
1. Open-loop: pure open-loop rollout, no correction.
2. Periodic Reset (K=10, 20): reset the latent from the GT frame every K steps.
3. Selective Correction: reset from the GT frame whenever flow consistency
   drops below a threshold.
4. Best-of-N (N=3, 5): sample N times and keep the best candidate, but only
   when flow consistency is low.
5. Latent Smoother: run the full rollout first, then interpolate latents for
   low-consistency frames.
"""

from pathlib import Path
import cv2
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from yixuan_utilities.draw_utils import center_crop
from yixuan_utilities.hdf5_utils import load_dict_from_hdf5
from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm
from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import LatentWorldModel
import lightning.pytorch as pl
import sys
sys.path.insert(0, 'my_project')

# ============================================================
# Utility functions
# ============================================================

def load_model(ckpt_path):
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x), replace=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg_path = Path(ckpt_path).parent.parent / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    dtype = torch.float32 if "dtype" not in cfg.algorithm else cfg.algorithm.dtype
    cfg.n_frames = 10
    cfg.algorithm.n_frames = 10
    if "diffusion" in cfg.algorithm and "sampling_timesteps" in cfg.algorithm.diffusion:
        cfg.algorithm.diffusion.sampling_timesteps = 10
    if ("diffusion" in cfg.algorithm.dynamics and
            "sampling_timesteps" in cfg.algorithm.dynamics.diffusion):
        cfg.algorithm.dynamics.diffusion.sampling_timesteps = 10
    cfg.algorithm.load_ae = None
    algo = LatentWorldModel.load_from_checkpoint(
        ckpt_path, cfg=cfg.algorithm, map_location="cuda:0",
        dtype=dtype, strict=False, weights_only=False,
    )
    algo.dynamics = algo.dynamics.to(dtype)
    algo.eval()
    algo.dynamics.eval()
    return algo


def encode_frame(model, frame_rgb, normalizer, obs_key, resolution, device):
    img = center_crop(frame_rgb, (resolution, resolution))
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img_tensor = normalizer[obs_key].normalize(img_tensor).to(device)
    with torch.no_grad():
        latent = model.encoder_forward(img_tensor)[:, None]
    return latent


def decode_latent(model, latent, resolution, normalizer):
    xs = render_img_cm(model, latent[:, -1], resolution,
                       normalizer=normalizer, num_views=1)
    pred_np = xs.permute(0, 2, 3, 1).detach().cpu().float().numpy()[0]
    pred_np = (pred_np * 255).astype(np.uint8)
    return np.clip(pred_np, 0, 255)


def flow_consistency_score(prev_frame, curr_frame):
    """Compute flow consistency of curr_frame vs prev_frame inside the T-block region."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    # T-block mask
    hsv = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([130, 40, 80]), np.array([175, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    if np.sum(mask > 0) < 50:
        return 1.0

    fx = flow[..., 0][mask > 0]
    fy = flow[..., 1][mask > 0]
    mean_fx, mean_fy = fx.mean(), fy.mean()
    mean_mag = np.sqrt(mean_fx**2 + mean_fy**2)

    if mean_mag < 0.1:
        return 1.0

    mags = np.sqrt(fx**2 + fy**2)
    valid = mags > 0.1
    if valid.sum() < 10:
        return 1.0

    cos_sim = (fx[valid] * mean_fx + fy[valid] * mean_fy) / (
        mags[valid] * mean_mag + 1e-8)
    return float(cos_sim.mean())


def compute_all_consistency(frames):
    scores = [1.0]
    for i in range(1, len(frames)):
        scores.append(flow_consistency_score(frames[i-1], frames[i]))
    return np.array(scores)


def save_video(frames, gt_images, output_path, label, resolution=128):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_size = 256
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 15, (out_size * 2, out_size))
    for i, pred in enumerate(frames):
        gt_idx = min(i + 1, len(gt_images) - 1)
        gt = gt_images[gt_idx]
        gt = center_crop(gt, (resolution, resolution))
        gt = cv2.resize(gt, (out_size, out_size), interpolation=cv2.INTER_AREA)
        pred_vis = cv2.resize(pred, (out_size, out_size), interpolation=cv2.INTER_AREA)
        gt_bgr = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        pred_bgr = cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR)
        cv2.putText(pred_bgr, label, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(gt_bgr, "Ground Truth", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        writer.write(np.concatenate([pred_bgr, gt_bgr], axis=1))
    writer.release()


# ============================================================
# Method implementations
# ============================================================

def rollout_base(model, gt_images, actions, normalizer, device, dtype,
                 obs_key, resolution):
    """Baseline rollout; returns pred_frames and all_latents."""
    curr_latent = encode_frame(
        model, gt_images[0], normalizer, obs_key, resolution, device)
    curr_action = torch.from_numpy(actions[0]).to(device).float()
    curr_action = normalizer["action"].normalize(curr_action)

    pred_frames = []
    all_latents = [curr_latent.clone()]
    action_hist, action_ls = [], []
    act_horizon, hist_context = 1, 10
    time_idx = 1
    T = len(actions)

    for step in range(T - 1):
        if time_idx >= T:
            break
        next_action = actions[time_idx]
        next_action_norm = normalizer["action"].normalize(
            torch.from_numpy(next_action)).to(device).float()
        curr_action = torch.clamp(next_action_norm, -1.0, 1.0)
        time_idx += 1
        action_ls.append(curr_action)

        if len(action_ls) == act_horizon:
            action_chunk = torch.stack(action_ls).reshape(1, -1)
            action_hist.append(action_chunk)
            action = torch.cat(action_hist, dim=0)[-(hist_context + 1):]
            action = rearrange(action, "t a -> 1 t a").to(device=device, dtype=dtype)
            with torch.no_grad():
                latent_pred = model.dynamics_forward(curr_latent, action)
            curr_latent = torch.cat([curr_latent, latent_pred], dim=1)
            curr_latent = curr_latent[:, -hist_context:]
            action_ls = []
            all_latents.append(curr_latent.clone())
            pred_frames.append(decode_latent(model, curr_latent, resolution, normalizer))

    return pred_frames, all_latents


def method_openloop(model, gt_images, actions, normalizer,
                    device, dtype, obs_key, resolution):
    print("  Running: Open-loop...")
    frames, _ = rollout_base(model, gt_images, actions, normalizer,
                             device, dtype, obs_key, resolution)
    return frames


def method_periodic_reset(model, gt_images, actions, normalizer,
                           device, dtype, obs_key, resolution, K):
    print(f"  Running: Periodic Reset K={K}...")
    curr_latent = encode_frame(
        model, gt_images[0], normalizer, obs_key, resolution, device)
    curr_action = torch.from_numpy(actions[0]).to(device).float()
    curr_action = normalizer["action"].normalize(curr_action)

    pred_frames = []
    action_hist, action_ls = [], []
    act_horizon, hist_context = 1, 10
    time_idx = 1
    T = len(actions)

    for step in range(T - 1):
        if time_idx >= T:
            break
        next_action = actions[time_idx]
        next_action_norm = normalizer["action"].normalize(
            torch.from_numpy(next_action)).to(device).float()
        curr_action = torch.clamp(next_action_norm, -1.0, 1.0)
        time_idx += 1
        action_ls.append(curr_action)

        if len(action_ls) == act_horizon:
            action_chunk = torch.stack(action_ls).reshape(1, -1)
            action_hist.append(action_chunk)
            action = torch.cat(action_hist, dim=0)[-(hist_context + 1):]
            action = rearrange(action, "t a -> 1 t a").to(device=device, dtype=dtype)
            with torch.no_grad():
                latent_pred = model.dynamics_forward(curr_latent, action)
            curr_latent = torch.cat([curr_latent, latent_pred], dim=1)
            curr_latent = curr_latent[:, -hist_context:]
            action_ls = []
            pred_frames.append(decode_latent(model, curr_latent, resolution, normalizer))

            if (step + 1) % K == 0 and (step + 1) < len(gt_images):
                curr_latent = encode_frame(
                    model, gt_images[step + 1], normalizer, obs_key, resolution, device)
                action_hist = []

    return pred_frames


def method_selective_correction(model, gt_images, actions, normalizer,
                                 device, dtype, obs_key, resolution, threshold):
    """Reset the latent from the GT frame whenever flow consistency is below threshold."""
    print(f"  Running: Selective Correction (thresh={threshold:.3f})...")
    curr_latent = encode_frame(
        model, gt_images[0], normalizer, obs_key, resolution, device)

    pred_frames = []
    action_hist, action_ls = [], []
    act_horizon, hist_context = 1, 10
    time_idx = 1
    T = len(actions)
    n_corrections = 0
    prev_frame = None
    stable_latent = curr_latent.clone()
    stable_action_hist = []

    for step in range(T - 1):
        if time_idx >= T:
            break
        next_action = actions[time_idx]
        next_action_norm = normalizer["action"].normalize(
            torch.from_numpy(next_action)).to(device).float()
        curr_action = torch.clamp(next_action_norm, -1.0, 1.0)
        time_idx += 1
        action_ls.append(curr_action)

        if len(action_ls) == act_horizon:
            action_chunk = torch.stack(action_ls).reshape(1, -1)
            action_hist.append(action_chunk)
            action = torch.cat(action_hist, dim=0)[-(hist_context + 1):]
            action = rearrange(action, "t a -> 1 t a").to(device=device, dtype=dtype)
            with torch.no_grad():
                latent_pred = model.dynamics_forward(curr_latent, action)
            curr_latent_cand = torch.cat([curr_latent, latent_pred], dim=1)
            curr_latent_cand = curr_latent_cand[:, -hist_context:]
            action_ls = []

            pred_np = decode_latent(model, curr_latent_cand, resolution, normalizer)

            # Check flow consistency
            if prev_frame is not None:
                score = flow_consistency_score(prev_frame, pred_np)
            else:
                score = 1.0

            if score < threshold and (step + 1) < len(gt_images):
                # Instability: reset from the GT frame
                curr_latent = encode_frame(
                    model, gt_images[step + 1], normalizer, obs_key, resolution, device)
                action_hist = []
                n_corrections += 1
                pred_np = decode_latent(model, curr_latent, resolution, normalizer)
            else:
                curr_latent = curr_latent_cand
                stable_latent = curr_latent.clone()
                stable_action_hist = list(action_hist)

            pred_frames.append(pred_np)
            prev_frame = pred_np

    print(f"    Corrections: {n_corrections}")
    return pred_frames


def method_best_of_n(model, gt_images, actions, normalizer,
                     device, dtype, obs_key, resolution, N, threshold):
    """Sample N times and pick the best only when flow consistency is low; otherwise sample once."""
    print(f"  Running: Best-of-N (N={N}, thresh={threshold:.3f})...")
    curr_latent = encode_frame(
        model, gt_images[0], normalizer, obs_key, resolution, device)

    pred_frames = []
    action_hist, action_ls = [], []
    act_horizon, hist_context = 1, 10
    time_idx = 1
    T = len(actions)
    prev_frame = None
    n_triggered = 0

    for step in range(T - 1):
        if time_idx >= T:
            break
        next_action = actions[time_idx]
        next_action_norm = normalizer["action"].normalize(
            torch.from_numpy(next_action)).to(device).float()
        curr_action = torch.clamp(next_action_norm, -1.0, 1.0)
        time_idx += 1
        action_ls.append(curr_action)

        if len(action_ls) == act_horizon:
            action_chunk = torch.stack(action_ls).reshape(1, -1)
            action_hist.append(action_chunk)
            action = torch.cat(action_hist, dim=0)[-(hist_context + 1):]
            action = rearrange(action, "t a -> 1 t a").to(device=device, dtype=dtype)
            action_ls = []

            # Run once to decide whether Best-of-N is needed
            with torch.no_grad():
                latent_pred = model.dynamics_forward(curr_latent, action)
            cand_latent = torch.cat([curr_latent, latent_pred], dim=1)
            cand_latent = cand_latent[:, -hist_context:]
            pred_np = decode_latent(model, cand_latent, resolution, normalizer)

            if prev_frame is not None:
                score = flow_consistency_score(prev_frame, pred_np)
            else:
                score = 1.0

            # Trigger Best-of-N only if instability was detected
            if score < threshold:
                n_triggered += 1
                best_score = score
                best_latent = cand_latent
                best_pred = pred_np

                for _ in range(N - 1):
                    with torch.no_grad():
                        latent_pred = model.dynamics_forward(curr_latent, action)
                    cand = torch.cat([curr_latent, latent_pred], dim=1)
                    cand = cand[:, -hist_context:]
                    p = decode_latent(model, cand, resolution, normalizer)
                    s = flow_consistency_score(prev_frame, p)
                    if s > best_score:
                        best_score = s
                        best_latent = cand
                        best_pred = p

                curr_latent = best_latent
                pred_frames.append(best_pred)
                prev_frame = best_pred
            else:
                curr_latent = cand_latent
                pred_frames.append(pred_np)
                prev_frame = pred_np

    print(f"    Best-of-N triggered: {n_triggered} times")
    return pred_frames


def method_latent_smoother(model, gt_images, actions, normalizer,
                            device, dtype, obs_key, resolution, threshold):
    """Run the full rollout, save all latents, then interpolate latents on low-consistency frames."""
    print(f"  Running: Latent Smoother (thresh={threshold:.3f})...")

    # Step 1: Forward pass
    pred_frames, all_latents = rollout_base(
        model, gt_images, actions, normalizer, device, dtype, obs_key, resolution)

    # Step 2: detect anomalous frames
    consistency = compute_all_consistency(pred_frames)
    anomaly_frames = set(np.where(consistency < threshold)[0])
    print(f"    Anomaly frames detected: {len(anomaly_frames)}")

    # Step 3: re-decode anomalous frames from an interpolation of neighbors' latents
    smoothed_frames = list(pred_frames)
    for t in sorted(anomaly_frames):
        # Find the nearest non-anomalous frames before and after
        prev_good = t - 1
        while prev_good in anomaly_frames and prev_good > 0:
            prev_good -= 1
        next_good = t + 1
        while next_good in anomaly_frames and next_good < len(all_latents) - 1:
            next_good += 1

        if prev_good < 0 or next_good >= len(all_latents):
            continue

        total = next_good - prev_good
        w_next = (t - prev_good) / total
        w_prev = 1.0 - w_next

        smoothed_latent = w_prev * all_latents[prev_good] + w_next * all_latents[next_good]
        smoothed_frames[t] = decode_latent(model, smoothed_latent, resolution, normalizer)

    return smoothed_frames


# ============================================================
# Main program
# ============================================================

def main():
    import matplotlib.pyplot as plt

    CKPT = "outputs/pusht_cam1/checkpoints/best.ckpt"
    EPISODE = "data/mini/pusht/val/episode_3.hdf5"
    OBS_KEY = "camera_1_color"
    RESOLUTION = 128
    OUTPUT_DIR = Path("my_project/output/ep3_results")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_model(CKPT)
    normalizer = model.normalizer
    device = model.device
    dtype = model.dtype

    print("Loading episode...")
    epi_data, _ = load_dict_from_hdf5(EPISODE)
    actions = epi_data["action"][()]
    gt_images = epi_data["obs"]["images"][OBS_KEY][()]

    # Compute GT flow consistency to set the threshold
    print("Computing GT flow consistency threshold...")
    gt_frames = []
    for img in gt_images:
        img = center_crop(img, (RESOLUTION, RESOLUTION))
        img = cv2.resize(img, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)
        gt_frames.append(img)
    gt_frames = np.array(gt_frames)
    gt_consistency = compute_all_consistency(gt_frames)
    THRESHOLD = gt_consistency.mean() - 2 * gt_consistency.std()
    print(f"Threshold: {THRESHOLD:.3f} (GT mean={gt_consistency.mean():.3f}, std={gt_consistency.std():.3f})")

    # Define all methods
    print("\nRunning all methods...")
    results = {}

    results["Open-loop"] = method_openloop(
        model, gt_images, actions, normalizer, device, dtype, OBS_KEY, RESOLUTION)

    results["Periodic Reset K=10"] = method_periodic_reset(
        model, gt_images, actions, normalizer, device, dtype, OBS_KEY, RESOLUTION, K=10)

    results["Periodic Reset K=20"] = method_periodic_reset(
        model, gt_images, actions, normalizer, device, dtype, OBS_KEY, RESOLUTION, K=20)

    results["Selective Correction"] = method_selective_correction(
        model, gt_images, actions, normalizer, device, dtype, OBS_KEY, RESOLUTION,
        threshold=THRESHOLD)

    results["Best-of-3"] = method_best_of_n(
        model, gt_images, actions, normalizer, device, dtype, OBS_KEY, RESOLUTION,
        N=3, threshold=THRESHOLD)

    results["Best-of-5"] = method_best_of_n(
        model, gt_images, actions, normalizer, device, dtype, OBS_KEY, RESOLUTION,
        N=5, threshold=THRESHOLD)

    results["Latent Smoother"] = method_latent_smoother(
        model, gt_images, actions, normalizer, device, dtype, OBS_KEY, RESOLUTION,
        threshold=THRESHOLD)

    # Save video
    print("\nSaving videos...")
    for name, frames in results.items():
        fname = name.lower().replace(" ", "_").replace("=", "")
        save_video(frames, gt_images, str(OUTPUT_DIR / f"{fname}.mp4"),
                   label=name, resolution=RESOLUTION)
        np.save(str(OUTPUT_DIR / f"{fname}_pred.npy"), np.array(frames))

    # Unified evaluation
    print("\nEvaluating all methods...")
    gt_anomaly_rate = (gt_consistency < THRESHOLD).mean()

    table_data = []
    for name, frames in results.items():
        frames_arr = np.array(frames)
        c = compute_all_consistency(frames_arr)
        anomaly_rate = (c < THRESHOLD).mean()
        mean_c = c.mean()
        min_c = c.min()
        table_data.append({
            "Method": name,
            "Anomaly Rate": anomaly_rate,
            "Mean Consistency": mean_c,
            "Min Consistency": min_c,
        })

    # Print table
    print("\n" + "="*70)
    print(f"{'Method':<25} {'Anomaly Rate':>14} {'Mean Consist.':>14} {'Min Consist.':>13}")
    print("-"*70)
    print(f"{'GT (reference)':<25} {gt_anomaly_rate:>13.1%} {gt_consistency.mean():>14.3f} {gt_consistency.min():>13.3f}")
    print("-"*70)
    for row in table_data:
        print(f"{row['Method']:<25} {row['Anomaly Rate']:>13.1%} "
              f"{row['Mean Consistency']:>14.3f} {row['Min Consistency']:>13.3f}")
    print("="*70)

    # Plot consistency curves for each method
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {
        "Open-loop": "red",
        "Periodic Reset K=10": "orange",
        "Periodic Reset K=20": "darkorange",
        "Selective Correction": "green",
        "Best-of-3": "blue",
        "Best-of-5": "darkblue",
        "Latent Smoother": "purple",
    }
    ax.plot(gt_consistency, label="GT", color="black", linewidth=2, alpha=0.5)
    for name, frames in results.items():
        c = compute_all_consistency(np.array(frames))
        ax.plot(c, label=name, color=colors.get(name, "gray"), alpha=0.7)
    ax.axhline(y=THRESHOLD, color="gray", linestyle="--", label="Anomaly threshold")
    ax.set_title("Flow Consistency over Time - Episode 3")
    ax.set_ylabel("Cosine consistency")
    ax.set_xlabel("Timestep")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "consistency_comparison.png"), dpi=150)
    print(f"\nPlot saved to {OUTPUT_DIR}/consistency_comparison.png")
    print(f"Videos saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()