# my_project/latent_smoother.py
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
from physics_metrics import extract_state, compute_physics_metrics


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


def run_latent_smoother(ckpt_path, episode_path, output_path,
                         angle_thresh=25.0,
                         obs_key="camera_1_color", resolution=128):
    """
    Latent Smoother (RTS-Smoother style):

    Step 1 - Forward Pass: run a full open-loop rollout and save every latent and predicted frame.
    Step 2 - Anomaly Detection: flag every frame with anomalous angle_jump.
    Step 3 - Backward Smoothing: for each anomalous frame, re-decode from an
        interpolation of its neighbors' latents.

    Difference from online methods: uses global information (future frames)
    to correct past anomalies. Course analogue: the RTS Smoother — the
    forward pass produces predictions, the backward pass corrects them
    using posterior information.
    """
    print("Loading model...")
    model = load_model(ckpt_path)
    normalizer = model.normalizer
    device = model.device
    dtype = model.dtype

    print(f"Loading episode from {episode_path}")
    epi_data, _ = load_dict_from_hdf5(episode_path)
    actions = epi_data["action"][()]
    gt_images = epi_data["obs"]["images"][obs_key][()]
    T = len(actions)

    # Initialize
    raw_img = gt_images[0]
    raw_img_resized = center_crop(raw_img, (resolution, resolution))
    raw_img_resized = cv2.resize(
        raw_img_resized, (resolution, resolution), interpolation=cv2.INTER_AREA)
    img = raw_img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img_tensor = normalizer[obs_key].normalize(img_tensor).to(device)
    with torch.no_grad():
        curr_latent = model.encoder_forward(img_tensor)[:, None]

    curr_action = torch.from_numpy(actions[0]).to(device).float()
    curr_action = normalizer["action"].normalize(curr_action)

    # ==========================================
    # Step 1: Forward Pass - store every latent
    # ==========================================
    print("Step 1: Forward pass - collecting all latents...")
    all_latents = [curr_latent.clone()]  # keep the latent at every step
    all_pred_frames = []
    action_hist = []
    action_ls = []
    act_horizon = 1
    hist_context = 10
    time_idx = 1

    for step in range(T - 1):
        if time_idx >= T:
            break

        next_action = actions[time_idx]
        next_action_norm = normalizer["action"].normalize(
            torch.from_numpy(next_action)
        ).to(device).float()
        curr_action = next_action_norm
        time_idx += 1
        curr_action = torch.clamp(curr_action, -1.0, 1.0)
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

            xs_pred = render_img_cm(
                model, curr_latent[:, -1], resolution,
                normalizer=normalizer, num_views=1,
            )
            pred_np = xs_pred.permute(0, 2, 3, 1).detach().cpu().float().numpy()[0]
            pred_np = (pred_np * 255).astype(np.uint8)
            pred_np = np.clip(pred_np, 0, 255)
            all_pred_frames.append(pred_np)

        if step % 50 == 0:
            print(f"  step {step}/{T-1}")

    print(f"Forward pass done. {len(all_pred_frames)} frames, {len(all_latents)} latents.")

    # ==========================================
    # Step 2: Anomaly Detection
    # ==========================================
    print("Step 2: Detecting anomaly frames...")
    angles = []
    for frame in all_pred_frames:
        state = extract_state(frame)
        angles.append(state['block_angle'])

    anomaly_frames = []
    for i in range(1, len(angles)):
        da = abs(angles[i] - angles[i-1])
        da = min(da, 90 - da) if da > 45 else da
        if da > angle_thresh:
            anomaly_frames.append(i)

    print(f"Found {len(anomaly_frames)} anomaly frames: {anomaly_frames[:10]}...")

    # ==========================================
    # Step 3: Backward Smoothing
    # ==========================================
    print("Step 3: Backward smoothing anomaly frames...")
    smoothed_frames = list(all_pred_frames)  # copy

    for t in anomaly_frames:
        # Find the nearest non-anomalous frames before and after
        prev_good = t - 1
        while prev_good in anomaly_frames and prev_good > 0:
            prev_good -= 1

        next_good = t + 1
        while next_good in anomaly_frames and next_good < len(all_latents) - 1:
            next_good += 1

        if prev_good < 0 or next_good >= len(all_latents):
            continue

        # Interpolate in latent space
        # Weights: linear interpolation by distance
        total = next_good - prev_good
        w_next = (t - prev_good) / total
        w_prev = (next_good - t) / total

        if prev_good < len(all_latents) and next_good < len(all_latents):
            smoothed_latent = (
                w_prev * all_latents[prev_good] +
                w_next * all_latents[next_good]
            )

            # Re-decode from the interpolated latent
            with torch.no_grad():
                xs_smoothed = render_img_cm(
                    model, smoothed_latent[:, -1], resolution,
                    normalizer=normalizer, num_views=1,
                )
            smoothed_np = xs_smoothed.permute(
                0, 2, 3, 1).detach().cpu().float().numpy()[0]
            smoothed_np = (smoothed_np * 255).astype(np.uint8)
            smoothed_np = np.clip(smoothed_np, 0, 255)
            smoothed_frames[t] = smoothed_np
            print(f"  Smoothed frame {t} "
                  f"(prev={prev_good}, next={next_good}, "
                  f"w_prev={w_prev:.2f}, w_next={w_next:.2f})")

    # ==========================================
    # Save video
    # ==========================================
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_size = 256
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 15, (out_size * 2, out_size))

    for i, pred in enumerate(smoothed_frames):
        gt_idx = min(i + 1, len(gt_images) - 1)
        gt = gt_images[gt_idx]
        gt = center_crop(gt, (resolution, resolution))
        gt = cv2.resize(gt, (out_size, out_size), interpolation=cv2.INTER_AREA)
        pred_vis = cv2.resize(pred, (out_size, out_size), interpolation=cv2.INTER_AREA)

        gt_bgr = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        pred_bgr = cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR)

        if i in anomaly_frames:
            cv2.line(pred_bgr, (0, 0), (0, out_size), (0, 255, 255), 3)

        cv2.putText(pred_bgr, "Latent Smoother", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(gt_bgr, "Ground Truth", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        combined = np.concatenate([pred_bgr, gt_bgr], axis=1)
        writer.write(combined)

    writer.release()
    pred_array = np.stack(smoothed_frames)
    np.save(output_path.replace(".mp4", "_pred.npy"), pred_array)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/pusht_cam1/checkpoints/best.ckpt")
    parser.add_argument("--episode", default="data/mini/pusht/val/episode_2.hdf5")
    parser.add_argument("--output",
                        default="my_project/output/latent_smoother_ep2.mp4")
    parser.add_argument("--thresh", type=float, default=25.0)
    parser.add_argument("--obs_key", default="camera_1_color")
    args = parser.parse_args()

    run_latent_smoother(
        ckpt_path=args.ckpt,
        episode_path=args.episode,
        output_path=args.output,
        angle_thresh=args.thresh,
    )