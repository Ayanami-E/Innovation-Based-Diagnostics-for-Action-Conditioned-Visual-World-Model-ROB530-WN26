# my_project/periodic_reset.py
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


def load_model(ckpt_path: str) -> pl.LightningModule:
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
    """Encode a single frame into a latent."""
    img = center_crop(frame_rgb, (resolution, resolution))
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img_tensor = normalizer[obs_key].normalize(img_tensor).to(device)
    with torch.no_grad():
        latent = model.encoder_forward(img_tensor)[:, None]
    return latent


def run_periodic_reset(ckpt_path, episode_path, output_path,
                       K=20, obs_key="camera_1_color", resolution=128):
    """
    Periodic Reset: re-encode the latent from the ground-truth frame every K steps.
    Smaller K means more frequent correction.
    """
    print(f"Loading model...")
    model = load_model(ckpt_path)
    normalizer = model.normalizer
    device = model.device
    dtype = model.dtype

    print(f"Loading episode from {episode_path}")
    epi_data, _ = load_dict_from_hdf5(episode_path)
    actions = epi_data["action"][()]
    gt_images = epi_data["obs"]["images"][obs_key][()]
    T = len(actions)
    print(f"Episode length: {T}, K={K}")

    # Initialize
    curr_latent = encode_frame(
        model, gt_images[0], normalizer, obs_key, resolution, device)
    curr_action = torch.from_numpy(actions[0]).to(device).float()
    curr_action = normalizer["action"].normalize(curr_action)

    pred_frames = []
    action_hist = []
    action_ls = []
    act_horizon = 1
    hist_context = 10
    time_idx = 1
    correction_times = []

    print("Running periodic reset rollout...")
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

            # Decode the predicted frame
            xs_pred = render_img_cm(
                model, curr_latent[:, -1], resolution,
                normalizer=normalizer, num_views=1,
            )
            pred_np = xs_pred.permute(0, 2, 3, 1).detach().cpu().float().numpy()[0]
            pred_np = (pred_np * 255).astype(np.uint8)
            pred_np = np.clip(pred_np, 0, 255)
            pred_frames.append(pred_np)

            # Periodic Reset: every K steps, re-encode the latent from the GT frame
            if (step + 1) % K == 0 and (step + 1) < len(gt_images):
                curr_latent = encode_frame(
                    model, gt_images[step + 1],
                    normalizer, obs_key, resolution, device
                )
                # Reset action_hist so past actions don't contaminate the new latent
                action_hist = []
                correction_times.append(step)
                print(f"  Reset at step {step}")

        if step % 50 == 0:
            print(f"  step {step}/{T-1}")

    print(f"Done. {len(pred_frames)} frames, {len(correction_times)} resets.")

    # Save video and numpy
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_size = 256
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 15, (out_size * 2, out_size))

    for i, pred in enumerate(pred_frames):
        gt_idx = min(i + 1, len(gt_images) - 1)
        gt = gt_images[gt_idx]
        gt = center_crop(gt, (resolution, resolution))
        gt = cv2.resize(gt, (out_size, out_size), interpolation=cv2.INTER_AREA)
        pred_vis = cv2.resize(pred, (out_size, out_size), interpolation=cv2.INTER_AREA)

        gt_bgr = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
        pred_bgr = cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR)

        # Mark reset frames with a green vertical line
        if i in correction_times:
            cv2.line(pred_bgr, (0, 0), (0, out_size), (0, 255, 0), 3)

        cv2.putText(pred_bgr, f"PR K={K}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(gt_bgr, "Ground Truth", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        combined = np.concatenate([pred_bgr, gt_bgr], axis=1)
        writer.write(combined)

    writer.release()
    pred_array = np.stack(pred_frames)
    np.save(output_path.replace(".mp4", "_pred.npy"), pred_array)
    print(f"Saved to {output_path}")
    return correction_times


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/pusht_cam1/checkpoints/best.ckpt")
    parser.add_argument("--episode", default="data/mini/pusht/val/episode_2.hdf5")
    parser.add_argument("--output", default="my_project/output/periodic_reset_ep2_K20.mp4")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--obs_key", default="camera_1_color")
    args = parser.parse_args()

    run_periodic_reset(
        ckpt_path=args.ckpt,
        episode_path=args.episode,
        output_path=args.output,
        K=args.K,
    )