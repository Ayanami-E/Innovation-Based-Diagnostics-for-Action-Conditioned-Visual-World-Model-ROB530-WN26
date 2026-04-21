"""
Headless open-loop rollout script for IWS on PushT.
No display required. Saves predicted frames and GT frames as side-by-side video.
"""
import math
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
from interactive_world_sim.utils.normalizer import LinearNormalizer
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


def run_rollout(ckpt_path: str, episode_path: str, output_path: str,
                obs_key: str = "camera_1_color", resolution: int = 128):

    print(f"Loading model from {ckpt_path}")
    model = load_model(ckpt_path)
    normalizer = model.normalizer
    device = model.device
    dtype = model.dtype

    print(f"Loading episode from {episode_path}")
    epi_data, _ = load_dict_from_hdf5(episode_path)

    # Print all keys to confirm the data structure
    def print_keys(d, prefix=''):
        for k, v in d.items():
            if isinstance(v, dict):
                print_keys(v, prefix + k + '/')
            else:
                shape = v.shape if hasattr(v, 'shape') else type(v)
                print(f'  {prefix}{k}: {shape}')
    print("Episode keys:")
    print_keys(epi_data)

    # Read actions and images
    actions = epi_data["action"][()]
    gt_images = epi_data["obs"]["images"][obs_key][()]
    T = len(actions)
    print(f"Episode length: {T} frames")

    # Initialize: encode the first-frame image into a latent
    t = 0
    raw_img = gt_images[t]
    raw_img = center_crop(raw_img, (resolution, resolution))
    raw_img = cv2.resize(raw_img, (resolution, resolution), interpolation=cv2.INTER_AREA)
    img = raw_img / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img_tensor = normalizer[obs_key].normalize(img_tensor).to(device)

    with torch.no_grad():
        curr_latent = model.encoder_forward(img_tensor)[:, None]

    # Initial action
    curr_action = torch.from_numpy(actions[t]).to(device).float()
    curr_action = normalizer["action"].normalize(curr_action)

    # open-loop rollout
    pred_frames = []
    action_hist = []
    action_ls = []
    act_horizon = 1
    hist_context = 10
    skip_frame = 1
    time_idx = skip_frame

    print("Running open-loop rollout...")
    for step in range(T - 1):
        if time_idx >= T:
            break

        next_action = actions[time_idx]
        next_action_norm = normalizer["action"].normalize(
            torch.from_numpy(next_action)
        ).to(device).float()

        curr_action = next_action_norm
        time_idx += skip_frame
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

            xs_pred = render_img_cm(
                model, curr_latent[:, -1], resolution,
                normalizer=normalizer, num_views=1,
            )
            pred_np = xs_pred.permute(0, 2, 3, 1).detach().cpu().float().numpy()[0]
            pred_np = (pred_np * 255).astype(np.uint8)
            pred_np = np.clip(pred_np, 0, 255)
            pred_frames.append(pred_np)

        if step % 50 == 0:
            print(f"  step {step}/{T-1}")

    print(f"Rollout done. {len(pred_frames)} frames generated.")

    # Save a side-by-side comparison video (left: prediction, right: ground truth)
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
        cv2.putText(pred_bgr, "Predicted", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(gt_bgr, "Ground Truth", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        combined = np.concatenate([pred_bgr, gt_bgr], axis=1)
        writer.write(combined)

    writer.release()
    print(f"Video saved to {output_path}")

    # Save the predicted frames as a numpy array for downstream metrics
    pred_array = np.stack(pred_frames)
    np.save(output_path.replace(".mp4", "_pred.npy"), pred_array)
    print(f"Pred frames saved to {output_path.replace('.mp4', '_pred.npy')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/pusht_cam1/checkpoints/best.ckpt")
    parser.add_argument("--episode", default="data/mini/pusht/val/episode_0.hdf5")
    parser.add_argument("--output", default="my_project/output/openloop_ep0.mp4")
    parser.add_argument("--obs_key", default="camera_1_color")
    parser.add_argument("--resolution", type=int, default=128)
    args = parser.parse_args()

    run_rollout(
        ckpt_path=args.ckpt,
        episode_path=args.episode,
        output_path=args.output,
        obs_key=args.obs_key,
        resolution=args.resolution,
    )