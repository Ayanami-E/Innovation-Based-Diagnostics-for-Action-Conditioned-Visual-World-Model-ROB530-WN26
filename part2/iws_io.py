"""Unified IWS rollout loader.

The PushT IWS data on disk is in two forms:

  - Per-episode `.npy` predicted-frame arrays (199, 128, 128, 3) uint8 RGB
    under `output/openloop_ep{N}_pred.npy` and
    `output/ep3_results/<method>_pred.npy`.

  - Per-episode side-by-side MP4s under `output/openloop_ep{N}.mp4` and
    `output/ep3_results/<method>.mp4`. Each frame is 256x512 with the
    predicted half on the left and the GT half on the right (both 256x256).
    See `headless_rollout.py:139-162` for the writer.

Both halves of the MP4 carry the same mp4v compression. We always read GT
from the MP4 (no other source on disk), and read rollouts from the MP4 as
well so the comparison stays codec-fair. Unit-tests can also pull rollouts
from the cleaner `_pred.npy` for cross-checks.

Frames are decoded once, resized to 256x256 RGB, and cached under
`results/part2_cache/{ep_id}__{source}.npz`. Subsequent loads hit the cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
CACHE_DIR = ROOT / "results" / "part2_cache"

FRAME_SIZE = 256  # MP4 half is 256x256 -- keep native resolution

# Episode -> (rollout_npy, mp4) -- the open-loop set used in §V.A.
OPENLOOP_EPISODES = {
    f"openloop_ep{i}": (
        OUTPUT_DIR / f"openloop_ep{i}_pred.npy",
        OUTPUT_DIR / f"openloop_ep{i}.mp4",
    )
    for i in range(5)
}

# Correction methods (all on episode 3) -- used in §V.C.
EP3_DIR = OUTPUT_DIR / "ep3_results"
CORRECTION_METHODS = {
    "open-loop": (EP3_DIR / "open-loop_pred.npy", EP3_DIR / "open-loop.mp4"),
    "periodic_reset_k10": (
        EP3_DIR / "periodic_reset_k10_pred.npy",
        EP3_DIR / "periodic_reset_k10.mp4",
    ),
    "periodic_reset_k20": (
        EP3_DIR / "periodic_reset_k20_pred.npy",
        EP3_DIR / "periodic_reset_k20.mp4",
    ),
    "latent_smoother": (
        EP3_DIR / "latent_smoother_pred.npy",
        EP3_DIR / "latent_smoother.mp4",
    ),
    "selective_correction": (
        EP3_DIR / "selective_correction_pred.npy",
        EP3_DIR / "selective_correction.mp4",
    ),
    "best-of-3": (EP3_DIR / "best-of-3_pred.npy", EP3_DIR / "best-of-3.mp4"),
    "best-of-5": (EP3_DIR / "best-of-5_pred.npy", EP3_DIR / "best-of-5.mp4"),
}


@dataclass
class Episode:
    episode_id: str
    source: str
    frames: np.ndarray  # (T, H, W, 3) uint8 RGB
    fps: float = 15.0
    task_name: str = "pusht"


def _ensure_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(episode_id: str, source: str) -> Path:
    return CACHE_DIR / f"{episode_id}__{source}.npz"


def _decode_mp4(mp4_path: Path) -> np.ndarray:
    """Return (T, 256, 512, 3) uint8 RGB by decoding every frame in the MP4."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {mp4_path}")
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    if not frames:
        raise RuntimeError(f"empty video {mp4_path}")
    return np.stack(frames, axis=0)


def _split_halves(mp4_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Side-by-side -> (pred_half, gt_half), each (T, 256, 256, 3) RGB."""
    h, w = mp4_frames.shape[1], mp4_frames.shape[2]
    if w != 2 * h:
        raise ValueError(f"unexpected MP4 layout {mp4_frames.shape}; "
                         f"expected width = 2*height")
    pred = mp4_frames[:, :, :h]
    gt = mp4_frames[:, :, h:]
    return pred.copy(), gt.copy()


def load_episode(episode_id: str, source: str,
                 use_cache: bool = True) -> Episode:
    """Load one episode of either GT or predicted frames at 256x256 RGB.

    `episode_id` is one of OPENLOOP_EPISODES.keys() or
    CORRECTION_METHODS.keys(). `source` is "gt", "rollout", or
    "rollout_npy" (the cleaner uncompressed pred frames upsampled to 256).
    """
    _ensure_cache()
    cache = _cache_path(episode_id, source)
    if use_cache and cache.exists():
        d = np.load(cache)
        return Episode(episode_id=episode_id, source=source,
                       frames=d["frames"], fps=float(d["fps"]),
                       task_name=str(d["task_name"]))

    if episode_id in OPENLOOP_EPISODES:
        npy_path, mp4_path = OPENLOOP_EPISODES[episode_id]
    elif episode_id in CORRECTION_METHODS:
        npy_path, mp4_path = CORRECTION_METHODS[episode_id]
    else:
        raise KeyError(f"unknown episode_id {episode_id!r}")

    if source in ("gt", "rollout"):
        if not mp4_path.exists():
            raise FileNotFoundError(f"missing MP4 for {episode_id}: {mp4_path}")
        full = _decode_mp4(mp4_path)
        pred_half, gt_half = _split_halves(full)
        frames = gt_half if source == "gt" else pred_half
    elif source == "rollout_npy":
        if not npy_path.exists():
            raise FileNotFoundError(
                f"missing pred npy for {episode_id}: {npy_path}")
        arr = np.load(npy_path)  # (T, 128, 128, 3) uint8 RGB
        # Upsample to 256 to match the MP4-half resolution. INTER_AREA is the
        # common "de-aliased upsample" choice that headless_rollout used the
        # other direction; for consistent comparison we keep INTER_AREA.
        out = np.empty((arr.shape[0], FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
        for i in range(arr.shape[0]):
            out[i] = cv2.resize(arr[i], (FRAME_SIZE, FRAME_SIZE),
                                interpolation=cv2.INTER_AREA)
        frames = out
    else:
        raise ValueError(f"unknown source {source!r}")

    np.savez_compressed(cache,
                        frames=frames,
                        fps=np.float64(15.0),
                        task_name=np.str_("pusht"))
    return Episode(episode_id=episode_id, source=source,
                   frames=frames, fps=15.0, task_name="pusht")


def iter_episodes(episode_ids, source: str,
                  use_cache: bool = True) -> Iterator[Episode]:
    for ep in episode_ids:
        yield load_episode(ep, source, use_cache=use_cache)


def available_openloop_episodes() -> list[str]:
    return sorted(
        ep for ep, (_, mp4) in OPENLOOP_EPISODES.items() if mp4.exists()
    )


def available_correction_methods() -> list[str]:
    return sorted(
        m for m, (_, mp4) in CORRECTION_METHODS.items() if mp4.exists()
    )


def frame_count(episode_id: str) -> Optional[int]:
    if episode_id in OPENLOOP_EPISODES:
        npy_path, _ = OPENLOOP_EPISODES[episode_id]
    elif episode_id in CORRECTION_METHODS:
        npy_path, _ = CORRECTION_METHODS[episode_id]
    else:
        return None
    if not npy_path.exists():
        return None
    return int(np.load(npy_path, mmap_mode="r").shape[0])
