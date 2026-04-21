# Part 2B — IWS Asset Inventory

Compiled 2026-04-18. Search scope: repo (`E:\let_work_work\rob530\
my_project\my_project`), Windows user home (`C:\Users\Enoch`),
HuggingFace cache (`~/.cache/huggingface`), other drives (`C:`, `D:`,
`E:`), Windows temp. **Nothing was executed.**

---

## Q1 — IWS checkpoint files

**Result: NONE on disk anywhere.**

```
find E:\let_work_work\rob530\my_project\my_project \( -name "*.ckpt"
  -o -name "*.pt" -o -name "*.safetensors" -o -name "*.bin" \)
→ 0 hits
```

`~/.cache/huggingface/hub/` exists and contains:

| Hub model dir                                | IWS-related? |
|----------------------------------------------|:------------:|
| `models--Qwen--Qwen2.5-0.5B`                 | no           |
| `models--Qwen--Qwen2.5-7B`                   | no           |
| `models--Qwen--Qwen3-VL-8B-Instruct`         | no           |
| `models--openai--clip-vit-base-patch32`      | no           |
| `models--openai--clip-vit-large-patch14`     | no           |
| `models--timm--vit_base_patch16_clip_224.openai` | no       |
| `models--yuvalkirstain--PickScore_v1`        | no           |

No `models--yixuan1999--interactive-world-sim-checkpoints` directory.
The IWS checkpoint download script (`interactive_world_sim/scripts/
download_checkpoints.sh`) has not been run on this machine.

**Where checkpoints would land if downloaded:**
`interactive_world_sim/outputs/{task}_cam{N}/checkpoints/best.ckpt`
plus `interactive_world_sim/outputs/{task}_cam{N}/.hydra/config.yaml`,
for the 7 official task/camera combos:

```
outputs/pusht_cam1/
outputs/single_grasp_cam0/
outputs/single_grasp_cam1/
outputs/bimanual_sweep_cam0/
outputs/bimanual_sweep_cam1/
outputs/bimanual_rope_cam0/
outputs/bimanual_rope_cam1/
```

(See `interactive_world_sim/scripts/download_checkpoints.sh` lines
7-14 and the README's "Download Checkpoints" section for the exact
table.)

---

## Q2 — IWS inference / rollout code

**Result: yes, in three flavours.**

| Path                                                                  | Purpose                                                   | Headless? |
|-----------------------------------------------------------------------|-----------------------------------------------------------|:---------:|
| `headless_rollout.py` (repo root)                                     | Open-loop rollout from a single HDF5 episode → MP4 + npy. Hardcoded for PushT, but trivially generalizable via CLI args. **This is what produced every `_pred.npy` / `*.mp4` already on disk.** | **yes** |
| `interactive_world_sim/scripts/inference/teleoperate_keyboard.py`     | Interactive keyboard teleop into the world model. Requires display + key input. | no        |
| `interactive_world_sim/scripts/inference/teleoperate_aloha.py`        | Interactive ALOHA-robot teleop. Requires real robot or sim. | no        |
| `interactive_world_sim/deploy/server.py`                              | Web-demo server. Requires browser session. | no        |
| `interactive_world_sim/interactive_world_sim/algorithms/latent_dynamics/latent_world_model.py` | The `LatentWorldModel` Lightning module itself (not a CLI). | n/a |
| `interactive_world_sim/interactive_world_sim/experiments/exp_latent_dyn.py` | Training experiment, not inference. | n/a |

The IWS repo also ships **launcher shell scripts per task/camera** under
`interactive_world_sim/scripts/inference/{keyboard,aloha}/`:

```
keyboard/                                aloha/
  pusht_kybd.sh                            (no pusht aloha launcher)
  single_grasp_cam0_kybd.sh                single_grasp_cam0_aloha.sh
  single_grasp_cam1_kybd.sh                single_grasp_cam1_aloha.sh
  bimanual_sweep_cam0_kybd.sh              bimanual_sweep_cam0_aloha.sh
  bimanual_sweep_cam1_kybd.sh              bimanual_sweep_cam1_aloha.sh
  bimanual_sweep_cam0_and_cam1_kybd.sh     bimanual_sweep_cam0_and_cam1_aloha.sh
  bimanual_rope_cam0_kybd.sh               bimanual_rope_cam0_aloha.sh
  bimanual_rope_cam1_kybd.sh               bimanual_rope_cam1_aloha.sh
  bimanual_rope_cam0_and_cam1_kybd.sh      bimanual_rope_cam0_and_cam1_aloha.sh
```

All call `teleoperate_keyboard.py` or `teleoperate_aloha.py` and require
either keyboard input or a live ALOHA robot. **There is no upstream
"batch open-loop rollout" launcher** — `headless_rollout.py` is the only
non-interactive driver, and it was authored locally, not by IWS.

---

## Q3 — Other-task GT or rollout data

**Result: NONE. Every cached frame on disk is PushT.**

### MP4 inventory (full repo scan)

| Path                                              | Task   | Notes |
|---------------------------------------------------|--------|-------|
| `output/openloop_ep{0..4}.mp4`                    | PushT  | open-loop, paired pred/GT halves |
| `output/latent_smoother_ep{2,3}.mp4`              | PushT  | correction, ep2/ep3 |
| `output/periodic_reset_ep2_K20.mp4`               | PushT  | correction, ep2 |
| `output/selective_correction_ep2*.mp4` (×7 variants) | PushT | correction, ep2 |
| `output/selective_correction_v{2,3}_ep{2,3}.mp4`  | PushT  | correction |
| `output/ep3_results/{open-loop, periodic_reset_k10, periodic_reset_k20, latent_smoother, selective_correction, best-of-3, best-of-5}.mp4` | PushT | the canonical 5+2 corrections on ep3 |
| `output/part1/part1_demo.mp4`                     | (Part 1 MuJoCo) | not IWS |

### NPY/NPZ inventory (excluding part1/part2 caches and venv)

```
output/openloop_ep{0..4}_pred.npy             # PushT open-loop preds (5 eps)
output/ep3_results/<method>_pred.npy           # 7 IWS PushT ep3 method preds
output/latent_smoother_ep{2,3}_pred.npy        # PushT correction preds
output/periodic_reset_ep2_K20_pred.npy
output/selective_correction_ep2*.{npy,_innovation.npy}
output/selective_correction_v{2,3}_ep{2,3}_pred.npy

interactive_world_sim/.../aloha_extrinsics/{left,right}_base_pose_in_world.npy
                                              # extrinsics, not rollout data
```

### HDF5 / Zarr / per-task PNG sequences

```
*.hdf5 / *.h5 / *.zarr → 0 hits (anywhere)
```

`output/anomaly_check/`, `flow_check/`, `area_anomaly/`, `consistency_check/`
are PNG dumps from previous Part 3 metric runs on the **same PushT
rollouts** above — not new task data.

### Other drives + cache

- `D:\` (game / steam / WSL / conda3) — no IWS data.
- `~/.cache/huggingface/datasets/` — only `emozilla___pg19`. **No
  `yixuan1999/interactive-world-sim-min-data`.**
- `/tmp`, `C:\Users\Enoch\AppData\Local\Temp` — no IWS files.
- `~/data`, `~/scratch`, `/c/temp` — do not exist.

**Nothing exists for `single_grasp`, `bimanual_sweep`, `bimanual_rope`,
or any non-PushT task** in any form.

---

## Q4 — Example command for a new-task rollout

**IWS does not support Block / Can / Square / Kitchen.** Those are
robomimic/robosuite benchmarks. The four official IWS tasks are:
`pusht`, `single_grasp`, `bimanual_sweep`, `bimanual_rope`.

To run a single open-loop rollout on, say, **bimanual_sweep cam1
episode 0**, the full bring-up + invocation would be (do not execute as
written without checking GPU/conda availability):

### One-shot setup (only needed once)

```bash
# 1. Create + activate the IWS conda env (Python 3.10 + torch + lightning)
mamba env create -f interactive_world_sim/conda_env.yaml
conda activate iws

# 2. Install IWS deps + the package itself
uv pip install -r interactive_world_sim/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126/
pip install -e interactive_world_sim

# 3. Pull checkpoints (~GBs over HF)
cd interactive_world_sim
bash scripts/download_checkpoints.sh
# → outputs/{pusht_cam1, single_grasp_*, bimanual_sweep_*, bimanual_rope_*}/

# 4. Pull mini eval episodes
bash scripts/download_mini_data.sh
# → data/mini/{pusht, single_grasp, bimanual_sweep, bimanual_rope}/val/episode_*.hdf5
cd ..
```

### One-shot rollout invocation (the actual generation)

The local headless wrapper `headless_rollout.py` accepts the IWS task
via its CLI (the script uses the obs key + ckpt path to disambiguate).
For bimanual_sweep / cam1 / episode 0:

```bash
conda activate iws  # uses the IWS env, NOT the part1 venv

python headless_rollout.py \
  --ckpt    interactive_world_sim/outputs/bimanual_sweep_cam1/checkpoints/best.ckpt \
  --episode interactive_world_sim/data/mini/bimanual_sweep/val/episode_0.hdf5 \
  --output  my_project/output/bimanual_sweep_cam1_ep0.mp4 \
  --obs_key camera_1_color \
  --resolution 128
```

Outputs that would land on disk:
- `my_project/output/bimanual_sweep_cam1_ep0.mp4` (199-ish frames,
  pred/GT side-by-side, mp4v 15 fps)
- `my_project/output/bimanual_sweep_cam1_ep0_pred.npy` (T, 128, 128, 3)

**Caveats before claiming this works:**
- The `+scene` value in the upstream launcher (`bimanual_sweep_cam_1`)
  is *not* exposed by `headless_rollout.py`. The local wrapper hard-
  codes the rendering path via `obs_key`. If the bimanual_sweep
  checkpoint expects a `+scene` override that `headless_rollout.py`
  doesn't pass, the wrapper may need a 2-line patch.
- The bimanual tasks use **two cameras**; the launchers have
  `*_cam0_and_cam1_*.sh` variants. `headless_rollout.py` is single-
  camera only — for paired-camera rollouts the wrapper needs more
  invasive changes.
- The conda env is several GB and downloads pinned PyTorch CUDA wheels.
  Don't attempt on a non-CUDA machine.

For the simplest possible test (no scene/camera complexity), use
**single_grasp cam1** instead — it has the same single-camera signature
as PushT:

```bash
python headless_rollout.py \
  --ckpt    interactive_world_sim/outputs/single_grasp_cam1/checkpoints/best.ckpt \
  --episode interactive_world_sim/data/mini/single_grasp/val/episode_0.hdf5 \
  --output  my_project/output/single_grasp_cam1_ep0.mp4 \
  --obs_key camera_1_color \
  --resolution 128
```

---

## Q5 — Public IWS links

Pulled from `interactive_world_sim/README.md` and the download scripts.

| Resource             | URL |
|----------------------|-----|
| GitHub repo          | <https://github.com/WangYixuan12/interactive_world_sim> |
| Project page         | <https://www.yixuanwang.me/interactive_world_sim/> |
| Paper PDF            | <https://www.yixuanwang.me/interactive_world_sim/texts/main.pdf> |
| Demo video           | <https://youtu.be/H6Um4zZYm5Y> |
| HF checkpoints       | <https://huggingface.co/yixuan1999/interactive-world-sim-checkpoints> |
| HF mini eval data    | <https://huggingface.co/datasets/yixuan1999/interactive-world-sim-min-data> |
| HF full ALOHA data   | (referenced in `download_full_data.sh`; same author namespace) |

Affiliations on the paper: Columbia, TRI, Amazon, UIUC. Code license
in `interactive_world_sim/LICENSE`.

---

## Bottom line for Part 2 paper scope decisions

- **Adding a new task to §V.A is non-trivial.** Requires (a) installing
  the IWS conda env with CUDA-pinned torch, (b) downloading multi-GB
  checkpoints and HDF5 episodes, (c) likely a small patch to
  `headless_rollout.py` for non-PushT scene/camera arguments, (d) GPU
  inference time per episode.
- **All four official IWS tasks are real-ALOHA bimanual scenes.** None
  are top-down or planar. The HSV-pink-T detector in
  `part2/perception_iws.py` will not transfer to single_grasp /
  bimanual_sweep / bimanual_rope without per-task re-tuning of HSV
  bounds (and rope is deformable — Path A perception will fail, Path
  C optical-flow fallback would be needed; this matches the "reject
  highly deformable" guidance in the original Part 2 plan §0.3).
- **No checkpoints / no episodes on disk for any task** — confirms the
  Phase-0 conclusion that Part 2 must run on the cached PushT rollouts
  unless the user opts into the multi-hour download + env setup detour.
