# Part 1 — Pre-implementation Notes

Reading of `part1_mujoco_kf.py` and `pusht_scene.xml` before any code change.

## Scene (`pusht_scene.xml`)

- Planar scene (gravity 0 0 0), physics timestep `0.002 s`.
- Top-down camera at `pos=(0, 0, 0.5)`, `fovy=45°`, rendered `256 × 256` RGB.
- Camera half-width in world = `0.5 * tan(22.5°) ≈ 0.2071 m`.
- `tblock` body: slide-x, slide-y, hinge-rot joints → `qpos[0:3] = [x, y, θ]`.
  Composite geometry: bar (10×2 cm) + stem (2×8 cm), purple material.
- `pusher` body: slide-x, slide-y → `qpos[3:5]`. Velocity actuators `(vx, vy)`
  with gain `kv=50`.

## Pipeline (`part1_mujoco_kf.py`)

- `generate_episode(n_steps, seed)` rolls a three-phase scripted pusher
  policy (approach / push-to-origin / orbit), rendering one RGB frame per
  step. Each env step runs 25 MuJoCo substeps ⇒ **`dt = 25 × 0.002 = 0.05 s`
  (20 Hz)**.
- `detect_t_block(image_rgb, add_noise, rng, prev_theta)`:
  - Corruption hook already present: additive Gaussian (σ=20), Gaussian blur
    (random odd kernel ∈ {3,5,7}), multiplicative brightness jitter
    (`0.85…1.15`). Controlled by `add_noise` flag.
  - Segmentation: HSV threshold `[135,30,40]–[170,255,255]` + open/close
    morphology.
  - Position: mask centroid.
  - Orientation: top-2 convex-hull defect points give the bar midpoint, stem
    direction = perpendicular to the bar pointing toward the centroid; PCA
    fallback when ≥2 defects are not found. A `-π/2` offset and continuity
    resolve sign ambiguity against `prev_theta`.
  - Returns `(cx_pixel, cy_pixel, θ_world_rad)` or `None`.
  - Note: the plan's description ("HSV threshold + centroid + PCA") is a
    simplification; the real detector uses convex-hull defects first, PCA
    only as fallback. `detect_tblock(img) → (x, y, θ) | None` in
    `part1/perception.py` will wrap this and convert pixel → world in a
    single call.
- `pixel_to_world` / `world_to_pixel` flip the y axis and use `fov_half_width`
  ≈ `0.2071`. (`world_to_pixel` has a stale `0.35` default but every call
  site overrides it — no bug in practice, will pass `fov_half` explicitly.)

## Current KF

- State `[x, y, θ, vx, vy, ω]`, 6-D. Constant-velocity model.
- `F`: identity with `dt` entries coupling position to velocity.
- `H`: `[I_3 | 0_3]` — observes `(x, y, θ)` directly.
- `Q = diag(1e-4, 1e-4, 1e-3, 5e-3, 5e-3, 1e-2)`.
- `R = diag(1e-4, 1e-4, 2e-2)`.
- `update` already wraps innovation[2] via `(a + π) mod 2π − π`. So the
  existing KF is already "EKF-Cartesian"-equivalent except it doesn't log
  NIS / innovation. The upgrade replaces it with `EKF_SE2_Cartesian` and
  adds logging.

## Confirmed facts (match the plan's assumptions)

| Plan claim | Verified? |
|---|---|
| `z = [x, y, θ]` (3-D) in world coords | ✓ (after `pixel_to_world`) |
| `qpos[0:3]` is T-block GT pose | ✓ (slide-x, slide-y, hinge-rot) |
| State is 6-D `[x,y,θ,vx,vy,ω]`, constant-velocity | ✓ |
| `dt = 0.05 s` | ✓ (25 substeps × 0.002 s) |
| Pixel-noise / blur injection hooks exist | ✓ (`add_noise=True` branch in `detect_t_block`) |

## Deviations from plan (to flag explicitly)

1. **Perception details.** Plan says "HSV + centroid + PCA"; actual detector
   uses convex-hull defects for orientation with PCA fallback. Keeping the
   existing (more accurate) detector — `part1/perception.py` just extracts it.
2. **Corruption hook.** Plan asks for `apply_corruption(img, sigma_pixel,
   blur_px, brightness_shift)`. The existing detector has equivalent logic
   baked in; `corruption.py` will expose the pure function and the detector
   will consume `add_noise=False` images so corruption is controlled
   externally (clean separation).
3. **Measurement `R`.** Plan sweep wants `R = (σ_px · meters/pixel)^2` on
   `(x,y)` and `5e-4 rad^2` on `θ` for every `σ`. The fixed `R` in the legacy
   script is looser — the new driver will override it per-sweep.
4. **Scope.** Everything in the plan's Section 1 goes under `part1/`. Legacy
   `part1_mujoco_kf.py` becomes a thin wrapper delegating to the new modules
   so the "no regression" check still passes. MuJoCo scene, detector
   thresholds, and camera intrinsics are untouched.
