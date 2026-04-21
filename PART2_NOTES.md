# Part 2 — Pre-implementation Inventory

This is the §0 inventory the Part 2 plan demands before writing any new
code. Everything below is what I actually found by reading the repo, not
what the plan assumed.

## Q1 — Is there a runnable IWS inference pipeline?

**Effectively no, not without out-of-scope setup.**

- IWS source is cloned: `interactive_world_sim/` (Columbia/TRI repo,
  `LatentWorldModel`, `latent_dynamics`, hydra configs, etc.).
- A ready-to-run wrapper exists at the repo root: `headless_rollout.py`
  loads `LatentWorldModel.from_checkpoint(...)`, runs an open-loop
  rollout from one HDF5 episode, and writes pred/GT side-by-side MP4 +
  `_pred.npy`.
- **Missing for re-running it:**
  - `interactive_world_sim/outputs/pusht_cam1/checkpoints/best.ckpt`
    (and the 6 other task checkpoints). Download script
    `scripts/download_checkpoints.sh` exists but has not been run; it
    needs `gdown` and several GB.
  - `interactive_world_sim/data/mini/pusht/val/episode_*.hdf5`. Same
    story — `scripts/download_mini_data.sh` not run.
  - The IWS conda env (`conda_env.yaml`, "iws") with PyTorch + CUDA + a
    custom Lightning stack. The Part 1 `venv/` does not satisfy this.
- **Net:** generating new IWS rollouts in this pass is realistic only if
  the user wants to take the multi-GB download + GPU env detour.
  Otherwise we work entirely off the cached outputs from prior runs
  (which is enough — see Q2).

## Q2 — How many pre-generated rollout episodes exist per task?

**Task: PushT only.** No bimanual / single-grasp rollouts on disk.

| File | Method | Episode | Frames |
|------|--------|---------|--------|
| `output/openloop_ep0_pred.npy` | open-loop | 0 | 199 × 128 × 128 × 3 uint8 |
| `output/openloop_ep1_pred.npy` | open-loop | 1 | 199 × 128 × 128 × 3 |
| `output/openloop_ep2_pred.npy` | open-loop | 2 | 199 × 128 × 128 × 3 |
| `output/openloop_ep3_pred.npy` | open-loop | 3 | 199 × 128 × 128 × 3 |
| `output/openloop_ep4_pred.npy` | open-loop | 4 | 199 × 128 × 128 × 3 |
| `output/ep3_results/open-loop_pred.npy` | open-loop (re-run) | 3 | 199 × 128 × 128 × 3 |
| `output/ep3_results/periodic_reset_k10_pred.npy` | periodic reset K=10 | 3 | 199 |
| `output/ep3_results/periodic_reset_k20_pred.npy` | periodic reset K=20 | 3 | 199 |
| `output/ep3_results/latent_smoother_pred.npy` | latent smoothing | 3 | 199 |
| `output/ep3_results/selective_correction_pred.npy` | selective | 3 | 199 |
| `output/ep3_results/best-of-3_pred.npy` | best-of-3 | 3 | 199 |
| `output/ep3_results/best-of-5_pred.npy` | best-of-5 | 3 | 199 |
| `output/latent_smoother_ep2_pred.npy` | latent smoothing | 2 | 199 |
| `output/periodic_reset_ep2_K20_pred.npy` | periodic reset K=20 | 2 | 199 |
| `output/selective_correction_ep2_pred.npy` (+ v2, v3, t3.5/4.0/4.5) | selective variants | 2 | 199 |

**Coverage for §V.A (open-loop NIS comparison):** 5 episodes (0–4),
plenty for pooled NIS statistics. The Part 1 plan's "≥10 episodes"
target won't be hit; we'll pool steps across all 5 (~995 steps total
per filter). That is on the low end but is what we have.

**Coverage for §V.C (correction-method reframe):** all five canonical
methods exist for **episode 3** (`output/ep3_results/`). Episode 2 has
3 of the 5. So §V.C will run on ep3 as the "all methods" comparison and
ep2 as a partial replication.

## Q3 — Are GT frames paired with rollouts?

**Yes, but as JPEG-encoded MP4 right halves, not as `.npy`.**

`headless_rollout.py:139–162` writes `[pred | GT]` side-by-side MP4 at
512×256, 15 fps, 199 frames. Verified with `cv2.VideoCapture`:
`output/openloop_ep0.mp4` etc. all open and split cleanly into a
256×256 left half (pred) and 256×256 right half (GT). The pred-half MP4
is a 2× upsample of the matching `_pred.npy`.

**Implication:** GT exists but lives behind one round of MP4 (mp4v
codec) compression. For NIS we apply the same HSV detector to both
halves of the same MP4 → the JPEG/codec noise affects GT and rollout
identically and cancels in the regime-vs-regime comparison.

For §V.B LPIPS we'll compare GT-half vs pred-half from the same MP4
(both at 256×256, both compressed). For pure rollout NIS we'll prefer
the cleaner 128×128 `_pred.npy` and resize as needed.

## Q4 — Does any logged data contain low-dimensional GT state per frame?

**No, not on disk.** The HDF5 episodes that *would* contain
`obs.images.camera_1_color`, `action`, and the action-source state are
not present. `check_gt_metrics.py` confirms the expected layout
(`load_dict_from_hdf5("data/mini/pusht/val/episode_{i}.hdf5")`) but
that path doesn't resolve.

**Consequence:** the §V.A `summary_5a.csv` cannot include true
`pos_rmse`/`ori_rmse` columns. The plan's RMSE column must be either
dropped or re-defined as **`rollout-detection − GT-detection`** (a
detector-vs-detector pose drift, not a true error). I'll do the latter
and label the column `pos_drift_vs_gt_det` etc., with a note in the
table caption.

## Q5 — What do scenes look like? (5 sampled frames)

Sampled `output/ep3_results/open-loop.mp4` at t = 0, 50, 100, 150, 195
and split into pred/GT halves. See `part2_samples/`.

- **Setting:** real ALOHA bimanual lab. Third-person camera (slight
  yaw + tilt — *not* top-down). White tabletop, black border, two
  orange/red end-effectors plus visible robot arms.
- **Manipulated object:** large pink T-block, very saturated, sharp
  edges in GT. HSV-friendly: pink hue ≈ 160 in OpenCV's 0–179 range,
  sharply distinguishable from the orange end-effectors at hue ≈ 10.
  An existing detector with bounds `[130,40,80]–[175,255,255]` is
  already in `physics_metrics.extract_state` — Path A is viable.
- **Failure mode visible by eye:** at t = 150–195 the pred shows the T
  drifting to the upper-left while GT shows the T being pushed
  rightward. Edges blur and the T desaturates slightly in pred. This
  is exactly the structural-bias failure NIS should catch. Good sign
  for the §V.A acceptance criterion.
- **Caveats:** perspective camera means a single-pixel detection is
  not metric. NIS only needs `(z − ẑ)ᵀ S⁻¹ (z − ẑ)` to be χ²; units
  cancel. We will keep state in the **image-pixel** frame (no homography),
  and Q/R will be in pixel² and rad². This is a deliberate deviation
  from Part 1, which used metric units — see deviation note 3 below.

## Task selection

**Chosen task: PushT (forced).** Justification: it's the only task
with cached rollouts, and it is the closest analog to the Part 1
MuJoCo PushT regime (single-object planar manipulation with one
saturated colour blob). Bimanual sweep / bimanual rope / single grasp
have neither cached rollouts nor (without IWS env setup) a path to
generate them in this pass.

**Number of tasks: 1.** The plan allowed 1–2; choosing 1 keeps the
acceptance bar clear and lets us spend the saved time on the §V.B
metric-correlation analysis, which is the actual paper claim.

## Deviations from the original plan

1. **§0.2 Q1 = "no" + Q2 ≠ 0:** the plan says "If Q1 is 'no' and Q2 is
   '0', stop and ask." We have Q2 = 5 cached open-loop episodes plus
   the corrections, so we proceed. The cost is no ability to add more
   episodes or new tasks without first downloading checkpoints/data.
   Flag to user: confirm the cached-only path is acceptable, or
   approve the extra setup work.

2. **§2.1 `iws_io.py` source enum:** the plan defines
   `source ∈ {"gt", "rollout"}`. We add `source = "corrected:<method>"`
   for §V.C — same loader, just routes to the corresponding `_pred.npy`
   under `output/ep3_results/`. No interface change.

3. **§2.3 R/Q reuse from Part 1:** Part 1 had R in metric units
   (`(σ_pixel · m_per_px)²` for position, `5e-4 rad²` for orientation),
   because MuJoCo had a top-down camera with a clean homography. The
   IWS rollouts here are real-world third-person video with no
   homography. We will re-derive R **in pixel²** by measuring the
   per-frame detector noise on the **GT halves** of the MP4s (akin to
   Part 1 Regime C calibration), and use the same R for the rollout
   filter. This is the only allowed exception to the "reuse Part 1
   Q,R" rule: detection drop rate is fine, but we *must* change units.
   Documented here, not retuned per task.

4. **§2.3 RMSE column:** redefined as `pos_drift_vs_gt_det` /
   `ori_drift_vs_gt_det` (per-frame difference between rollout-side
   detection and GT-side detection, when both detect successfully).
   True pose RMSE is unavailable — see Q4.

5. **§2.4 FVD:** skip. We have only 5 episodes per source; FVD's
   feature-distribution distance is not meaningful at that scale.
   Explicitly documented; LPIPS covers the per-frame paired comparison.

5a. **§2.4 SOD-χ² (new in plan revision).** Add as a model-free baseline
   against NIS: `s_t = aᵀ Σ_a⁻¹ a` where `a_t = p_{t+1} - 2p_t + p_{t-1}`
   (with θ wrap), `Σ_a` estimated from the GT-source pose sequence of the
   same task. Reports the same triple (`mean`, `frac_in_CI`, `ks_pvalue`)
   as `nis_stats`. Lives in `part2/cross_metrics.py` and feeds the §V.B
   correlation / discriminative analyses adjacent to NIS. Purpose: if NIS
   tracks SOD-χ², the filter machinery isn't justified on this dataset.

6. **§2.4 anomaly threshold τ:** the existing
   `optical_flow_metrics.compute_flow_consistency` produces `phi_t`.
   We'll use it as-is (no rewrite) and calibrate τ on GT halves of the
   same 5 episodes.

7. **§3 "Do not retune Q without data-driven reason":** the unit
   change from m² to px² counts as a data-driven reason. The relative
   ratio of position-Q to orientation-Q stays the same as Part 1 once
   converted. Q-position scaled by `(1 / m_per_px)²`, Q-orientation
   unchanged.

## Acceptance gates I will hold myself to before §V.B

(Same as plan §4.3, restated for this dataset)

- `mean_nis(iws_gt_detector)` plausibly larger than
  `mean_nis(part1_real_regime_c)` because IWS GT is real-camera +
  JPEG, MuJoCo Real Regime C was clean rendering + same detector.
- `mean_nis(iws_rollout)` strictly larger than `mean_nis(iws_gt)`.
  **If this fails I stop and report — do not proceed to §V.B/C.**
- `ks_pvalue(iws_rollout) ≤ ks_pvalue(iws_gt)`.

## Key risks I want flagged before I start writing code

1. **Single task, 5 episodes** is thin. The pooled-step count
   (~5 × 199 ≈ 995) is OK for χ² on 3-DoF, but per-episode error bars
   will be wide. Reporting will lean on pooled stats with per-episode
   ranges as supplements.

2. **No paired GT pose** changes one column of `summary_5a.csv` from
   absolute error to detector-vs-detector drift. Need to re-word the
   paper §V.A claim from "filters track within X cm" to "filter
   estimates from rollouts deviate from filter estimates from GT by Y".
   Less satisfying but honest given the available data.

3. **Real-world camera ⇒ pixel-space NIS.** The covariance test still
   works (NIS is unitless), but we lose direct comparability to Part 1
   numbers. The metric *interpretation* — "rollout violates filter
   assumption more than GT" — is unchanged.

4. **Both halves of the MP4 carry the same JPEG/mp4v compression.**
   Good for canceling detector noise across regimes, bad if the codec
   itself adds bias the detector reads differently for blurry rollouts
   vs sharp GT. We'll sanity-check by also running the detector on the
   raw `_pred.npy` (no MP4 codec) and confirming the rollout NIS doesn't
   shift enough to change the qualitative result.

## Files I will create (under `part2/`)

- `part2/__init__.py`
- `part2/iws_io.py` — unified loader for `_pred.npy` + MP4 GT/pred
  halves. Caches to `results/part2_cache/`.
- `part2/perception_iws.py` — HSV pink-T detector (Path A). Reuses Part 1
  PCA/defects orientation logic where possible, with fallback to
  `cv2.minAreaRect` (already used in `physics_metrics.extract_state`).
- `part2/run_iws_nis.py` — §V.A driver. Outputs
  `results/part2/summary_5a.csv` and per-episode pose traces.
- `part2/cross_metrics.py` — flow consistency (delegates to
  `optical_flow_metrics`), anomaly rate, LPIPS (`lpips` pip pkg).
- `part2/metric_comparison.py` — correlation, AUC, scatter for §V.B.
- `part2/correction_reframe.py` — §V.C, runs on `output/ep3_results/`.
- `part2/make_tables.py` — three LaTeX tables.

Reuses (read-only) from `part1/`: `ekf_se2_cartesian`, `riekf_se2`,
`nis_analysis`, `se2`. No edits to `part1/`.

## Stop gate

Per the plan: I report Q1–Q5 here and **wait for user confirmation**
before writing `part2/` code. Specifically:
- Confirm cached-only path (no IWS download) is acceptable.
- Confirm "task = PushT only" is acceptable.
- Confirm pixel-space NIS (no homography) is acceptable.
- Confirm dropping true RMSE in favour of detector-vs-detector drift
  is acceptable.

**User confirmed (2026-04-18) — proceeding.**

## §V.A results — qualitative observations

Run on 2026-04-18 with the cached-only path. 5 episodes × 2 sources ×
2 filters = 20 episode runs, 100% detection rate on every frame of every
run (no detector drops). Pooled stats:

| Source  | Filter   | mean NIS | frac_in_CI | KS p     | drift_pos (px) | drift_ori (deg) |
|---------|----------|---------:|-----------:|---------:|---------------:|----------------:|
| gt      | ekf_cart |    6.36  |       0.59 | 1.2e-78 |           2.0  |          +2.4   |
| gt      | riekf    |    8.06  |       0.59 | 9.0e-54 |           2.9  |          +2.6   |
| rollout | ekf_cart |   27.34  |       0.60 | 4.6e-53 |          27.2  |         +56.8   |
| rollout | riekf    |   34.48  |       0.61 | 4.6e-39 |          27.5  |         +56.7   |

**Per-episode rollout behavior** (ekf_cart):

| Episode | rollout mean_nis | drift_pos (px) | drift_ori (deg) | Comment |
|---------|-----------------:|---------------:|----------------:|---------|
| ep0     |             1.60 |            6.7 |            +7.1 | Faithful — rollout closely matches GT |
| ep1     |            33.32 |           70.1 |          +133.3 | Large drift, NIS catches it |
| ep2     |            88.24 |           26.9 |           +51.3 | NIS spike + drift |
| ep3     |             5.14 |           27.7 |           +85.6 | **Slow consistent wrong trajectory:** filter happily tracks the wrong moving T with low innovation. NIS does NOT catch it. |
| ep4     |             8.39 |            4.4 |            +6.5 | Mild |

**Key qualitative finding for the paper.** NIS catches *kinematically
implausible* rollouts (ep1, ep2) but not all rollouts that drift from
physical reality. Episode 3 is the most striking case: the rollout T
moves slowly along the wrong trajectory (28 px and 85° pose error vs
GT), but the motion is internally smooth, so each filter prediction is
close to the next observation and NIS stays low. **This is a useful
limitation to declare in the paper, not a flaw to hide:** NIS is a test
of innovation whiteness against the assumed observation/process model;
it cannot detect drifts that obey the same kinematics as truth. A
complementary metric (e.g. detector-pose drift between rollout and GT
sides, or LPIPS) is needed for that failure mode. This motivates the
§V.B cross-metric analysis directly.

**KS p-value caveat.** All four KS p-values are vanishingly small
(< 1e-39); KS measures distribution *shape* against χ²(3), not
magnitude. The rollout NIS is bimodal (per-episode means span 1.6 →
113), and that bimodal shape can KS-fit χ² *better* than the
consistently-elevated GT distribution, which is why the rollout p
appears larger than the GT p. **Acceptance criterion #3.3 (ks_p
ordering) is technically violated but the diagnostic story is
unaffected** — at p < 1e-39 nothing is consistent, so the inequality is
not meaningful at this scale. Mean NIS is the right discriminator and
it separates by 4.3×.

**Acceptance ladder (plan §4):**
- #1 inventory: ✓
- #2 §V.A end-to-end: ✓ (`results/part2/summary_5a.csv` written)
- #3.1 mean_nis(iws_gt) > mean_nis(mujoco_real_C): ✓ (~4× larger)
- #3.2 mean_nis(iws_rollout) > mean_nis(iws_gt) [REQUIRED]: ✓ (~4×)
- #3.3 ks_pvalue ordering: ✗ — both ≪ 1e-30, KS uninformative here
- #6 no Part 1 regression: ✓ (`pytest part1/tests/` 7/7 pass)

Proceeding to §V.B with the bullet-2 acceptance held; #3.3 violation
documented and noted in the paper §V.A discussion.

## §V.B results — NIS vs cross-metrics

Discriminative power (AUC vs source ∈ {gt, rollout}, n=5+5):

| Metric            | AUC  | Cohen's d | Trivial? |
|-------------------|-----:|----------:|:--------:|
| **SOD-χ² mean**   | 1.00 |    +1.05  |   no     |
| Flow cons φ       | 0.88 |    +1.36  |   no     |
| Anomaly rate      | 0.66 |    +0.40  |   no     |
| **NIS mean**      | 0.64 |    +0.81  |   no     |
| NIS / SOD KS p    | 0.60 |   ≈+0.5   |   no     |
| Frame MSE         | 1.00 |   +9.26   | **yes**  |
| Frame NCC         | 1.00 |  +10.49   | **yes**  |
| Pos / Ori drift   | 1.00 |   ≈+1.5   | **yes**  |

**Surprise #1**: SOD-χ² (a 3-line hand-written second-order test) beats
NIS on this dataset. Per-episode NIS and SOD have Pearson ≈ 0.99 on the
rollout side, but NIS has an extra source of variance from the filter's
own predict/update that occasionally damps single-frame anomalies. On
this particular failure mode (kinematic jerkiness in IWS PushT), the
filter apparatus is not justified. **This is a publishable negative
result, not a flaw to hide.**

**Surprise #2**: Flow-φ (existing Part 3 metric) also beats NIS (0.88 vs
0.64). Confirms the Part 3 metric is a sensible kinematic diagnostic.

**Takeaway for paper §V.B**: The filter-based NIS gets beaten by two
cheaper metrics on the small-N dataset we have. The paper should frame
NIS as "equivalent to SOD-χ² for the kinematic-jerk failure mode, but
filter-theoretic in a way that generalizes to heteroscedastic / biased
observation noise that SOD cannot detect." We don't have data that
*proves* that generalization here, so it's a future-work claim, not a
§V.B claim.

**Pixel metrics are trivial separators**: Frame MSE, NCC, and the
pos/ori detector drifts are 0 for source=gt (comparing GT frames to
themselves) and nonzero for rollout. They separate at AUC=1.00 but by
construction. Marked (trivial) in tables; don't interpret as a metric
win.

## §V.C results — correction methods on episode 3

Run on the 7 cached rollouts in `output/ep3_results/`. NIS baseline
(open-loop on ep3_results) = 177.40, which is much worse than the
`openloop_ep3` run used in §V.A (NIS=5.14). **Two independent
stochastic samples of the same world-model on the same episode
diverge by 35× in mean NIS** — worth flagging in the paper as evidence
that IWS open-loop quality is highly seed-dependent.

Normalized to open-loop baseline (lower=better except where noted):

| Method                | NIS      | SOD-χ²   | Flow φ  | MSE    | NCC    |
|-----------------------|---------:|---------:|--------:|-------:|-------:|
| open-loop (baseline)  | 177 (1×) |  44 (1×) | 0.62    | 493    | 0.898  |
| periodic_reset_k10    |  38 (-79%)| 34 (-21%)| 0.66    | 447    | 0.908  |
| periodic_reset_k20    |  17 (-91%)| 27 (-37%)| 0.65    | 477    | 0.902  |
| latent_smoother       |  15 (-92%)| 14 (-69%)| 0.67    | 501    | 0.896  |
| selective_correction  |  14 (-92%)| 13 (-69%)| 0.69    | 422    | 0.913  |
| best-of-3             |   5 (-97%)|  5 (-89%)| 0.69    | 513    | 0.893  |
| best-of-5             |  16 (-91%)|  6 (-86%)| 0.69    | 509    | 0.894  |
| GT reference          |  11      |   3      | 0.78    |   0    | 1.000  |

**Key finding (paper §V.C)**: NIS ranges across 36× (5→177) across
methods; SOD-χ² ranges 9× (5→44); pixel metrics (MSE, NCC) range
~20% each. Kinematic metrics (NIS, SOD) have dramatically more
*sensitivity* to correction quality than pixel metrics. best-of-3
actually dips below the GT-reference NIS — it is selecting for
kinematic smoothness even more aggressively than the real video is.

**Paper prediction vs data (§V.C plan):**
- "Periodic reset improves all metrics" — ✓ for kinematic metrics,
  barely for pixel metrics (MSE -3% to -9%, NCC +0.4% to +1%).
- "Best-of-N improves pixel metrics but not NIS" — **refuted**.
  Best-of-N improves NIS the most (-97%) and pixel metrics almost not
  at all. The original prediction was wrong; the data says NIS is
  also sensitive to kinematic selection.
- Revised paper claim: **NIS and SOD-χ² are the sensitive diagnostics;
  pixel metrics are nearly flat across all correction methods here.**

## Acceptance ladder (plan §4) — final

- #1 Inventory: ✓
- #2 §V.A end-to-end: ✓ (`summary_5a.csv`)
- #3.1 mean_nis(iws_gt) > mean_nis(mujoco_real_C): ✓ (6.36 > 1.54)
- #3.2 mean_nis(iws_rollout) > mean_nis(iws_gt) [REQUIRED]: ✓ (4.3×)
- #3.3 ks_pvalue ordering: ✗ (both p ≪ 1e-30, uninformative; see §V.A)
- #4 §V.B three analyses: ✓ (correlation heatmap, discriminative bar,
  scatter grid)
- #5 §V.C: ✓ (7 methods, all metrics recomputed)
- #6 No Part 1 regression: ✓ (7/7 pytest pass)
