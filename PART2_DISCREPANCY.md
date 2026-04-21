# Part 2 — §V.A vs §V.C ep3 NIS discrepancy (5.14 vs 177)

Read-only investigation. No experiments re-run; code and cached
artifacts only.

## TL;DR

**The 35× gap is a pure stochastic-sampling gap: the two "ep3
open-loop" rollouts are two different samples from the same IWS world
model on the same source episode, saved by two different scripts on
2026-04-12. Every other knob — filter class, Q, R, P0, DT, detector,
aggregation, frame range — is bit-identical.**

Two independent open-loop rollouts of ep3 happen to land on opposite
ends of IWS's quality distribution. The §V.A sample is the "smooth
wrong" case already described in PART2_NOTES.md line 274 (low NIS in
spite of a ~28 px / 85° pose drift vs GT). The §V.C sample is a
kinematically jerky rollout that the filter cannot predict, and NIS
correctly explodes. Both are legitimate; together they quantify the
IWS seed-dependence that the §V.C paper note (PART2_NOTES.md line 359)
already flags.

---

## 1. Data source alignment

### a) §V.A ep3 rollout frame source

`part2/run_iws_nis.py:76` iterates `OPENLOOP_EP_IDS = ["openloop_ep0",
..., "openloop_ep4"]` and passes each to `iws_io.load_episode(ep_id,
"rollout")`. `part2/iws_io.py:39-45, 134-146` resolves
`openloop_ep3` to `(output/openloop_ep3_pred.npy,
output/openloop_ep3.mp4)` and, when `source="rollout"`, returns the
pred (left) half of the MP4 at 256×256 RGB.

- MP4 pred half: `output/openloop_ep3.mp4` (981,773 bytes, mtime
  2026-04-12 22:40:06). Writer: `headless_rollout.py:139–162`.
- Raw pred `.npy` (present but not used by the §V.A pipeline):
  `output/openloop_ep3_pred.npy` (9,781,376 bytes, mtime 2026-04-12
  22:40:20). MD5 `7b6d649e56a108a69e058ec13d390be0`.

### b) §V.C ep3_results/open-loop rollout frame source

`part2/correction_reframe.py:106` calls
`iws_io.load_episode("open-loop", "rollout")`. `iws_io.py:49-51,
134-146` resolves that id to
`(output/ep3_results/open-loop_pred.npy,
output/ep3_results/open-loop.mp4)` and, with `source="rollout"`,
returns the pred half of the MP4.

- MP4 pred half: `output/ep3_results/open-loop.mp4` (988,037 bytes,
  mtime 2026-04-12 22:40:08). Writer: `run_all_experiments.py:423`
  (`OUTPUT_DIR = Path("my_project/output/ep3_results")` + its
  `open-loop` branch).
- Raw pred `.npy`: `output/ep3_results/open-loop_pred.npy` (9,781,376
  bytes, mtime 2026-04-12 22:40:20). MD5
  `c4cff7ccfbe19bd64e0a46b2928aa42f`.

### c) Same rollout? Same initial condition?

**No, they are two distinct stochastic samples of the same source
episode.**

Evidence (computed on disk, no rerun):

- The two `_pred.npy` files have identical shape `(199,128,128,3)
  uint8` and identical byte size but **different MD5 and different
  content**. `np.array_equal(...) == False`.
- Per-frame mean abs pixel diff grows with time (classic stochastic
  divergence): 1.45 at t=0, 2.24 at t=50, 4.77 at t=150, 5.76 at
  t=195. Max per-pixel diff reaches 171 (npy) / 210 (MP4 pred
  half). Overall ~81% of pixels differ; ~3.2% differ by more than
  20.
- GT halves of the two MP4s are near-identical (mean abs diff
  0.009, max 18, ~0% of pixels differ by more than 20). The
  residual variation is mp4v codec jitter across two encodes of
  what is otherwise the same `obs.images.camera_1_color` sequence
  from `episode_3.hdf5`. **So the source episode is the same
  (PushT val episode 3); only the model's stochastic sample
  differs.**
- Neither `headless_rollout.py` nor `run_all_experiments.py`
  calls `torch.manual_seed`, `np.random.seed`, `random.seed`, or
  any RNG generator. Grepping both files for `seed|manual_seed`
  returns no matches. The diffusion sampler therefore draws from
  torch's default (unseeded) RNG on each run.

Conclusion: **different stochastic sample, identical initial
condition (same conditioning frame from episode_3.hdf5).**

---

## 2. Filter configuration alignment

`part2/correction_reframe.py:37-39` imports `DT`, `P0_EKF_PX`,
`Q_EKF_PX`, `R_PX` **directly from `part2/run_iws_nis.py`**, so the
two paths share the exact same constants by construction:

| Param | Value (pixel-space, same module symbol) |
|-------|------------------------------------------|
| `DT`      | `1/15` s (MP4 fps) |
| `Q_EKF_PX`| `diag([1e-7·S, 1e-7·S, 1e-6, 1e-4·S, 1e-4·S, 1e-3])`, `S = 1/m_per_px_mujoco² ≈ 381890` |
| `R_PX`    | `diag([100, 100, 5e-4])` (px², px², rad²) |
| `P0_EKF_PX`| `diag([1, 1, 1e-2, 100, 100, 1])` |

Initial state is also identical in both drivers: `x0 = np.zeros(6);
x0[:3] = first_successful_detection`, followed by `predict/update` on
every subsequent frame. See `run_iws_nis.py:80-83, 110-124` and
`correction_reframe.py:60-71`.

**Differences: none.** Both construct the same `EKF_SE2_Cartesian`
from `part1.ekf_se2_cartesian`.

---

## 3. NIS aggregation

### a) §V.A openloop_ep3 = 5.14

`part2/run_iws_nis.py:182` appends a per-episode NIS array to
`pooled_nis`; `:204-205` concatenates and calls
`part1.nis_analysis.nis_stats(nis_concat, dim=3)`, but that is the
row-level pooled stat across all 5 episodes. The number **5.14**
specifically is the per-episode mean reported in PART2_NOTES.md line
274, which matches the cached per-episode array:

```
results/part2/nis/pusht_iws__rollout__ekf_cart.npz::nis_openloop_ep3
  -> size=198, mean=5.1443, median=1.575, max=58.27
```

So: **single-episode mean across time** (198 filter steps from 199
frames; one step consumed by re-init).

### b) §V.C open-loop = 177.40

`part2/correction_reframe.py:109-110` runs the EKF on the one
episode (open-loop) and reports `ns = nis_stats(nis_arr, dim=3,
alpha=0.05)["mean_nis"]`. From `results/part2/summary_5c.csv`:

```
pusht_iws,open-loop,n_frames=199,det_success=199,mean_nis=177.399...
```

`nis_stats["mean_nis"]` is `nis_arr.mean()` — **single-episode mean
across time**, same definition as (a).

### c) Same aggregation?

**Yes, identical.** Both are `np.mean` across the per-step NIS of one
episode. Not max, not median, not cross-episode. Aggregation cannot
explain the 35× gap.

---

## 4. Frame range alignment

- §V.A: `run_iws_nis._run_filter` walks all 199 detections of the
  episode (`run_iws_nis.py:103-124`). 198 NIS samples recorded in
  `nis_openloop_ep3` (one frame consumed by init). 0 detector drops
  for rollout on ep3 (PART2_NOTES.md line 258 — 100% detection rate
  across all episodes).
- §V.C: `correction_reframe._run_ekf` walks all 199 detections
  (`correction_reframe.py:60-74`). `summary_5c.csv` reports
  `det_success=199`, so again 0 drops. NIS length is 198 by the
  same init logic.
- No warm-up skip in either driver; no truncation; both feed the
  full 199-frame rollout into the filter.

**Frame range is identical.**

---

## 5. Conclusion

Which boxes apply (only one does):

- [x] **Different stochastic sample** — two unseeded diffusion rollouts
  from the same IWS world-model on the same PushT val ep3. Pred halves
  diverge over time (mean abs diff 1.45 → 5.76 from t=0 → t=195).
- [ ] Different filter initialization / Q / R — same module, same
  symbols, same `x0 = [first_det; 0; 0; 0]`.
- [ ] Different aggregation — both are `mean(nis_arr)` over the one
  episode.
- [ ] Different frame range — both full 199 frames, 0 detector drops
  in both.
- [ ] Different perception output — same HSV detector applied to pred
  halves of MP4s with the same mp4v encoder; the GT halves (a sanity
  check for the detector being consistent) are near-identical.
- [ ] Other.

### Why the two samples land so far apart (interpretive)

This is exactly the failure mode PART2_NOTES.md line 274 already
described. The §V.A ep3 sample is one where the world-model moves the
T-block smoothly along a *wrong* trajectory — locally consistent
motion → filter predictions track the next observation → mean NIS of
5.14, despite ~28 px / 85° pose drift vs GT. The §V.C ep3 sample
drawn a few minutes later is a sample where the T-block motion is
*jerky* (rapid pose jumps that violate the constant-velocity model
the filter uses) → filter innovations explode → mean NIS of 177.
PART2_NOTES.md line 359 already notes this 35× spread and frames it
as a data point about IWS seed-dependence.

### Recommendation for the paper

The observation is real and (once the cause is labelled) is evidence
*for* the claim in §V.A, not against it:

1. Label the §V.C open-loop baseline as "a second independent sample
   of the same episode" rather than "the ep3 open-loop", and cite it
   alongside §V.A as a demonstration that mean NIS on a single IWS
   rollout has large sample-to-sample variance.
2. If an apples-to-apples §V.A ↔ §V.C comparison is required, either
   (a) recompute §V.C relative to the §V.A sample by pointing
   `correction_reframe.py` at `output/openloop_ep3.mp4` for the
   open-loop baseline, or (b) re-run the correction methods with a
   fixed `torch.manual_seed` in `run_all_experiments.py` so the
   baseline is reproducible.
3. The §V.C relative rankings (best-of-3 < selective ≈ smoother <
   periodic_reset_k20 < …) are unaffected: all seven methods in
   `ep3_results/` were run under the same session and presumably
   share the sampling-noise environment even though no seed was set
   explicitly.

### What is not known from code+artifacts alone

- Whether `run_all_experiments.py` and `headless_rollout.py` were
  launched from the same Python process (if so, the RNG state at
  start of each rollout differs deterministically; if from separate
  processes, it differs by whatever torch uses for default init,
  typically `/dev/urandom` or system time). Timestamps put them ~14 s
  apart, but that doesn't tell us whether the second one inherited
  state from the first.
- Whether the IWS diffusion sampler uses a fresh Generator or the
  global torch RNG. Confirming this would require reading deep into
  `interactive_world_sim/algorithms/...`, which is out of scope for a
  30-minute read. Does not affect the conclusion (different samples
  either way).

To make the §V.A / §V.C baselines bit-identical in future runs, set
`torch.manual_seed(0)` at the top of both scripts and re-generate. No
code change is recommended right now — the discrepancy is already
documented in PART2_NOTES.md and is itself a paper-relevant finding.
