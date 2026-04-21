# §V.A vs §V.C GT-NIS alignment (6.36 vs 11.3)

Read-only comparison of the two GT-side NIS numbers cited in the
Part 2 paper.

## 1. §V.A GT-rendered frames, NIS = 6.36

- CSV row: `results/part2/summary_5a.csv` **line 2** —
  `pusht_iws, gt, ekf_cart, mean_nis=6.357..., n_episodes=5,
  n_steps=990`.
- Coverage: **5 episodes pooled** (`openloop_ep0..ep4`), 5×198 = 990
  filter steps concatenated, then `np.mean` over the full pool.
- Source frames: GT (right) half of each episode's own MP4, i.e.
  `output/openloop_ep{0..4}.mp4`.
- Per-episode breakdown (from
  `results/part2/nis/pusht_iws__gt__ekf_cart.npz`):

| Episode | n  | mean NIS |
|---------|----|---------:|
| ep0     | 198 |   0.88  |
| ep1     | 198 |   9.10  |
| ep2     | 198 |   3.81  |
| ep3     | 198 |  11.13  |
| ep4     | 198 |   6.87  |
| **pooled (990 steps)** | | **6.357** ← matches CSV |

## 2. §V.C GT reference, NIS = 11.25 (≈ "11.3")

- CSV row: `results/part2/summary_5c.csv` **line 2** —
  `pusht_iws, GT_reference, n_frames=199, det_success=199,
  mean_nis=11.252...`.
- Coverage: **ep3 only** (single episode). See
  `part2/correction_reframe.py:84` —
  `gt_ep = iws_io.load_episode("open-loop", "gt")`, which pulls the
  GT (right) half of `output/ep3_results/open-loop.mp4`. All seven
  correction methods share that one GT reference.
- Filter run over all 199 frames → 198 NIS samples, aggregated as
  `nis_stats["mean_nis"] = np.mean`.
- **This number is the ep3-only GT-side mean.** The §V.A
  per-episode value for ep3 on the GT side is 11.13 (from the same
  underlying source frames). The tiny 11.25 vs 11.13 gap comes from
  using two different MP4 encodes of the same source: the right
  halves of `openloop_ep3.mp4` and `ep3_results/open-loop.mp4`
  differ by mean abs 0.009 / max 18 (mp4v codec jitter only — see
  PART2_DISCREPANCY.md §1c). That codec noise propagates through the
  HSV detector into ~1% relative drift in mean NIS.

## 3. Are the two GT pipelines bit-identical?

**No — same filter, same detector, same frame range per episode, but
different episode coverage and (for ep3) a second MP4 encode of the
source.**

| Knob | §V.A GT row | §V.C GT_reference row | Identical? |
|------|-------------|-----------------------|:----------:|
| Filter class | `EKF_SE2_Cartesian` (part1) | same | ✓ |
| `DT` | `1/15 s` | `1/15 s` | ✓ |
| `Q` | `Q_EKF_PX` (imported from `run_iws_nis.py`) | `Q_EKF_PX` (same import) | ✓ |
| `R` | `R_PX` = `diag(100, 100, 5e-4)` | `R_PX` same | ✓ |
| `P0` | `P0_EKF_PX` = `diag(1, 1, 1e-2, 100, 100, 1)` | same | ✓ |
| `x0` init | `zeros(6); x0[:3] = first_det` | same (`correction_reframe.py:64-66`) | ✓ |
| Detector | `part2.perception_iws.detect_episode` (HSV) | same function | ✓ |
| Frame range per episode | all 199 frames → 198 NIS | all 199 frames → 198 NIS | ✓ |
| Detector drops | 0/990 (CSV `drop_rate=0.0`) | 0/199 (CSV `det_success=199`) | ✓ |
| Aggregation | `mean` over pooled pool | `mean` over single ep | **different granularity** |
| Episode coverage | **5 episodes (ep0–ep4) pooled** | **1 episode (ep3 only)** | **different** |
| GT frame source (for ep3) | right half of `openloop_ep3.mp4` | right half of `ep3_results/open-loop.mp4` | **different MP4 encode** (same underlying HDF5 frames) |

### Where the 6.36 vs 11.3 gap comes from (decomposed)

1. **Episode coverage (dominant):** §V.A averages across 5
   episodes; §V.C uses only ep3. Ep3 happens to have the highest
   per-episode GT NIS (11.13) of the five. If you restrict §V.A to
   ep3 only, the GT NIS is 11.13 — essentially the same as §V.C's
   11.25. Conversely, if you had picked ep0 as the single episode,
   GT NIS would be 0.88.
2. **MP4 re-encode (negligible):** ~1% drift (11.13 → 11.25) from
   the fact that §V.C reads the GT half from a different MP4 file.
   Codec noise only.
3. **Filter/Q/R/P0/detector/frame-count:** bit-identical. Cannot
   contribute any difference.

### Implication for the paper

The two GT numbers are **not** two measurements of the same quantity
— one is a 5-episode pool, the other is ep3 only. Either (a) rename
the §V.C row to "GT reference (ep3)" and cite the per-episode value
11.13 from §V.A alongside it for consistency, or (b) recompute a
5-episode GT pool for §V.C (requires the corrections on all 5
episodes, which we do not have on disk — only ep3 has the full
method set, per PART2_NOTES.md Q2). Option (a) is the honest path
and requires no new experiments.
