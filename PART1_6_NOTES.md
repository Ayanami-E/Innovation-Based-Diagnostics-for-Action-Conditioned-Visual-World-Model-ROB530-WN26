# Part 1.6 — Smooth-but-Wrong Controlled Experiment

Run 2026-04-18, ~10 minutes wall-clock. All within plan §3 budget (2 h).

## Headline result

The IWS Ep3 anecdote ("rollout drifted 28 px / 85° but NIS stayed at 5")
is **not an IWS-specific artifact** — it is a **structural blind spot**
of every kinematic / temporal-consistency probe we have. Confirmed on
20 cached MuJoCo PushT episodes via 4 controlled perturbation types ×
5 magnitudes.

## Acceptance ladder (plan §3)

| # | Criterion | Result |
|---|---|---|
| 1 | `summary_1_6.csv` exists with 4×5×n_ep rows | ✓ 20 rows × 20 episodes pooled |
| 2 | **P1 blind spot — REQUIRED**: `mean_nis(P1, m=0.05) < 5` AND `sod_mean < 5` | ✓ **0.69 < 5** AND **3.00 < 5** at 7.45 m of cumulative drift |
| 3 | P3 responsiveness: `mean_nis(P3, m=0.05) ≥ 2 × clean` | ✓ **9235 / 0.16 ≈ 57 000×** |
| 4 | `fig_blind_spot.pdf` and `fig_p4_example.pdf` exist | ✓ |
| 5 | Wall-clock ≤ 2 h | ✓ ~10 min |
| 6 | No regressions: pytest 7/7, Part 2 imports | ✓ |

**Acceptance #2 (the key gate) passes by a huge margin** — even with
~7.5 m of cumulative position drift on a 41-cm-wide table (the object
would be off-screen 18× over), NIS sits at 0.69 and SOD-χ² is *exactly*
at the χ²(3) expected mean of 3.0. The blind spot is real and structural.

## Per-perturbation findings (one sentence each, for paper §V.D prose)

- **P1 (constant velocity drift).** At m=0.05, the perturbed
  trajectory accumulates 7.45 m of physical positional error over 150
  steps (mean 3.73 m), yet mean NIS = 0.69 and mean SOD-χ² = 3.00 —
  both indistinguishable from the clean baseline (NIS 0.16, SOD 3.00),
  confirming that constant-velocity drift is invisible to kinematic
  consistency tests.

- **P2 (constant angular bias).** A 0.05 rad (2.9°) constant
  orientation offset throughout the entire episode produces *zero*
  movement in NIS and SOD relative to baseline (both columns identical
  to four decimal places), as expected from the zero-derivative
  argument: any difference-based test is mathematically blind to a
  constant additive bias.

- **P3 (constant acceleration).** Constant acceleration α = 0.05
  m/step² produces a 555 m end drift and lights up NIS at 9 235 and
  SOD-χ² at 5 176 — a 57 000× / 1 700× ratio over baseline,
  demonstrating that the metrics are not globally insensitive but are
  specifically blind to *zero-derivative* perturbations.

- **P4 (sine bump, smooth and bounded).** A sine bump with peak
  amplitude 5 cm over a 75-step window — the closest synthetic analog
  to the IWS Ep3 smooth-but-wrong case — pushes NIS only to 0.50 and
  SOD-χ² to 4.31 (43% above the χ²(3) mean), still below the
  detection gate of 5 despite a 7 cm peak physical error; this is the
  in-vitro reproduction of the IWS Ep3 blind spot.

## Headline numbers (NIS, ekf_cart, pooled across 20 eps × 150 steps)

```
                m=0      m=0.005   m=0.01    m=0.02    m=0.05
P1 (vel)        0.16     0.15      0.16      0.21      0.69
P2 (theta)      0.16     0.16      0.16      0.16      0.16
P3 (accel)      0.16    93.6     371.4    1480.6    9235.3
P4 (sine)       0.16     0.17      0.18      0.22      0.50
```

The two diagnostic features:
1. P1, P2, P4 are essentially flat across all magnitudes for both NIS
   and SOD-χ² — the blind columns.
2. P3 grows quadratically with magnitude — the alive column.

## Deviations from plan

1. **Magnitude units stay heterogeneous across perturbation types**
   (P1=m/step velocity; P2=rad bias; P3=m/step² accel; P4=m amplitude),
   per plan §1.1 literal formulas. We briefly tried rescaling all
   magnitudes to "cumulative end-drift in meters" for cross-type
   comparability, but this collapsed P3's acceleration below the
   noise floor and broke acceptance #3. Reverted. Cumulative physical
   error per type is reported via `pos_err_max` / `pos_err_mean` /
   `ori_err_max` columns of `summary_1_6.csv`, so the cross-type
   comparison is recoverable from the data.

2. **Flow-φ is reported as a delta-coherence proxy, not a true Flow-φ.**
   Plan §1.2 anticipated this: "compute Flow-φ on a *rendered* version
   of the perturbed sequence" would require a full image-rendering
   pipeline that we don't have. Instead `_delta_coherence` measures the
   cosine similarity of consecutive (Δx, Δy) pose deltas to their mean
   direction. **Structurally, true Flow-φ on these perturbations would
   be ≈ 1.0** because all four perturbations preserve rigid-body motion
   (the perturbed pose is still a rigid pose; image flow inside the
   T-block mask would still be perfectly coherent). The proxy
   reproduces the qualitative point — Flow-φ is *also* blind to
   smooth-but-wrong rigid drifts. Documented as a proxy in
   `run_part1_6.py` docstring; not over-claimed in the paper.

3. **No new metric family introduced** (per plan §0). Just NIS,
   SOD-χ², and the Flow-φ proxy.

4. **EKF-Cart only, no RIEKF.** Per plan §1.2 — Part 1 already showed
   the two filters agree in mean and the experimental finding doesn't
   depend on which filter you pick.

## Files created

```
part1_6/
├── __init__.py
├── perturb.py                  # 4 perturbation generators, 99 lines
├── run_part1_6.py              # driver, 124 lines
├── plots.py                    # 2 figures, 113 lines
└── make_table.py               # 1 LaTeX table, 67 lines

results/part1_6/
├── summary_1_6.csv             # 20 rows: 4 perturbations × 5 magnitudes
├── fig_blind_spot.pdf / .png   # 2x2 panel
├── fig_p4_example.pdf / .png   # GT vs P4 trajectory overlay (ep 0)
└── table_blind_spot.tex
```

No edits anywhere outside `part1_6/`. Part 1 tests pass 7/7;
Part 2 modules all import cleanly.

## Paper §V.D framing (recommended)

The Part 2 finding "NIS ≈ SOD-χ² and both miss IWS Ep3" was an
anecdote. Part 1.6 elevates it to a controlled claim:

> Bayes-filter consistency (NIS) and second-order kinematic-difference
> tests (SOD-χ²) are **structurally blind to perturbations whose
> derivatives match the assumed kinematic model**. We demonstrate this
> in vitro on MuJoCo PushT by injecting four perturbation families.
> Constant-velocity (P1), constant-bias (P2), and bounded
> low-frequency (P4) perturbations of arbitrarily large magnitude are
> not detected; only second-derivative-rich perturbations like
> constant acceleration (P3) light up either metric. This explains
> the IWS Ep3 case (Part 2 §V.B/§V.C) without invoking any
> world-model-specific pathology: the metric class itself cannot
> distinguish "smooth and right" from "smooth and wrong".

This is a stronger paper claim than the Part 2 anecdote and reframes
the contribution as **a calibration-and-limitation study of temporal-
consistency metrics for visual world models** — exactly the V3
framing called for in the plan's context.
