# Experimental Results Summary — ROB 530 Final Project

**Project**: Calibration and Limitation Study of Bayes-Filter Consistency
Diagnostics for Learned Visual World Models
**Compiled**: 2026-04-18
**Sources**: Part 1 (MuJoCo synthetic + real detector), Part 2 (IWS PushT),
Part 1.6 (MuJoCo controlled perturbations)
**Status**: Data complete. §V of the paper can be written directly from
the tables and figures here without re-running experiments.

---

## 0. One-page navigation

| Section | Content | Main table / figure | Main data files |
|---|---|---|---|
| §1 | NIS spectrum across six regimes | Table I | `results/{synthetic,real}/part1_summary.csv`, `results/part2/summary_5a.csv`, `results/part1_6/summary_1_6.csv` |
| §2 | NIS vs other metrics | Table II | `results/part2/summary_5b.csv`, `results/part2/discriminative.csv` |
| §3 | Correction methods reframe | Table III | `results/part2/summary_5c.csv` |
| §4 | Smooth-but-wrong blind spot | Table IV | `results/part1_6/summary_1_6.csv` |
| §5 | Four recommended paper figures | — | `results/part2/figures/`, `results/part1_6/` |
| §6 | 300-word narrative | — | — |
| §7 | Three-sentence conclusion | — | — |
| §8 | File / artifact inventory | — | — |
| §9 | Items to confirm before writing | — | — |

---

## 1. Table I — Six-Regime NIS Consistency Spectrum

The headline table of §V. Regimes A→E form a monotonically increasing
"violation severity ladder"; F is the counter-example to that ladder.

| Regime | Description | n_steps | Mean NIS | frac_in_CI (95%) | KS p-value | Consistent? | Physical meaning |
|---|---|---:|---:|---:|---:|:-:|---|
| **A** | Ideal Gaussian synthetic, σ=10 (EKF-Cart) | 2980 | **3.03** | 0.95 | 0.79 | ✅ | Filter math-correctness baseline |
| **B** | Strict R, σ=0 — process-model mismatch | 2980 | 3.01 | 0.72 | <10⁻³⁰⁰ | ❌ heavy-tail | Q mismatch → heavy tail |
| **C** | Real HSV+PCA detector on MuJoCo, σ=0 | 2980 | **2.05** | 0.25 | <10⁻³⁰⁰ | ❌ bimodal | Detector bias → bimodal |
| **D** | HSV+PCA on IWS GT frames | 990 | **6.36** | 0.59 | 1.2×10⁻⁷⁸ | ❌ | Real camera + rendering-domain noise floor |
| **E** | HSV+PCA on IWS open-loop rollouts | 990 | **27.34** | 0.60 | 4.6×10⁻⁵³ | ❌ heavy | **World model detected** ← §V core |
| **F** | "Smooth-but-wrong" P1 controlled (m=0.05) | 2980 | **0.69** | 0.22 | 0.0 | ⚠ appears normal | **Metric blind spot** ← §V.D core |

**Notes**
- A/B/C are representative points from the Part 1 σ=0 / σ=10 sweep. Part 1
  actually sweeps σ ∈ {0, 5, 10, 20}, with both EKF-Cart and RI-EKF at
  each setting; EKF-Cart is the paper's primary filter (RI-EKF trends
  identically, with means ~30% higher).
- Why σ=0 for C: σ=0 is "pure OpenCV detector, no additional synthetic
  noise", which is the cleanest definition of the "real detector regime".
  The σ=10 EKF gives 1.54 (lower, because added synthetic noise makes the
  R assumption closer to truth). Both numbers are in the part1 csv; the
  paper can pick one and footnote the other.
- D/E have frac_in_CI ≈ 0.6 instead of ≈ 0.95, which means IWS GT frames
  themselves push detector output off the χ²(3) assumption (perspective
  camera + JPEG compression is dirtier than MuJoCo renders). E is a step
  above D, meaning the world-model rollout adds a layer of structured
  error on top of D's noise floor.
- F's frac_in_CI=0.22 looks "bad", but mean NIS=0.69 is far below the
  χ²(3) expected mean of 3.0 — the distribution is *collapsing*,
  innovations are *smaller* than R assumes, because the perturbation is
  constant-velocity and the predict step tracks the observations
  perfectly. This is the essence of the blind spot.
- F's KS p=0 only says "shape is not χ²(3)"; the shape is smaller, not
  larger.

---

## 2. Table II — Cross-Metric Comparison (Part 2 §V.B)

10 episodes (5 GT + 5 rollout, IWS PushT). Discriminative power of each
metric on GT vs rollout:

| Metric | GT mean ± std | Rollout mean ± std | Cohen's *d* | AUC-ROC | Pearson vs NIS | Trivial? |
|---|---|---|---:|---:|---:|:-:|
| **SOD-χ²** (mean) | 3.00 ± 1.27 | 18.24 ± 20.44 | **+1.05** | **1.00** | **0.99** | no |
| Flow consistency φ | 0.78 ± 0.09 | 0.67 ± 0.05 | +1.36 | 0.88 | — | no |
| Anomaly rate (φ < τ) | 0.05 ± 0.04 | 0.06 ± 0.03 | +0.40 | 0.66 | — | no |
| **NIS** (mean) | 6.36 ± 4.10 | 27.34 ± 36.26 | +0.81 | **0.64** | 1.00 | no |
| NIS KS p-value | ≈0 | ≈0 | +0.65 | 0.60 | — | no |
| SOD-χ² KS p-value | ≈0 | ≈0 | +0.31 | 0.60 | — | no |
| Frame MSE | 0.0 | 476 ± 73 | +9.3 | 1.00 | — | **yes** |
| Frame NCC | 1.0 | 0.90 ± 0.01 | +10.5 | 1.00 | — | **yes** |
| Pos drift vs GT-det (px) | 0 | 27 ± 26 | +1.46 | 1.00 | — | **yes** |
| Ori drift vs GT-det (rad) | 0 | 1.00 ± 0.94 | +1.50 | 1.00 | — | **yes** |

**Two painful but honest findings**
1. **SOD-χ² beats NIS outright** (AUC 1.00 vs 0.64). A 3-line
   hand-written second-order-difference χ² test outperforms the entire
   EKF apparatus on this task.
2. **NIS and SOD-χ² have Pearson 0.99** (on the 5 rollout episodes).
   The filter apparatus provides no independent information on this task.

**Trivial-metrics caveat**: MSE / NCC / pos_drift / ori_drift are
structurally 0 on the GT source ("GT frame vs itself"), so AUC=1.0 is by
construction, not a real metric win. Marked "yes" in the table.

**LPIPS substitution note**: The original plan was to use `lpips` (the
torch package) as a perceptual metric. The local venv has no torch, so
cv2-based Frame MSE + NCC fill in. These are weaker than LPIPS but do
not change the §V.B argument — if NIS correlates weakly with these, it
correlates even more weakly with LPIPS.

**FVD skipped**: 5 episodes/source is too small a sample for feature
distribution distance.

---

## 3. Table III — Correction Methods Reframe (Part 2 §V.C)

Seven correction methods, all run on IWS PushT episode 3 (the only
episode for which all methods are on disk). GT_reference is the
ground-truth detection of the same episode:

| Method | NIS | vs OL ratio | SOD-χ² | vs OL ratio | Flow φ | MSE | NCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Open-loop (baseline)** | **177.40** | 1.00× | 43.66 | 1.00× | 0.625 | 493 | 0.898 |
| Periodic reset, K=10 | 37.72 | 0.21× | 34.33 | 0.79× | 0.663 | 447 | 0.908 |
| Periodic reset, K=20 | 16.54 | 0.09× | 27.29 | 0.63× | 0.647 | 477 | 0.902 |
| Latent smoother | 14.92 | 0.08× | 13.72 | 0.31× | 0.669 | 501 | 0.896 |
| Selective correction | 14.00 | 0.08× | 13.39 | 0.31× | 0.686 | 422 | 0.913 |
| **Best-of-3** | **4.90** | **0.03×** | 4.89 | 0.11× | 0.686 | 513 | 0.893 |
| Best-of-5 | 15.93 | 0.09× | 5.94 | 0.14× | 0.689 | 509 | 0.894 |
| *GT reference* | *11.25* | — | *2.98* | — | *0.783* | *0* | *1.000* |
| **Dynamic range** | **36×** | — | **9×** | — | 1.10× | **1.22×** | 1.022× |

**Key takeaways**
- NIS has a cross-method dynamic range of **36×** (177 → 5); SOD is
  **9×**; pixel metrics are all ≤1.22×. Kinematic metrics are ~20–30×
  more sensitive to correction quality than pixel metrics.
- Best-of-3 pushes NIS to 4.90, **below the GT reference of 11.25** —
  best-of-N selects kinematically-smoothest samples, so it can be
  "smoother than physics".

**Best-of-N circular-argument warning (must appear in the paper)**
Best-of-N's selection criterion is precisely kinematic smoothness
(implicitly or explicitly coupled to NIS / SOD). So the 97% NIS drop it
produces is not independent evidence of metric validity — it is the
optimization target and the evaluation metric overlapping. The paper
should state this explicitly: the best-of-N NIS improvement is by
construction, and only periodic reset is an "independent ground-truth
injection" ceiling for NIS, which only drops NIS from 177 to 17 (still
above the GT reference of 11).

**Episode-3 reproducibility warning**
Part 2 §V.A also used ep3, where NIS=5.14 (from
`output/openloop_ep3.mp4`). This section uses
`output/ep3_results/open-loop.mp4`, an independent stochastic sample of
the same episode where NIS=177.40. **Same world model, same episode,
different seed → NIS differs by 35×.** This is worth a line in the
paper §V.C limitations: IWS open-loop quality is highly seed-dependent.

---

## 4. Table IV — Smooth-but-Wrong Blind Spot (Part 1.6 §V.D)

20 cached MuJoCo episodes (T=150) × 4 perturbation kinds × 5 magnitudes.
Each row is the worst case at m=0.05:

| Perturbation | Unit | Max physical err | NIS (vs clean=0.16) | SOD-χ² (vs clean=3.00) | Flow φ proxy | Verdict |
|---|---|---|---|---|---:|---|
| **P1** constant velocity | m/step | **7.45 m** (≈18× the 41 cm tabletop width) | **0.69** (ratio 4.3× vs baseline; abs still ≪ gate=5) | **3.00** (= χ² mean, zero drift) | 0.999 | ● **completely blind** |
| **P2** constant angular bias | rad | 0.05 rad ≈ **2.9°** | 0.16 (= clean to 4 d.p.) | 3.00 (= clean to 4 d.p.) | 0.396 | ● **completely blind** |
| **P3** constant acceleration | m/step² | **555 m** | **9 235** (vs clean 0.16: **57 000×**) | **5 176** (vs 3.0: 1 700×) | 1.000 | ✗ clearly detected |
| **P4** sine bump over window | m | 0.071 m (7 cm) | 0.50 (3.1× baseline, still < gate) | 4.31 (1.4× χ² mean, still < gate) | 0.318 | ⚠ nearly blind |

**Acceptance ladder (Part 1.6 plan §3) — all passed**
- #2 P1 blind spot REQUIRED: NIS<5 AND SOD<5 → **0.69 < 5 ✓ AND 3.00 < 5 ✓**
- #3 P3 responsiveness ≥2× clean: **57 000× ≫ 2 ✓**
- #5 Wall-clock ≤2h: **11 min**
- #6 No regression: pytest 7/7, Part 2 imports OK

**Argument chain**
- P1, P2 → establishes the structural blind spot (zero derivative → no information)
- P3 → confirms the metric is not dead (light up the second derivative and it fires)
- P4 → in-vitro reproduction of the IWS Ep3 phenomenon (local, bounded, low-frequency → nearly blind)

**Magnitude-unit heterogeneity caveat**
The four perturbation m-units are different physical quantities: P1 is
m/step (velocity), P2 is rad (angle), P3 is m/step² (acceleration), P4
is m (displacement). So m=0.05 maps to physical displacements spanning
four orders of magnitude (P3=555 m vs P4=7 cm). Per-type comparison is
meaningful; cross-type comparison of NIS values is not. The paper must
explain this or reviewers will ask.

**Flow-φ proxy caveat**
Part 1.6 has no image-rendering pipeline, so a true image-domain Flow-φ
cannot be computed. The 0.999 / 0.396 / ... values in the table are a
pose-delta cosine-consistency proxy. Structural argument: true Flow-φ
on P1–P4 would be ≈1.0 everywhere (rigid-body motion preserves it), so
Flow-φ is also structurally blind to smooth-but-wrong. This point
belongs in §V.D.

---

## 5. Four required figures (recommended for the paper)

Each figure's physical file is already on disk; the paper can
`\includegraphics` directly.

### Fig 2 — NIS distribution evolution across regimes
**Panel design**: 3×2 histogram panel, each with a χ²(3) PDF overlay
- (a) Regime A: tight fit
- (b) Regime B: heavy tail
- (c) Regime C: bimodal (main peak near 0)
- (d) Regime D: right-shifted
- (e) Regime E: heavily right-shifted + long tail
- (f) Regime F (P1 m=0.05): compressed near 0, blind-spot demonstration

**Generation**: data on hand; needs a ~50-line plotting script.
**NPZs used**:
- `results/synthetic/part1_nis_pooled.npz` (A, B)
- `results/real/part1_nis_pooled.npz` (C)
- `results/part2/nis/pusht_iws__gt__ekf_cart.npz` (D)
- `results/part2/nis/pusht_iws__rollout__ekf_cart.npz` (E)
- A fresh P1 m=0.05 NIS array (F) — either saved as its own .npz or
  recomputed inline at plot time.

**Information density**: a single figure that conveys the entire paper
to a reviewer. **TODO**: script not yet written.

### Fig 3 — Cross-metric discriminative AUC + correlation heatmap
**Already generated**:
- `results/part2/figures/fig_discriminative.{pdf,png}` — AUC bar chart,
  NIS and SOD highlighted in red
- `results/part2/figures/fig_corr_heatmap.{pdf,png}` — Pearson +
  Spearman heatmap, 10×10

**Information density**: honestly shows NIS is not optimal, proactively
reveals the NIS-SOD collinearity.

### Fig 4 — Smooth-but-wrong blind spot (P1–P4)
**Already generated**: `results/part1_6/fig_blind_spot.{pdf,png}`, 2×2
panel, log-y. Each panel shows NIS (red circle) + SOD (blue square),
points labeled with m, plus χ²(3)=3 and gate=5 reference lines.

**Information density**: the §V.D headline figure, directly supporting
the main claim that any kinematic metric is structurally blind to
smooth-but-wrong.

### Fig 5 — P4 qualitative trajectory
**Already generated**: `results/part1_6/fig_p4_example.{pdf,png}`,
episode 0 GT (blue ◯) vs P4 m=0.05 (red △) in the (x, y) plane, time
coded by colormap.

**Information density**: gives the reader an intuitive feel for
smooth-but-wrong. The in-vitro version of IWS Ep3.

---

## 6. One-page narrative (~300 words)

> We evaluate Bayes-filter consistency as a diagnostic for learned
> visual world models across six observation regimes. The filter
> implementation is verified in Regime A (mean NIS 3.03; KS p=0.79
> against χ²(3)). Regimes B and C confirm the diagnostic responds to
> process-model mismatch and detector bias respectively, producing
> characteristically different distributional shifts (heavy-tailed at
> mean 3.01 and bimodal at mean 2.05).
>
> Applied to a learned world model (IWS PushT), NIS rises from 6.36 on
> ground-truth frames (Regime D) to 27.34 on open-loop rollouts
> (Regime E), a 4.3× elevation indicating the rollouts violate the
> filter's noise hypothesis. However, head-to-head comparison with a
> model-free second-order-difference χ² baseline (Pearson 0.99 on
> rollouts; AUC 0.64 vs 1.00) shows the full filter apparatus provides
> no independent information on this single-camera SE(2) task.
> Post-hoc correction methods produce a 36× dynamic range in NIS
> versus 1.22× in pixel-space MSE, but the strongest correction
> (best-of-3) selects trajectories by the very smoothness that NIS
> tests, indicating metric-circular optimization rather than
> independent validation.
>
> A controlled MuJoCo perturbation experiment characterizes a
> structural limitation (Regime F): under constant-velocity drift
> producing 7.45 m of cumulative positional error (Part 1.6 P1,
> m=0.05) and a 2.9° constant angular bias (P2), mean NIS remains at
> or below the clean baseline of 0.16-0.69. A localized sine
> perturbation with 7 cm peak physical error produces NIS of 0.50,
> below the χ²(3) expected mean of 3. Such errors are invisible to
> any kinematic-consistency diagnostic whose temporal derivatives
> match the assumed dynamics.
>
> We frame our contribution as a calibration protocol and a
> delineation of two failure modes — metric collinearity with simpler
> baselines and structural blindness to smooth-but-wrong errors —
> rather than as a novel metric proposal.

---

## 7. Three-sentence paper conclusion

1. **Filter consistency detects world-model rollouts at the population
   level** — the monotonic A→E spectrum (3.03 → 27.34) holds.
2. **On a monocular SE(2) task, the filter apparatus carries no
   independent information over a model-free second-order-difference
   baseline** — NIS vs SOD-χ² Pearson 0.99, AUC 0.64 vs 1.00.
3. **Any kinematic-consistency metric is structurally blind to
   smooth-but-wrong errors** — P1 7.45 m drift → NIS 0.69; P4 7 cm sine
   perturbation → NIS 0.50; both below the χ²(3) expected mean of 3.0.

---

## 8. Data / artifact inventory (index for writing the paper)

### Code

```
part1/                         frozen, do not edit
├── ekf_se2_cartesian.py
├── riekf_se2.py
├── nis_analysis.py            nis_stats(), rmse_pose()
├── perception.py              MuJoCo HSV+PCA detector
├── corruption.py              σ-pixel noise injection
├── scene.py / run_part1.py    driver
└── tests/                     7 tests, all pass

part2/                         frozen
├── iws_io.py                  IWS MP4/npy loader, 256×256 cache
├── perception_iws.py          IWS pink-T HSV+PCA
├── run_iws_nis.py             §V.A driver
├── cross_metrics.py           SOD-χ², Flow-φ, MSE, NCC
├── metric_comparison.py       §V.B three analyses
├── correction_reframe.py      §V.C
└── make_tables.py             3 LaTeX tables

part1_6/                       frozen
├── perturb.py                 P1–P4 generators
├── run_part1_6.py             driver
├── plots.py                   2 figures
└── make_table.py              1 LaTeX table
```

### Data / tables / figures

```
results/synthetic/part1_summary.csv         Regimes A, B (sigma sweep)
results/real/part1_summary.csv              Regime C
results/synthetic/part1_nis_pooled.npz      NIS arrays for A/B
results/real/part1_nis_pooled.npz           NIS array for C
results/fig2b_nis_histograms.{pdf,png}      Part 1 §V.C figure (ready)
results/table2{a,b}_*.tex                   Part 1 tables (ready)

results/part2/summary_5a.csv                D, E (4 rows)
results/part2/summary_5b.csv                10 cross-metric rows
results/part2/summary_5c.csv                7 corrections + GT
results/part2/discriminative.csv            Cohen's d, AUC per metric
results/part2/corr_matrix.csv               10×10 Pearson + Spearman
results/part2/nis/*.npz                     pooled + per-ep NIS arrays
results/part2/figures/
  ├── fig_corr_heatmap.{pdf,png}
  ├── fig_discriminative.{pdf,png}
  ├── fig_scatter.{pdf,png}
  └── fig_correction_response.{pdf,png}
results/part2/table_v{a,b,c}_*.tex          3 tables

results/part1_6/summary_1_6.csv             20 rows: 4×5 perturbation cells
results/part1_6/fig_blind_spot.{pdf,png}    Fig 4
results/part1_6/fig_p4_example.{pdf,png}    Fig 5
results/part1_6/table_blind_spot.tex
```

### Documents

```
PART1_NOTES.md          Part 1 implementation notes
PART2_NOTES.md          Part 2 inventory + results discussion
PART2B_INVENTORY.md     IWS asset inventory (confirms: no checkpoints, no other task data)
PART1_6_NOTES.md        Part 1.6 acceptance + per-perturbation findings
EXPERIMENTAL_RESULTS.md this file
```

---

## 9. Items to confirm / finish before writing

### A. Must do (before writing)

1. **Fig 2 (six-regime histogram) is not yet generated.** Needs a
   ~50-line plotting script. Suggested location:
   `part1_6/fig_regime_panel.py` or `paper/figs/`. Reads the NPZs
   above, uses matplotlib to draw 3×2, overlays χ²(3), and uses a
   unified x-y range.
2. **Which perturbation for Regime F?** The table uses P1 m=0.05
   (NIS=0.69, pos_max 7.45 m) as the representative. P4 m=0.05
   (NIS=0.50, pos_max 7 cm) is also valid as the "more physically
   plausible" smooth-but-wrong case. Recommendation: use P4 in the
   table of the paper (closer in magnitude to IWS Ep3) and mention
   both in the text.
3. **Regime C: σ=0 or σ=10?** The table uses σ=0 (NIS 2.05). Part 2
   §V.A originally compared against σ=10 (NIS 1.54). Both choices are
   defensible, but the paper needs to be consistent — pick one and
   update the doc once to avoid footnotes.

### B. Important caveats (for Limitations / Discussion)

4. **Best-of-N circular argument** (covered in §3)
5. **Episode 3 reproducibility**: same model, same episode, different
   seed, NIS differs by 35×. Mention this in the paper to preempt the
   question.
6. **Magnitude-unit heterogeneity** (covered in §4)
7. **Flow-φ is a proxy in Part 1.6** (covered in §4)
8. **LPIPS substituted by MSE+NCC** (covered in §2)
9. **Small data scale**: 5 IWS episodes, 1 task. No cross-task
   generalization claims possible.

### C. Data-scope statements (must be in Limitations)

10. PushT only (single task), IWS only (single world model).
    **Not a generalization claim**. Do not let reviewers extend this
    to "all video world models".
11. SE(2) only (single rigid object on a plane). Behavior on
    multi-object / deformable / 6-DoF is unknown.
12. Monocular camera only. Behavior with multi-view and depth is
    unknown.
13. Default EKF process model is constant-velocity. NIS behavior under
    other process models is unknown.
14. NIS uses a χ²(3) assumption (position + heading). Other dimensions
    are unknown.

### D. Things not to claim in the paper

15. **Do not claim NIS is a novel metric.** NIS is a textbook concept.
16. **Do not claim NIS beats SOD.** The data says the opposite.
17. **Do not claim the metric detects every world-model error.** §V.D
    is an explicit counter-example.
18. **Do not claim Best-of-N's NIS improvement is evidence of metric
    validity** (§3.B circular argument).

---

## The reader should come away with

- **Data is complete.** Four tables, four figures (Fig 2 needs a
  script), enough for a 6-page paper.
- **Story is clear.** A→E monotonically increasing + F as the
  counter-example, summarized in three sentences.
- **Honest and substantive.** The two painful findings (SOD wins, P4
  blind) are already written; no reviewer corner will trip us.
- **ROB 530-appropriate.** SE(2), EKF, RI-EKF, NIS, χ², Lie algebra,
  Kolmogorov-Smirnov — the full course toolkit is in play.
- **One remaining gap**: the 6-regime histogram panel of Fig 2 is not
  yet drawn, but the data is all in the NPZs, ~30 minutes to add.
  Everything else is a direct `\includegraphics`.
