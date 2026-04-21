# Calibrating Innovation-Based Diagnostics for Action-Conditioned Visual World Models

ROB 530 (ROB 568) Final Project. University of Michigan, 2026.
**Authors:** Chaoyu Zhang, Peizhou Huang.

This repository studies what a classical Bayes filter (an EKF on SE(2))
can and cannot tell you about the frames produced by a learned visual
world model. The pipeline is

```
video frames  →  HSV pose tracker  →  EKF smoothing  →  trajectory + NIS
```

Our claim in one sentence: **the consistency test does not first fail at
world-model rollouts — it already fails once measurements come from a
real visual tracker; rollout generation adds a second, separable failure
on top.**

## Main result

| Regime                            | mean NIS | in-CI fraction | Reading           |
|-----------------------------------|---------:|---------------:|-------------------|
| Matched synthetic (MuJoCo, σ=10)  |     3.03 |           0.95 | calibrated        |
| Real tracker on MuJoCo renders    |     5.78 |           0.52 | tracker mismatch  |
| IWS GT-rendered frames            |     6.36 |           0.59 | larger mismatch   |
| IWS open-loop rollout             |    27.34 |           0.60 | most severe       |

Source CSVs: `results/synthetic/`, `results/real/`, `results/part2/summary_5a.csv`.

See `poster/poster.tex` for the full poster and
`EXPERIMENTAL_RESULTS.md` for the number-to-CSV map.

## Repository layout

```
part1/              Part 1:   synthetic / real MuJoCo EKF + RIEKF calibration
part1_6/            Part 1.6: blind-spot study on 4 synthetic perturbation families
part2/              Part 2:   IWS rollout diagnostics (§V.A open-loop, §V.B cross-metrics, §V.C corrections)
poster/             Conference poster (LaTeX, Beamer, umblue theme)

headless_rollout.py Produces an open-loop IWS rollout video + per-frame pred .npy.
run_all_experiments.py   Runs all 7 correction methods on IWS PushT episode 3.
latent_smoother.py, periodic_reset.py, selective_correction_v3.py
                    Individual correction-method implementations.
part1_mujoco_kf.py  Standalone Part 1 MuJoCo calibration entry point.
physics_metrics.py, optical_flow_metrics.py, check_*.py, visualize_color.py
                    Perception / metric helpers reused across parts.
pusht_scene.xml     MuJoCo PushT scene.

EXPERIMENTAL_RESULTS.md   Paper-number → CSV-column map (read this first).
PART1_NOTES.md            Part 1 design notes and acceptance gates.
PART1_6_NOTES.md          Part 1.6 blind-spot study notes.
PART2_NOTES.md            Part 2 inventory and §V.A/B/C running notes.
PART2B_INVENTORY.md       Part 2 data inventory.
PART2_DISCREPANCY.md      Why §V.A ep3 NIS=5.14 and §V.C open-loop NIS=177 differ.
GT_DISCREPANCY.md         Why §V.A GT NIS=6.36 and §V.C GT NIS=11.3 differ.

results/                  Paper artifacts (CSV, TEX, PNG/PDF figures). Small and checked in.
  synthetic/              Part 1 matched-noise (fig2_nis, fig3_rmse_vs_sigma, CSV)
  real/                   Part 1 real HSV tracker (same structure)
  part1_6/                blind-spot CSVs + figures
  part2/                  summary_5a/5b/5c CSVs, figures/, table_*.tex
  fig2b_nis_histograms.{png,pdf}    Part 1 three-panel NIS histograms (poster Fig 1)

output/                   (gitignored) IWS rollout MP4s + per-frame _pred.npy (~234 MB).
                          Regenerate with run_all_experiments.py / headless_rollout.py.
results/part1_cache/      (gitignored) Part 1 detector cache.
results/part2_cache/      (gitignored) Part 2 frame cache.
paper_package/            (gitignored) Archived submission snapshot (~264 MB).
interactive_world_sim/    (gitignored) Upstream IWS clone. See setup below.
venv/                     (gitignored) Python virtualenv.
```

## Setup

### Analysis-only (reproduces every number in `results/*.csv`)

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The Part 1 synthetic and real experiments use a local MuJoCo scene and
run end-to-end. Part 2 reads cached IWS rollouts from `output/`; those
artifacts are gitignored — see below to regenerate.

### Regenerate IWS rollouts (optional, GPU recommended)

```bash
pip install -r requirements-iws.txt

# Clone the upstream Interactive World Sim repo alongside this project
# (do NOT nest it inside, keep it as a sibling or add as a submodule):
git clone https://github.com/real-stanford/interactive_world_sim.git
pip install -e interactive_world_sim/

# Download checkpoints and mini data (script ships with IWS):
bash interactive_world_sim/scripts/download_checkpoints.sh
bash interactive_world_sim/scripts/download_mini_data.sh

# Regenerate the five open-loop rollouts for §V.A:
for i in 0 1 2 3 4; do python headless_rollout.py --episode $i; done

# Regenerate all seven correction methods for §V.C:
python run_all_experiments.py
```

## Reproduce the paper numbers

```bash
# Part 1: synthetic + real MuJoCo NIS calibration (sigma sweep)
python -m part1.run_part1 --config synthetic
python -m part1.run_part1 --config real

# Part 1.6: blind-spot perturbation study
python -m part1_6.run_part1_6

# Part 2, §V.A: IWS open-loop NIS for ekf_cart + riekf × {gt, rollout}
python -m part2.run_iws_nis

# Part 2, §V.B: cross-metric discriminative analysis
python -m part2.metric_comparison

# Part 2, §V.C: correction-method reframing on episode 3
python -m part2.correction_reframe

# Generate final LaTeX tables
python -m part2.make_tables
```

## Build the poster

```bash
cd poster
latexmk -pdf poster.tex
# or: pdflatex poster.tex  (twice)
```

Expected output: `poster.pdf`, 122 × 91 cm, three-column Beamer
`umblue` theme. Both figures (`figs/fig1_part1_nis_histograms.png`,
`figs/fig3b_discriminative_auc.png`) are copies of
`results/fig2b_nis_histograms.png` and
`results/part2/figures/fig_discriminative.png`.

## Acknowledgments

- The Interactive World Sim (IWS) world-model checkpoints and the
  PushT mini dataset are from the upstream Columbia/TRI repository
  ([real-stanford/interactive_world_sim](https://github.com/real-stanford/interactive_world_sim)).
  This repository does not redistribute them — see the setup section
  to download from source.
- MuJoCo PushT scene adapted from the Diffusion Policy / LeRobot
  community scenes.
- ROB 530 / ROB 568 teaching staff for the original linear-filtering
  framing that Part 1 starts from.

## License

MIT — see `LICENSE`.
