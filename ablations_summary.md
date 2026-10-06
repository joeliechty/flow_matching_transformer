# Ablation Study Summary

**Status (2026-10-06):** the SE(3) pose ablations are complete (6 variants × 5 seeds).
The MNIST ablations are set up but have not been trained yet.
All results were produced with the code at commit `cd6d937` on branch `dev/ablations`.

The study asks what two components add to the flow matching transformer:

- **OT**: minibatch optimal-transport pairing between noise samples and data samples during training.
- **CFG**: classifier-free guidance. Training randomly replaces the conditioning with a learned null token, and sampling extrapolates away from the unconditional prediction.

## Key findings

1. **OT makes few-step sampling work for unconditional models.** With a single integration step, the OT model lands
   0.95 from the nearest mode versus 7.02 without OT. Both converge by about 9 steps.
2. **OT makes conditional models worse, most likely because of how pairing is implemented.** Minibatch OT pairs noise and goal
   samples across the whole batch, ignoring each goal's condition. Without CFG, the OT model lands 0.85 ± 0.17 from
   its target mode versus 0.29 ± 0.01 without OT. It is also the only variant below 100% mode accuracy.
   The conditional OT numbers below reflect this flaw rather than OT itself
   (see [Known issues](#known-issues-and-caveats)).
3. **CFG doesn't help on this task.** The plain conditional model (no OT, no CFG) matches or beats every
   guided variant. The CFG-trained no-OT model does best at guidance 1–1.5, and the default guidance of 3.0 makes it worse.
4. **Guidance 3.0 breaks few-step sampling.** At 1 step, the CFG variants land 11.5–14.9 from their target. They
   only recover from about 9 steps.
5. **Mode accuracy saturates.** Every conditional variant except "OT, no CFG" scores 100% at 100 steps, so twist
   distance carries the comparison.

## Setup

**Task.** Generate SE(3) poses. Start poses are drawn from a unit Gaussian in twist space. Goal poses come from 4
tight modes (σ = 0.1 per twist dimension) at (5, ±5, ±5) with different rotations. Conditional models receive two
action tokens (up/down and left/right) that identify the target mode.

**Variants.**

| Variant | Conditional | OT pairing | CFG training |
| --- | :---: | :---: | :---: |
| cond · OT + CFG | ✓ | ✓ | ✓ |
| cond · OT, no CFG | ✓ | ✓ | |
| cond · no OT + CFG | ✓ | | ✓ |
| cond · no OT, no CFG | ✓ | | |
| uncond · OT | | ✓ | n/a |
| uncond · no OT | | | n/a |

**Training.**
- Schedule: 100 epochs × 100 batches × 128 samples, AdamW (learning rate 1e-4, weight decay 1e-5).
- Model: transformer with hidden size 128, 4 layers and 4 heads.
- Interpolation: 10 points per path.
- CFG: drops the whole conditioning with p = 0.1. Every conditional model also drops individual tokens with
  p = 0.1. Without CFG, that per-token dropout never blanks both tokens at once.
- Seeds: 1–5 for every variant, with all other hyperparameters identical.
- Hardware: one RTX 4090, 6 runs in parallel. Training started 2026-10-06 13:59 and finished 2026-10-06 14:31 (32 min for 30 runs); evaluation finished 2026-10-06 14:35 (4 min).

**Evaluation.**
- Samples: 256 per model. For conditional models that's 64 per mode, conditioned on that mode's action tokens.
- Sampling: Euler integration with 100 steps unless the steps are being swept.
- Noise: the sampling noise is seeded identically for every variant and sweep point. Spread across seeds therefore
  reflects training seeds only.
- Guidance: CFG-trained conditional variants use guidance 3.0, and no-CFG variants are sampled unguided (1.0).

**Metrics.**
- **Mode accuracy**: the fraction of conditional samples whose nearest mode (by twist distance) is the target mode.
- **Twist distance**: the mean twist-space distance from each sample to its target mode center, or to its nearest
  mode for unconditional models. Real goal samples score **0.219** (measured on 80,000 draws from the goal
  distribution). That is the floor: values below it mean samples cluster more tightly around the mode center than
  real data, which is collapse, not extra accuracy.
- **Mode coverage KL**: the KL divergence of the histogram of assigned modes from uniform. It is informative for
  unconditional models. For conditional models the evaluation batch is balanced by construction, so this KL only
  mirrors accuracy.

All tables report mean ± standard deviation over the 5 seeds.

## Results

### Main results (100 steps)

| Variant | Guidance scale | Mode accuracy ↑ | Twist distance (data: 0.219) | Mode coverage KL ↓ |
| --- | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 3.0 | 1.000 ± 0.000 | 0.425 ± 0.039 | 0.0000 ± 0.0000 |
| cond · OT, no CFG | 1.0 (unguided) | 0.989 ± 0.020 | 0.847 ± 0.172 | 0.0010 ± 0.0021 |
| cond · no OT + CFG | 3.0 | 1.000 ± 0.000 | 0.346 ± 0.034 | 0.0000 ± 0.0000 |
| cond · no OT, no CFG | 1.0 (unguided) | 1.000 ± 0.000 | 0.288 ± 0.011 | 0.0000 ± 0.0000 |
| uncond · OT | n/a | — | 0.288 ± 0.006 | 0.0083 ± 0.0031 |
| uncond · no OT | n/a | — | 0.278 ± 0.009 | 0.0061 ± 0.0052 |

- Without OT, both conditional variants hit 100% accuracy. The unguided model is the closest to real data at
  0.288, against a floor of 0.219.
- Conditional OT variants are worse on twist distance. "OT, no CFG" is also the least stable across seeds: seed 4
  scores 95.3% accuracy and a distance of 1.145 (see [Appendix](#appendix-per-seed-results-100-steps)).
- Both unconditional models cover all four modes about evenly (KL < 0.01) and land about as close as the best
  conditional model.

### Sampling-steps sweep

Figure: [experiments/results/pose/steps_sweep.png](experiments/results/pose/steps_sweep.png). It is generated
locally and gitignored, like all files under `experiments/results/`.

**Twist distance vs. integration steps** (real data: 0.219):

| Variant | 1 step | 2 steps | 3 steps | 5 steps | 9 steps | 20 steps | 50 steps | 100 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 11.54 ± 0.33 | 4.63 ± 0.17 | 2.19 ± 0.10 | 0.71 ± 0.03 | 0.35 ± 0.03 | 0.36 ± 0.03 | 0.41 ± 0.04 | 0.43 ± 0.04 |
| cond · OT, no CFG | 1.72 ± 0.18 | 1.11 ± 0.20 | 0.98 ± 0.19 | 0.91 ± 0.18 | 0.88 ± 0.17 | 0.86 ± 0.17 | 0.85 ± 0.17 | 0.85 ± 0.17 |
| cond · no OT + CFG | 14.88 ± 0.08 | 5.22 ± 0.10 | 2.23 ± 0.07 | 0.61 ± 0.07 | 0.35 ± 0.05 | 0.35 ± 0.04 | 0.34 ± 0.04 | 0.35 ± 0.03 |
| cond · no OT, no CFG | 0.23 ± 0.02 | 0.11 ± 0.01 | 0.12 ± 0.01 | 0.15 ± 0.02 | 0.19 ± 0.01 | 0.25 ± 0.01 | 0.28 ± 0.01 | 0.29 ± 0.01 |
| uncond · OT | 0.95 ± 0.04 | 0.53 ± 0.01 | 0.41 ± 0.00 | 0.33 ± 0.01 | 0.31 ± 0.01 | 0.29 ± 0.01 | 0.29 ± 0.01 | 0.29 ± 0.01 |
| uncond · no OT | 7.02 ± 0.05 | 3.84 ± 0.20 | 1.28 ± 0.06 | 0.46 ± 0.05 | 0.26 ± 0.03 | 0.26 ± 0.01 | 0.27 ± 0.01 | 0.28 ± 0.01 |

**Mode accuracy vs. integration steps** (conditional models):

| Variant | 1 step | 2 steps | 3 steps | 5 steps | 9 steps | 20 steps | 50 steps | 100 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 0.898 ± 0.013 | 0.995 ± 0.010 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| cond · OT, no CFG | 0.900 ± 0.021 | 0.974 ± 0.032 | 0.977 ± 0.030 | 0.979 ± 0.028 | 0.984 ± 0.029 | 0.985 ± 0.029 | 0.988 ± 0.024 | 0.989 ± 0.020 |
| cond · no OT + CFG | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| cond · no OT, no CFG | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

**Mode coverage KL vs. integration steps**:

| Variant | 1 step | 2 steps | 3 steps | 5 steps | 9 steps | 20 steps | 50 steps | 100 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 0.003 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| cond · OT, no CFG | 0.004 ± 0.004 | 0.002 ± 0.003 | 0.002 ± 0.003 | 0.001 ± 0.003 | 0.001 ± 0.003 | 0.001 ± 0.003 | 0.001 ± 0.002 | 0.001 ± 0.002 |
| cond · no OT + CFG | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| cond · no OT, no CFG | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| uncond · OT | 0.009 ± 0.004 | 0.008 ± 0.003 | 0.008 ± 0.003 | 0.008 ± 0.003 | 0.008 ± 0.003 | 0.008 ± 0.003 | 0.008 ± 0.003 | 0.008 ± 0.003 |
| uncond · no OT | 0.539 ± 0.139 | 0.085 ± 0.028 | 0.021 ± 0.006 | 0.005 ± 0.003 | 0.005 ± 0.005 | 0.006 ± 0.005 | 0.006 ± 0.005 | 0.006 ± 0.005 |

- **Unconditional:** OT is the clear win at low step counts. At 1–3 steps, OT is 0.95 / 0.53 / 0.41 against
  7.02 / 3.84 / 1.28 without OT. Without OT, the 1-step model also collapses onto a few modes (coverage KL 0.539
  vs. 0.009 with OT). By 9 steps the gap is gone, with no-OT slightly ahead (0.31 vs. 0.26). So straighter OT paths
  only pay off when sampling with very few steps.
- **Conditional, no CFG:** the no-OT model is accurate even at 1 step (100%, 0.23). Each condition's goals form a
  tight Gaussian, so independently paired paths are already nearly straight. Its 0.11–0.19 at 2–9 steps is below
  the 0.219 floor, which means its samples are too tightly clustered, not more accurate. The OT model improves from
  1.72 at 1 step but plateaus around 0.85–0.9 from 5 steps on.
- **Conditional, CFG at guidance 3.0:** guidance overshoots badly with few steps (11.5 / 14.9 at 1 step, about 2.2
  at 3 steps) and only recovers from about 9 steps. This sweep holds guidance at 3.0, so CFG's few-step numbers mix
  the effect of guidance with the effect of step count.

### Guidance (CFG) sweep, 100 steps

Figure: [experiments/results/pose/cfg_sweep.png](experiments/results/pose/cfg_sweep.png).

| Variant | Guidance scale | Mode accuracy ↑ | Twist distance (data: 0.219) | Mode coverage KL ↓ |
| --- | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 1 | 0.966 ± 0.026 | 1.279 ± 0.088 | 0.0014 ± 0.0017 |
| cond · OT + CFG | 1.5 | 0.998 ± 0.003 | 0.729 ± 0.078 | 0.0001 ± 0.0001 |
| cond · OT + CFG | 2 | 1.000 ± 0.000 | 0.544 ± 0.060 | 0.0000 ± 0.0000 |
| cond · OT + CFG | 3 | 1.000 ± 0.000 | 0.425 ± 0.039 | 0.0000 ± 0.0000 |
| cond · OT + CFG | 5 | 1.000 ± 0.000 | 0.376 ± 0.027 | 0.0000 ± 0.0000 |
| cond · OT + CFG | 7 | 1.000 ± 0.000 | 0.382 ± 0.023 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 1 | 1.000 ± 0.000 | 0.287 ± 0.017 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 1.5 | 1.000 ± 0.000 | 0.279 ± 0.019 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 2 | 1.000 ± 0.000 | 0.294 ± 0.021 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 3 | 1.000 ± 0.000 | 0.346 ± 0.034 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 5 | 1.000 ± 0.000 | 0.585 ± 0.166 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 7 | 1.000 ± 0.000 | 1.260 ± 0.537 | 0.0000 ± 0.0000 |

- **With OT:** the model needs guidance to compensate for the pairing flaw. Its distance falls from 1.279 at
  guidance 1 to about 0.38 at guidance 5–7.
- **Without OT:** the model is best unguided or lightly guided (0.279 at guidance 1.5). Guidance beyond 2 hurts:
  0.346 at 3, and 1.260 ± 0.537 at 7.
- **Default guidance:** 3.0 sits between the two optima. It handicaps the no-OT CFG model in the main table.

### Training

| Variant | Final-epoch training loss | Minutes per run (6 in parallel) |
| --- | ---: | ---: |
| cond · OT + CFG | 0.1508 ± 0.0013 | 7.4 |
| cond · OT, no CFG | 0.1381 ± 0.0020 | 7.2 |
| cond · no OT + CFG | 0.6497 ± 0.0079 | 6.3 |
| cond · no OT, no CFG | 0.4433 ± 0.0048 | 6.3 |
| uncond · OT | 0.2474 ± 0.0014 | 5.7 |
| uncond · no OT | 2.2846 ± 0.0127 | 4.7 |

Loss isn't comparable across settings. OT pairing changes the regression target, and CFG adds unconditional
samples whose targets vary much more. Within the unconditional pair, OT's ~9× lower loss (0.247 vs. 2.285) is
consistent with it removing crossing paths.

## Known issues and caveats

- **Conditional OT pairs across conditions.** Training samples goals and their action tokens mode by mode, but
  `_run_flow_matching_step` in `utils/train_utils.py` runs minibatch OT over the whole mixed batch. Each condition
  is therefore trained on only the region of noise nearest its mode, while sampling draws from the full Gaussian.
  This fits the evidence:
  - OT helps unconditional models but hurts conditional ones.
  - The conditional OT model needs strong guidance to recover.

  The standard fix is to run OT within each condition. The conditional OT pose results should be rerun after that
  fix; the other four variants are unaffected.
- **Twist distance has a floor.** Values below 0.219 mean the samples are too tightly clustered. A
  distribution-level metric, such as MMD against real goal samples, would penalize both missing the mode and
  collapsing onto it.
- **The steps sweep uses guidance 3.0 for CFG variants.** Their poor few-step results are partly a guidance effect.
- **Pose training only sees 10 fixed time values** (multiples of 1/9). Sampling with 1, 3 or 9 steps queries only
  time values seen in training; other step counts interpolate in time.
- **5 seeds.** The standard deviations are sample standard deviations over 5 runs. Read differences smaller than
  about 2 standard deviations as unresolved.

## MNIST status

Not trained yet. The pipeline is ready (`./mnist_ablations.sh`), and its evaluation stage has been tested on an old
checkpoint. Things to know before running it:

- **No-CFG dropout fixed:** No-CFG MNIST models previously still trained with ~10% unconditional dropout, because
  MNIST has a single conditioning token. With this fix it is 0%.
- **CFG dropout rate:** CFG MNIST models train with ~19% null-token dropout rather than the nominal 10%, because
  per-token and whole-sample dropout stack when there is only one token. This was left unchanged to match the
  earlier tuned model.
- **Same OT pairing flaw:** MNIST conditional OT (flat OT across the batch) pairs across classes too. It should be
  fixed before training.
- **Cost:** one 400-epoch run takes about 26 minutes, so 5 seeds × 6 variants is about 13 hours sequentially.
  Parallel speedup for MNIST hasn't been measured.

## Next steps

1. Make OT pairing per-condition and retrain the 10 conditional OT pose runs (about 15 minutes).
2. Add a distribution-level pose metric that penalizes collapse.
3. Run the MNIST ablations after the OT fix.

## Reproducing

```bash
conda activate FmT
./pose_ablations.sh             # train (resumable) + evaluate + aggregate; ./pose_ablations.sh eval to re-evaluate
./mnist_ablations.sh            # same for MNIST; ./mnist_ablations.sh eval for evaluation only
```

Outputs (gitignored):

- `checkpoints/pose/seed_<N>/`: checkpoints, training configs and logs. The full run log is
  `checkpoints/pose/pose_ablations_console.txt`.
- `experiments/results/pose/`: `metrics_summary.csv`, `steps_sweep_summary.csv`, `cfg_sweep_summary.csv`,
  `steps_sweep.png` and `cfg_sweep.png`.
- `experiments/results/pose/seed_<N>/`: per-seed `metrics.csv`, `steps_sweep.csv`, `cfg_sweep.csv`,
  `pose_grid.png` (sample trajectories) and `cfg_sweep.png`.

## Appendix: per-seed results (100 steps)

**Mode accuracy:**

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cond · OT, no CFG | 1.000 | 1.000 | 0.992 | 0.953 | 1.000 |
| cond · no OT + CFG | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cond · no OT, no CFG | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**Twist distance** (real data: 0.219):

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 0.399 | 0.425 | 0.466 | 0.375 | 0.462 |
| cond · OT, no CFG | 0.714 | 0.806 | 0.820 | 1.145 | 0.750 |
| cond · no OT + CFG | 0.395 | 0.329 | 0.322 | 0.365 | 0.316 |
| cond · no OT, no CFG | 0.289 | 0.288 | 0.305 | 0.282 | 0.277 |
| uncond · OT | 0.295 | 0.284 | 0.288 | 0.292 | 0.281 |
| uncond · no OT | 0.293 | 0.273 | 0.275 | 0.272 | 0.279 |

**Mode coverage KL:**

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| cond · OT, no CFG | 0.0000 | 0.0000 | 0.0002 | 0.0048 | 0.0000 |
| cond · no OT + CFG | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| cond · no OT, no CFG | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| uncond · OT | 0.0043 | 0.0073 | 0.0126 | 0.0101 | 0.0071 |
| uncond · no OT | 0.0005 | 0.0029 | 0.0136 | 0.0089 | 0.0044 |
