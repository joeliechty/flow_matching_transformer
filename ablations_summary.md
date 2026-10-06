# Ablation Study Summary

**Status (2026-10-06):**
- **Pose:** the SE(3) pose ablations are complete (6 variants × 5 seeds, each trained for 100 epochs). The main
  results below use the **epoch-50 checkpoints**, where differences between variants are clearer. Epoch-100 results
  are kept [for reference](#reference-epoch-100-results).
- **MNIST:** set up but not trained yet.

**Code versions:**
- Training and the epoch-100 evaluation ran at commit `cd6d937` on `dev/ablations`.
- The epoch-50 evaluation ran at `1b31280`, which only adds checkpoint-epoch selection.

The study asks what two components add to the flow matching transformer:

- **OT**: minibatch optimal-transport pairing between noise samples and data samples during training.
- **CFG**: classifier-free guidance. Training randomly replaces the conditioning with a learned null token, and
  sampling extrapolates away from the unconditional prediction.

## Key findings (epoch 50)

1. **OT makes few-step sampling work for unconditional models.**
   - At 1 step, the OT model lands 1.06 from the nearest mode versus 6.99 without OT. At 3 steps it's 0.45 vs. 1.45.
   - Both converge by about 9 steps. With many steps, no-OT is slightly ahead (0.33 vs. 0.37).
2. **OT makes conditional models worse, most likely because of how pairing is implemented.** Minibatch OT pairs
   noise and goal samples across the whole batch, ignoring each goal's condition.
   - Without CFG, the OT model lands 0.84 ± 0.11 from its target mode versus 0.31 ± 0.01 without OT. With CFG
     it's 0.55 vs. 0.42.
   - "OT, no CFG" is the only variant below 100% mode accuracy.
   - It is also the only variant that stops improving after epoch 50.
   - The conditional OT numbers reflect this flaw rather than OT itself
     (see [Known issues](#known-issues-and-caveats)).
3. **CFG hurts at the default guidance.** Without OT, training with CFG and sampling at guidance 3.0 costs +0.11
   (0.42 vs. 0.31), twice the gap at epoch 100. At its best guidance (1.5), the CFG model only matches the plain
   model (0.307 vs. 0.310), and at guidance 7 it falls apart (2.52).
4. **Guidance 3.0 breaks few-step sampling.** At 1 step, the CFG variants land 11.5–15.0 from their target. They
   only recover from about 9–20 steps.
5. **Mode accuracy saturates even at epoch 50.** Every conditional variant except "OT, no CFG" scores 100% at 100
   steps, so twist distance carries the comparison.

## Why epoch 50

Checkpoints are saved every 10 epochs, so epoch 25 wasn't available without retraining. The pose trainer uses a
constant learning rate with no schedule, so the epoch-50 checkpoint of a 100-epoch run is the model a 50-epoch
training run would produce. Epochs 20 and 30 were checked with a quick pilot (main metrics at 100 and 3 steps
only). The table shows each ablation effect at each checkpoint:

| Checkpoint | CFG effect, no OT (Δ distance) | OT effect, cond, no CFG (Δ distance) | Uncond, 3 steps: no-OT / OT distance | Best cond distance (data: 0.219) |
| --- | ---: | ---: | ---: | ---: |
| epoch 20 | +0.272 | +0.675 | 2.7× | 0.491 |
| epoch 30 | +0.232 | +0.629 | 3.1× | 0.380 |
| epoch 50 | +0.113 | +0.532 | 3.2× | 0.310 |
| epoch 100 | +0.057 | +0.559 | 3.1× | 0.288 |

Δ distance = twist distance of the variant with the component minus the variant without it, at 100 steps.

- **Earlier checkpoints:** they widen the raw gaps, but every model is still far from the data (best conditional
  distance 0.38–0.49 vs. 0.219). The gaps there mostly measure how fast each variant learns. Measured against the
  seed-to-seed spread, epochs 20–30 separate the variants only 1.0–1.6× more clearly than epoch 50.
- **Epoch 50 vs. 100:** epoch 50 shows the CFG effect twice as clearly (4.6 vs. 2.3 seed standard deviations) and
  the conditional OT effect 1.5× as clearly (6.9 vs. 4.6). Its models are close to converged (0.310).
- **Unconditional few-step OT advantage:** this one is about equally clear at every checkpoint (2.7–3.2×).

**Twist distance at 100 steps, by training epoch:**

| Variant | epoch 20 | epoch 30 | epoch 50 | epoch 100 |
| --- | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 0.867 ± 0.106 | 0.689 ± 0.043 | 0.550 ± 0.069 | 0.425 ± 0.039 |
| cond · OT, no CFG | 1.166 ± 0.089 | 1.009 ± 0.127 | 0.842 ± 0.109 | 0.847 ± 0.172 |
| cond · no OT + CFG | 0.763 ± 0.065 | 0.612 ± 0.051 | 0.423 ± 0.033 | 0.346 ± 0.034 |
| cond · no OT, no CFG | 0.491 ± 0.038 | 0.380 ± 0.021 | 0.310 ± 0.010 | 0.288 ± 0.011 |
| uncond · OT | 0.571 ± 0.018 | 0.452 ± 0.027 | 0.371 ± 0.026 | 0.288 ± 0.006 |
| uncond · no OT | 0.509 ± 0.027 | 0.394 ± 0.011 | 0.326 ± 0.022 | 0.278 ± 0.009 |

**Unconditional twist distance at 3 steps, by training epoch:**

| Variant (3 steps) | epoch 20 | epoch 30 | epoch 50 | epoch 100 |
| --- | ---: | ---: | ---: | ---: |
| uncond · OT | 0.716 ± 0.042 | 0.564 ± 0.035 | 0.448 ± 0.051 | 0.409 ± 0.002 |
| uncond · no OT | 1.926 ± 0.106 | 1.761 ± 0.072 | 1.447 ± 0.098 | 1.277 ± 0.063 |

Every variant keeps improving from epoch 50 to 100 except "OT, no CFG", which has plateaued (0.842 → 0.847).
That points to a structural problem rather than slow training.

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
- Schedule: 100 epochs × 100 batches × 128 samples, AdamW (constant learning rate 1e-4, weight decay 1e-5).
  Checkpoints are saved every 10 epochs.
- Model: transformer with hidden size 128, 4 layers and 4 heads.
- Interpolation: 10 points per path.
- CFG: drops the whole conditioning with p = 0.1. Every conditional model also drops individual tokens with
  p = 0.1. Without CFG, that per-token dropout never blanks both tokens at once.
- Seeds: 1–5 for every variant, with all other hyperparameters identical.
- Hardware: one RTX 4090, 6 runs in parallel. Training ran from 2026-10-06 13:59 to 2026-10-06 14:31 (32 min for 30 runs).

**Evaluation.**
- Samples: 256 per model. For conditional models that's 64 per mode, conditioned on that mode's action tokens.
- Sampling: Euler integration with 100 steps unless the steps are being swept.
- Noise: the sampling noise is seeded identically for every variant, checkpoint and sweep point. Spread across
  seeds therefore reflects training seeds only.
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

## Results (epoch-50 checkpoints)

### Main results (100 steps)

| Variant | Guidance scale | Mode accuracy ↑ | Twist distance (data: 0.219) | Mode coverage KL ↓ |
| --- | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 3.0 | 1.000 ± 0.000 | 0.550 ± 0.069 | 0.0000 ± 0.0000 |
| cond · OT, no CFG | 1.0 (unguided) | 0.995 ± 0.010 | 0.842 ± 0.109 | 0.0004 ± 0.0008 |
| cond · no OT + CFG | 3.0 | 1.000 ± 0.000 | 0.423 ± 0.033 | 0.0000 ± 0.0000 |
| cond · no OT, no CFG | 1.0 (unguided) | 1.000 ± 0.000 | 0.310 ± 0.010 | 0.0000 ± 0.0000 |
| uncond · OT | n/a | — | 0.371 ± 0.026 | 0.0096 ± 0.0026 |
| uncond · no OT | n/a | — | 0.326 ± 0.022 | 0.0148 ± 0.0050 |

- Without OT, both conditional variants hit 100% accuracy. The unguided model is the closest to real data at
  0.310, against a floor of 0.219.
- Conditional OT variants are worse on twist distance. "OT, no CFG" is also the least stable across seeds: seed 4
  scores 97.7% accuracy and a distance of 1.020 (see [Appendix](#appendix-per-seed-results-epoch-50-100-steps)).
- Both unconditional models cover all four modes about evenly (KL ≤ 0.015). At 100 steps no-OT is slightly
  closer than OT (0.326 vs. 0.371, about 2 seed standard deviations apart).

### Sampling-steps sweep

Figure: [experiments/results/pose/epoch_50/steps_sweep.png](experiments/results/pose/epoch_50/steps_sweep.png).
It is generated locally and gitignored, like all files under `experiments/results/`.

**Twist distance vs. integration steps** (real data: 0.219):

| Variant | 1 step | 2 steps | 3 steps | 5 steps | 9 steps | 20 steps | 50 steps | 100 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 11.51 ± 0.40 | 4.75 ± 0.04 | 2.43 ± 0.04 | 0.95 ± 0.05 | 0.45 ± 0.06 | 0.46 ± 0.07 | 0.53 ± 0.07 | 0.55 ± 0.07 |
| cond · OT, no CFG | 1.65 ± 0.21 | 1.05 ± 0.14 | 0.94 ± 0.13 | 0.89 ± 0.12 | 0.87 ± 0.12 | 0.85 ± 0.11 | 0.84 ± 0.11 | 0.84 ± 0.11 |
| cond · no OT + CFG | 15.03 ± 0.12 | 5.86 ± 0.13 | 2.95 ± 0.10 | 1.10 ± 0.07 | 0.57 ± 0.05 | 0.46 ± 0.04 | 0.43 ± 0.03 | 0.42 ± 0.03 |
| cond · no OT, no CFG | 0.33 ± 0.02 | 0.16 ± 0.02 | 0.16 ± 0.01 | 0.19 ± 0.01 | 0.24 ± 0.01 | 0.28 ± 0.01 | 0.30 ± 0.01 | 0.31 ± 0.01 |
| uncond · OT | 1.06 ± 0.05 | 0.55 ± 0.05 | 0.45 ± 0.05 | 0.40 ± 0.04 | 0.38 ± 0.03 | 0.37 ± 0.03 | 0.37 ± 0.03 | 0.37 ± 0.03 |
| uncond · no OT | 6.99 ± 0.02 | 3.96 ± 0.25 | 1.45 ± 0.10 | 0.58 ± 0.04 | 0.37 ± 0.02 | 0.33 ± 0.02 | 0.33 ± 0.02 | 0.33 ± 0.02 |

**Mode accuracy vs. integration steps** (conditional models):

| Variant | 1 step | 2 steps | 3 steps | 5 steps | 9 steps | 20 steps | 50 steps | 100 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 0.907 ± 0.024 | 0.992 ± 0.009 | 0.998 ± 0.005 | 0.999 ± 0.002 | 1.000 ± 0.000 | 0.999 ± 0.002 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| cond · OT, no CFG | 0.923 ± 0.021 | 0.985 ± 0.021 | 0.988 ± 0.018 | 0.992 ± 0.015 | 0.994 ± 0.014 | 0.994 ± 0.014 | 0.995 ± 0.010 | 0.995 ± 0.010 |
| cond · no OT + CFG | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| cond · no OT, no CFG | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

**Mode coverage KL vs. integration steps**:

| Variant | 1 step | 2 steps | 3 steps | 5 steps | 9 steps | 20 steps | 50 steps | 100 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 0.004 ± 0.002 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| cond · OT, no CFG | 0.004 ± 0.003 | 0.001 ± 0.002 | 0.001 ± 0.002 | 0.001 ± 0.002 | 0.001 ± 0.001 | 0.001 ± 0.001 | 0.000 ± 0.001 | 0.000 ± 0.001 |
| cond · no OT + CFG | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| cond · no OT, no CFG | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| uncond · OT | 0.010 ± 0.002 | 0.010 ± 0.003 | 0.010 ± 0.003 | 0.010 ± 0.002 | 0.010 ± 0.002 | 0.010 ± 0.003 | 0.010 ± 0.003 | 0.010 ± 0.003 |
| uncond · no OT | 0.534 ± 0.300 | 0.079 ± 0.036 | 0.024 ± 0.006 | 0.013 ± 0.005 | 0.013 ± 0.006 | 0.014 ± 0.005 | 0.014 ± 0.005 | 0.015 ± 0.005 |

- **Unconditional:** OT is the clear win at low step counts. At 1–3 steps, OT is 1.06 / 0.55 / 0.45 against
  6.99 / 3.96 / 1.45 without OT.
  - Without OT, the 1-step model also collapses onto a few modes (coverage KL 0.534 vs. 0.010 with OT).
  - By 9 steps the two are level (0.38 vs. 0.37), and beyond that no-OT is slightly ahead (0.33 vs. 0.37).
  - So straighter OT paths only pay off when sampling with very few steps.
- **Conditional, no CFG:**
  - The no-OT model is accurate even at 1 step (100%, 0.33). Each condition's goals form a tight Gaussian, so
    independently paired paths are already nearly straight.
  - Its 0.16–0.19 at 2–5 steps is below the 0.219 floor, which means its samples are too tightly clustered, not
    more accurate.
  - The OT model improves from 1.65 at 1 step but plateaus around 0.84–0.89 from 5 steps on.
- **Conditional, CFG at guidance 3.0:** guidance overshoots badly with few steps (11.5 / 15.0 at 1 step, 2.4 /
  3.0 at 3 steps). It recovers by 9–20 steps (0.45 / 0.57 at 9, 0.46 for both at 20). This sweep holds guidance
  at 3.0, so CFG's few-step numbers mix the effect of guidance with the effect of step count.

### Guidance (CFG) sweep, 100 steps

Figure: [experiments/results/pose/epoch_50/cfg_sweep.png](experiments/results/pose/epoch_50/cfg_sweep.png).

| Variant | Guidance scale | Mode accuracy ↑ | Twist distance (data: 0.219) | Mode coverage KL ↓ |
| --- | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 1 | 0.961 ± 0.020 | 1.341 ± 0.171 | 0.0015 ± 0.0013 |
| cond · OT + CFG | 1.5 | 0.994 ± 0.009 | 0.844 ± 0.126 | 0.0004 ± 0.0006 |
| cond · OT + CFG | 2 | 0.998 ± 0.003 | 0.671 ± 0.093 | 0.0000 ± 0.0001 |
| cond · OT + CFG | 3 | 1.000 ± 0.000 | 0.550 ± 0.069 | 0.0000 ± 0.0000 |
| cond · OT + CFG | 5 | 1.000 ± 0.000 | 0.504 ± 0.054 | 0.0000 ± 0.0000 |
| cond · OT + CFG | 7 | 1.000 ± 0.000 | 0.537 ± 0.056 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 1 | 1.000 ± 0.000 | 0.318 ± 0.014 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 1.5 | 1.000 ± 0.000 | 0.307 ± 0.016 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 2 | 1.000 ± 0.000 | 0.328 ± 0.020 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 3 | 1.000 ± 0.000 | 0.423 ± 0.033 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 5 | 1.000 ± 0.000 | 0.979 ± 0.106 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 7 | 1.000 ± 0.000 | 2.519 ± 0.315 | 0.0000 ± 0.0000 |

- **With OT:** the model needs guidance to compensate for the pairing flaw. Its distance falls from 1.341 at
  guidance 1 to a best of 0.504 at guidance 5.
- **Without OT:** the model is best at guidance 1.5 (0.307), which only matches the no-CFG model (0.310).
  Guidance beyond 2 hurts: 0.423 at 3, 0.979 at 5 and 2.519 ± 0.315 at 7. That's about twice as sensitive as at
  epoch 100 (1.260 at guidance 7).
- **Default guidance:** 3.0 sits between the two optima. It handicaps the no-OT CFG model in the main table.

### Training

| Variant | Training loss, epoch 50 | Training loss, epoch 100 | Minutes per 100-epoch run (6 in parallel) |
| --- | ---: | ---: | ---: |
| cond · OT + CFG | 0.1769 ± 0.0022 | 0.1508 ± 0.0013 | 7.4 |
| cond · OT, no CFG | 0.1625 ± 0.0021 | 0.1381 ± 0.0020 | 7.2 |
| cond · no OT + CFG | 0.7067 ± 0.0056 | 0.6497 ± 0.0079 | 6.3 |
| cond · no OT, no CFG | 0.4947 ± 0.0051 | 0.4433 ± 0.0048 | 6.3 |
| uncond · OT | 0.2844 ± 0.0046 | 0.2474 ± 0.0014 | 5.7 |
| uncond · no OT | 2.3307 ± 0.0165 | 2.2846 ± 0.0127 | 4.7 |

Loss isn't comparable across settings. OT pairing changes the regression target, and CFG adds unconditional
samples whose targets vary much more. Within the unconditional pair, OT's ~8–9× lower loss is consistent with it
removing crossing paths.

## Reference: epoch-100 results

Main results for the final checkpoints, evaluated the same way:

| Variant | Guidance scale | Mode accuracy ↑ | Twist distance (data: 0.219) | Mode coverage KL ↓ |
| --- | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 3.0 | 1.000 ± 0.000 | 0.425 ± 0.039 | 0.0000 ± 0.0000 |
| cond · OT, no CFG | 1.0 (unguided) | 0.989 ± 0.020 | 0.847 ± 0.172 | 0.0010 ± 0.0021 |
| cond · no OT + CFG | 3.0 | 1.000 ± 0.000 | 0.346 ± 0.034 | 0.0000 ± 0.0000 |
| cond · no OT, no CFG | 1.0 (unguided) | 1.000 ± 0.000 | 0.288 ± 0.011 | 0.0000 ± 0.0000 |
| uncond · OT | n/a | — | 0.288 ± 0.006 | 0.0083 ± 0.0031 |
| uncond · no OT | n/a | — | 0.278 ± 0.009 | 0.0061 ± 0.0052 |

The full epoch-100 sweeps are in `experiments/results/pose/epoch_100/`.

## Known issues and caveats

- **Conditional OT pairs across conditions.** Training samples goals and their action tokens mode by mode, but
  `_run_flow_matching_step` in `utils/train_utils.py` runs minibatch OT over the whole mixed batch. Each condition
  is therefore trained on only the region of noise nearest its mode, while sampling draws from the full Gaussian.
  This fits the evidence:
  - OT helps unconditional models but hurts conditional ones.
  - The conditional OT model needs strong guidance to recover.
  - "OT, no CFG" plateaus by epoch 50.

  The standard fix is to run OT within each condition. The conditional OT pose results should be rerun after that
  fix; the other four variants are unaffected.
- **Twist distance has a floor.** Values below 0.219 mean the samples are too tightly clustered. A
  distribution-level metric, such as MMD against real goal samples, would penalize both missing the mode and
  collapsing onto it.
- **The steps sweep uses guidance 3.0 for CFG variants.** Their poor few-step results are partly a guidance effect.
- **Pose training only sees 10 fixed time values** (multiples of 1/9). Sampling with 1, 3 or 9 steps queries only
  time values seen in training; other step counts interpolate in time.
- **Epochs 20 and 30 come from a pilot** with main metrics only, at 100 and 3 steps.
  `EVAL_EPOCH=20 ./pose_ablations.sh eval` produces the full set.
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
- **Learning rate:** the MNIST trainer uses a warmup + cosine schedule. Unlike pose, an intermediate MNIST
  checkpoint is *not* equivalent to a shorter training run.
- **Cost:** one 400-epoch run takes about 26 minutes, so 5 seeds × 6 variants is about 13 hours sequentially.
  Parallel speedup for MNIST hasn't been measured.

## Next steps

1. Make OT pairing per-condition and retrain the 10 conditional OT pose runs (about 15 minutes).
2. Add a distribution-level pose metric that penalizes collapse.
3. Run the MNIST ablations after the OT fix.

## Reproducing

```bash
conda activate FmT
./pose_ablations.sh                     # train 5 seeds × 6 variants (resumable), evaluate the epoch-100 checkpoints
EVAL_EPOCH=50 ./pose_ablations.sh eval  # evaluate the epoch-50 checkpoints (this report's main results)
./mnist_ablations.sh                    # same for MNIST; ./mnist_ablations.sh eval for evaluation only
```

Outputs (gitignored):

- `checkpoints/pose/seed_<N>/`: checkpoints, training configs and logs. Run logs are
  `checkpoints/pose/pose_ablations_console.txt` (training + epoch-100 evaluation) and
  `checkpoints/pose/pose_eval_epoch50_console.txt`.
- `experiments/results/pose/epoch_<E>/`: `metrics_summary.csv`, `steps_sweep_summary.csv`, `cfg_sweep_summary.csv`,
  `steps_sweep.png` and `cfg_sweep.png`.
- `experiments/results/pose/epoch_<E>/seed_<N>/`: per-seed `metrics.csv`, `steps_sweep.csv`, `cfg_sweep.csv`,
  `pose_grid.png` (sample trajectories) and `cfg_sweep.png`.

## Appendix: per-seed results (epoch 50, 100 steps)

**Mode accuracy:**

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cond · OT, no CFG | 1.000 | 1.000 | 1.000 | 0.977 | 1.000 |
| cond · no OT + CFG | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cond · no OT, no CFG | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**Twist distance** (real data: 0.219):

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 0.526 | 0.509 | 0.551 | 0.497 | 0.669 |
| cond · OT, no CFG | 0.740 | 0.824 | 0.774 | 1.020 | 0.853 |
| cond · no OT + CFG | 0.396 | 0.448 | 0.407 | 0.468 | 0.395 |
| cond · no OT, no CFG | 0.303 | 0.307 | 0.327 | 0.307 | 0.305 |
| uncond · OT | 0.399 | 0.371 | 0.392 | 0.351 | 0.339 |
| uncond · no OT | 0.361 | 0.313 | 0.330 | 0.324 | 0.302 |

**Mode coverage KL:**

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| cond · OT, no CFG | 0.0000 | 0.0000 | 0.0000 | 0.0018 | 0.0000 |
| cond · no OT + CFG | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| cond · no OT, no CFG | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| uncond · OT | 0.0075 | 0.0137 | 0.0105 | 0.0080 | 0.0084 |
| uncond · no OT | 0.0097 | 0.0137 | 0.0219 | 0.0179 | 0.0110 |
