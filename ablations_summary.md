# Ablation Study Summary

**Status (2026-10-06):**
- **Pose:** the SE(3) pose ablations are complete (6 variants × 5 seeds, each trained for 100 epochs). A pairing bug
  in conditional OT, found in the first round, has been [fixed](#conditional-ot-pairing-fix), and the 10 affected
  runs retrained. All results below are after the fix.
  - Main results use the **epoch-50 checkpoints**, where differences between variants are clearer. Epoch-100
    results are kept [for reference](#reference-epoch-100-results).
- **MNIST:** deferred, not trained.

**Code versions (branch `dev/ablations`):**
- The 20 runs the fix doesn't affect were trained at `cd6d937`.
- The 10 conditional-OT runs were retrained at `f80364c` (the fix).
- Every evaluation reported here ran at `f80364c`.

The study asks what two components add to the flow matching transformer:

- **OT**: minibatch optimal-transport pairing between noise samples and data samples during training.
- **CFG**: classifier-free guidance. Training randomly replaces the conditioning with a learned null token, and
  sampling extrapolates away from the unconditional prediction.

## Key findings (epoch 50)

1. **OT makes few-step sampling work for unconditional models.**
   - At 1 step, the OT model lands 1.06 from the nearest mode versus 6.99 without OT. At 3 steps it's 0.45 vs. 1.45.
   - Both converge by about 9 steps. With many steps, no-OT is slightly ahead (0.33 vs. 0.37).
2. **With correct pairing, OT neither helps nor hurts conditional models.**
   - Without CFG it lands 0.319 ± 0.025 from the target mode versus 0.310 ± 0.010 without OT. With CFG it's
     0.461 vs. 0.423. Both gaps are under 1 seed standard deviation.
   - The large OT penalty in the first round was a pairing bug, now fixed.
3. **CFG hurts at the default guidance.**
   - Without OT, training with CFG and sampling at guidance 3.0 costs +0.11 (0.423 vs. 0.310), twice the gap at
     epoch 100.
   - At their best guidance (1.5), both CFG variants only match the unguided models, and guidance 7 falls apart
     (2.5–2.7).
4. **Guidance 3.0 breaks few-step sampling.** At 1 step, both CFG variants land 15.0 from their target. That is
   close to what a perfect model would do on this task (see [below](#why-the-task-is-too-easy)).
5. **The conditional task is too easy to show what OT or CFG add.** Every conditional variant scores 100% mode
   accuracy, and one-step results match hand calculations for a perfect model. The next phase adds better metrics and
   a harder pose task ([Next steps](#next-steps)).

## Conditional OT pairing fix

**The bug.** Training samples goals and their action tokens mode by mode. But `_run_flow_matching_step` in
`utils/train_utils.py` ran minibatch OT over the whole mixed batch, so each condition was paired with the noise
nearest its own mode. A conditional model therefore never saw the rest of the noise distribution for that condition,
while sampling draws from all of it.

Measured over 100 training batches of 128, the starting noise each condition trained on sat **+1.085 ± 0.005 noise
standard deviations toward its own goal** (mean ± standard error). With per-condition pairing it is −0.017 ± 0.009.

**The fix (`f80364c`).** `_run_flow_matching_step` takes a condition index per sample: the mode index for pose, the
class label for MNIST. OT then pairs within each condition only. Unconditional models keep global OT through an
unchanged code path, so only the 10 conditional-OT runs needed retraining.

Unit tests checked three things: paired noise never crosses conditions, the result equals running OT on each
condition separately, and a single condition reduces to the old global pairing.

**Success criteria** were set before the retrained results came in. Within one condition all goals sit in a tight
Gaussian, so the expectation was *parity* with no-OT, not improvement. "SD" is the gap divided by the pooled seed
standard deviation.

| Criterion | Before fix | After fix | Pass |
| --- | ---: | ---: | ---: |
| Mode accuracy 100% (worst OT variant, epochs 50/100) | 0.989 | 1.000 | ✓ |
| Distance within ~2 SD of no-OT: cond · OT + CFG, epoch 50 | +2.3 SD | +0.8 SD | ✓ |
| Distance within ~2 SD of no-OT: cond · OT, no CFG, epoch 50 | +6.9 SD | +0.5 SD | ✓ |
| Distance within ~2 SD of no-OT: cond · OT + CFG, epoch 100 | +2.2 SD | -0.3 SD | ✓ |
| Distance within ~2 SD of no-OT: cond · OT, no CFG, epoch 100 | +4.6 SD | +0.0 SD | ✓ |
| Keeps improving epoch 50 → 100 (OT, no CFG) | 0.842 → 0.847 | 0.319 → 0.289 | ✓ |
| Best guidance for OT + CFG near no-OT's (1.5), epoch 50 | 5 | 1.5 | ✓ |

**Before and after, at 100 steps:**

| Variant | Epoch | Accuracy before | Accuracy after | Distance before | Distance after | Matching no-OT distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 50 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.550 ± 0.069 | 0.461 ± 0.062 | 0.423 ± 0.033 (cond · no OT + CFG) |
| cond · OT, no CFG | 50 | 0.995 ± 0.010 | 1.000 ± 0.000 | 0.842 ± 0.109 | 0.319 ± 0.025 | 0.310 ± 0.010 (cond · no OT, no CFG) |
| cond · OT + CFG | 100 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.425 ± 0.039 | 0.330 ± 0.054 | 0.346 ± 0.034 (cond · no OT + CFG) |
| cond · OT, no CFG | 100 | 0.989 ± 0.020 | 1.000 ± 0.000 | 0.847 ± 0.172 | 0.289 ± 0.015 | 0.288 ± 0.011 (cond · no OT, no CFG) |

**Guidance sweep before and after** (twist distance, epoch 50). The buggy model needed strong guidance to compensate;
the fixed one behaves like no-OT:

| Guidance scale | OT + CFG before | OT + CFG after | no OT + CFG |
| --- | ---: | ---: | ---: |
| 1 | 1.341 ± 0.171 | 0.327 ± 0.005 | 0.318 ± 0.014 |
| 1.5 | 0.844 ± 0.126 | 0.320 ± 0.014 | 0.307 ± 0.016 |
| 2 | 0.671 ± 0.093 | 0.347 ± 0.027 | 0.328 ± 0.020 |
| 3 | 0.550 ± 0.069 | 0.461 ± 0.062 | 0.423 ± 0.033 |
| 5 | 0.504 ± 0.054 | 1.096 ± 0.227 | 0.979 ± 0.106 |
| 7 | 0.537 ± 0.056 | 2.741 ± 0.489 | 2.519 ± 0.315 |

The pre-fix checkpoints and results are archived in `checkpoints/pose_before_ot_fix/` and
`experiments/results/pose_before_ot_fix/`.

## Why the task is too easy

- **The modes are far apart.** Neighboring modes are 10 units apart in position, with σ = 0.1 per mode (about
  100σ). A sample has to be roughly 7 units off before mode accuracy counts it as wrong.
- **The conditioning gives the answer away.** It is deterministic and one-to-one, so each condition's target is a
  single tight Gaussian. The velocity field is then close to constant, and the model (about 1.5M parameters) has
  far more capacity than it needs. Differences between variants come from optimization speed, not capability.
- **The few-step numbers are what a perfect model gives on paper:**
  - Without OT, the best one-step velocity sends every unconditional sample to the average of the 4 modes. That
    point is about √50 ≈ 7.07 from each mode, and the measured value is 6.99.
  - With guidance w = 3, one step lands at `mean + 3·(mode − mean)`, about 2 × 7.07 ≈ 14.1 past the target. The
    measured value is 15.0.
- **What that means for the components:**
  - Guidance has nothing to sharpen, so it can only overshoot.
  - Within one condition there's nothing for OT to straighten, which the fix's parity confirms.
  - The "no benefit" results for CFG and conditional OT probably won't carry over to harder tasks.
- **The metrics have blind spots of their own:**
  - Mode accuracy only catches a model that ignores its condition.
  - Coverage KL for conditional models only echoes accuracy, because the evaluation batch is balanced.
  - Twist distance rewards collapse (floor 0.219) and adds translation and rotation in one norm.
  - Nothing measures path straightness directly.

## Why epoch 50

Checkpoints are saved every 10 epochs, so epoch 25 wasn't available without retraining. The pose trainer uses a
constant learning rate with no schedule, so the epoch-50 checkpoint of a 100-epoch run is the model a 50-epoch
training run would produce.

Epochs 20 and 30 were checked with a quick pilot (main metrics at 100 and 3 steps only, all fixed models). Each
effect below is the twist distance with the component minus without it, at 100 steps, with its size in pooled seed
standard deviations:

| Checkpoint | CFG effect, no OT | OT effect, cond, no CFG | Uncond, 3 steps: no-OT / OT distance | Best cond distance (data: 0.219) |
| --- | ---: | ---: | ---: | ---: |
| epoch 20 | +0.272 (+5.1 SD) | +0.042 (+1.1 SD) | 2.7× | 0.491 |
| epoch 30 | +0.232 (+5.9 SD) | +0.014 (+0.8 SD) | 3.1× | 0.380 |
| epoch 50 | +0.113 (+4.6 SD) | +0.009 (+0.5 SD) | 3.2× | 0.310 |
| epoch 100 | +0.057 (+2.3 SD) | +0.000 (+0.0 SD) | 3.1× | 0.288 |

- **Earlier checkpoints:** they widen the CFG gap, but every model is still far from the data (best conditional
  distance 0.38–0.49 vs. 0.219). The gaps there mostly measure how fast each variant learns. Measured against the
  seed spread, epochs 20–30 separate the variants only 1.1–1.6× more clearly than epoch 50.
- **Epoch 50 vs. 100:** epoch 50 shows the CFG effect twice as clearly (4.6 vs. 2.3 SD), and its models are close
  to converged (0.310).
- **Unchanged effects:** the unconditional few-step OT advantage is about equally clear at every checkpoint
  (2.7–3.2×). After the fix, the conditional OT effect is about zero everywhere (at most 1.1 SD, at epoch 20).

**Twist distance at 100 steps, by training epoch:**

| Variant | epoch 20 | epoch 30 | epoch 50 | epoch 100 |
| --- | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 0.808 ± 0.065 | 0.583 ± 0.018 | 0.461 ± 0.062 | 0.330 ± 0.054 |
| cond · OT, no CFG | 0.533 ± 0.040 | 0.393 ± 0.010 | 0.319 ± 0.025 | 0.289 ± 0.015 |
| cond · no OT + CFG | 0.763 ± 0.065 | 0.612 ± 0.051 | 0.423 ± 0.033 | 0.346 ± 0.034 |
| cond · no OT, no CFG | 0.491 ± 0.038 | 0.380 ± 0.021 | 0.310 ± 0.010 | 0.288 ± 0.011 |
| uncond · OT | 0.571 ± 0.018 | 0.452 ± 0.027 | 0.371 ± 0.026 | 0.288 ± 0.006 |
| uncond · no OT | 0.509 ± 0.027 | 0.394 ± 0.011 | 0.326 ± 0.022 | 0.278 ± 0.009 |

**Unconditional twist distance at 3 steps, by training epoch:**

| Variant (3 steps) | epoch 20 | epoch 30 | epoch 50 | epoch 100 |
| --- | ---: | ---: | ---: | ---: |
| uncond · OT | 0.716 ± 0.042 | 0.564 ± 0.035 | 0.448 ± 0.051 | 0.409 ± 0.002 |
| uncond · no OT | 1.926 ± 0.106 | 1.761 ± 0.072 | 1.447 ± 0.098 | 1.277 ± 0.063 |

All variants improve steadily from epoch 20 to 100.

## Setup

**Task.** Generate SE(3) poses. Start poses are drawn from a unit Gaussian in twist space. Goal poses come from 4
tight modes (σ = 0.1 per twist dimension) at (5, ±5, ±5) with different rotations. Conditional models receive two
action tokens (up/down and left/right) that identify the target mode.

**Variants.**

| Variant | Conditional | OT pairing | CFG training |
| --- | :---: | :---: | :---: |
| cond · OT + CFG | ✓ | ✓ (within condition) | ✓ |
| cond · OT, no CFG | ✓ | ✓ (within condition) | |
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
- Hardware: one RTX 4090, 6 runs in parallel. The original round (30 runs) took 32 min; the 10 retrained conditional-OT runs ran 2026-10-06 16:47–17:04 (17 min).

**Evaluation.**
- Samples: 256 per model. For conditional models that's 64 per mode, conditioned on that mode's action tokens.
- Sampling: Euler integration with 100 steps unless the steps are being swept.
- Noise: the sampling noise is seeded identically for every variant, checkpoint and sweep point. Spread across
  seeds therefore reflects training seeds only, and re-evaluating an unchanged model reproduces its numbers exactly.
- Guidance: CFG-trained conditional variants use guidance 3.0, and no-CFG variants are sampled unguided (1.0).

**Metrics.**
- **Mode accuracy**: the fraction of conditional samples whose nearest mode (by twist distance) is the target mode.
- **Twist distance**: the mean twist-space distance from each sample to its target mode center, or to its nearest
  mode for unconditional models. Real goal samples score **0.219** (measured on 80,000 draws from the goal
  distribution). That is the floor: values below it mean samples cluster more tightly around the mode center than
  real data, which is collapse, not extra accuracy.
- **Mode coverage KL**: the KL divergence of the histogram of assigned modes from uniform. It is informative for
  unconditional models only (see [above](#why-the-task-is-too-easy)).

All tables report mean ± standard deviation over the 5 seeds.

## Results (epoch-50 checkpoints)

### Main results (100 steps)

| Variant | Guidance scale | Mode accuracy ↑ | Twist distance (data: 0.219) | Mode coverage KL ↓ |
| --- | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 3.0 | 1.000 ± 0.000 | 0.461 ± 0.062 | 0.0000 ± 0.0000 |
| cond · OT, no CFG | 1.0 (unguided) | 1.000 ± 0.000 | 0.319 ± 0.025 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 3.0 | 1.000 ± 0.000 | 0.423 ± 0.033 | 0.0000 ± 0.0000 |
| cond · no OT, no CFG | 1.0 (unguided) | 1.000 ± 0.000 | 0.310 ± 0.010 | 0.0000 ± 0.0000 |
| uncond · OT | n/a | — | 0.371 ± 0.026 | 0.0096 ± 0.0026 |
| uncond · no OT | n/a | — | 0.326 ± 0.022 | 0.0148 ± 0.0050 |

- Every conditional variant hits 100% accuracy. The unguided ones are closest to real data: 0.310 without OT and
  0.319 with OT, against a floor of 0.219. The guided ones (guidance 3.0) are further off, at 0.423 and 0.461.
- Both unconditional models cover all four modes about evenly (KL ≤ 0.015). At 100 steps no-OT is slightly
  closer than OT (0.326 vs. 0.371, about 2 seed standard deviations apart).

### Sampling-steps sweep

Figure: [experiments/results/pose/epoch_50/steps_sweep.png](experiments/results/pose/epoch_50/steps_sweep.png).
It is generated locally and gitignored, like all files under `experiments/results/`.

**Twist distance vs. integration steps** (real data: 0.219):

| Variant | 1 step | 2 steps | 3 steps | 5 steps | 9 steps | 20 steps | 50 steps | 100 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 15.00 ± 0.24 | 5.86 ± 0.16 | 3.03 ± 0.15 | 1.21 ± 0.12 | 0.65 ± 0.09 | 0.52 ± 0.07 | 0.47 ± 0.06 | 0.46 ± 0.06 |
| cond · OT, no CFG | 0.33 ± 0.02 | 0.20 ± 0.03 | 0.22 ± 0.03 | 0.25 ± 0.03 | 0.28 ± 0.03 | 0.30 ± 0.03 | 0.32 ± 0.03 | 0.32 ± 0.03 |
| cond · no OT + CFG | 15.03 ± 0.12 | 5.86 ± 0.13 | 2.95 ± 0.10 | 1.10 ± 0.07 | 0.57 ± 0.05 | 0.46 ± 0.04 | 0.43 ± 0.03 | 0.42 ± 0.03 |
| cond · no OT, no CFG | 0.33 ± 0.02 | 0.16 ± 0.02 | 0.16 ± 0.01 | 0.19 ± 0.01 | 0.24 ± 0.01 | 0.28 ± 0.01 | 0.30 ± 0.01 | 0.31 ± 0.01 |
| uncond · OT | 1.06 ± 0.05 | 0.55 ± 0.05 | 0.45 ± 0.05 | 0.40 ± 0.04 | 0.38 ± 0.03 | 0.37 ± 0.03 | 0.37 ± 0.03 | 0.37 ± 0.03 |
| uncond · no OT | 6.99 ± 0.02 | 3.96 ± 0.25 | 1.45 ± 0.10 | 0.58 ± 0.04 | 0.37 ± 0.02 | 0.33 ± 0.02 | 0.33 ± 0.02 | 0.33 ± 0.02 |

**Mode accuracy vs. integration steps** (conditional models):

| Variant | 1 step | 2 steps | 3 steps | 5 steps | 9 steps | 20 steps | 50 steps | 100 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| cond · OT, no CFG | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| cond · no OT + CFG | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| cond · no OT, no CFG | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

**Mode coverage KL vs. integration steps**:

| Variant | 1 step | 2 steps | 3 steps | 5 steps | 9 steps | 20 steps | 50 steps | 100 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| cond · OT, no CFG | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| cond · no OT + CFG | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| cond · no OT, no CFG | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| uncond · OT | 0.010 ± 0.002 | 0.010 ± 0.003 | 0.010 ± 0.003 | 0.010 ± 0.002 | 0.010 ± 0.002 | 0.010 ± 0.003 | 0.010 ± 0.003 | 0.010 ± 0.003 |
| uncond · no OT | 0.534 ± 0.300 | 0.079 ± 0.036 | 0.024 ± 0.006 | 0.013 ± 0.005 | 0.013 ± 0.006 | 0.014 ± 0.005 | 0.014 ± 0.005 | 0.015 ± 0.005 |

- **Unconditional:** OT is the clear win at low step counts. At 1–3 steps, OT is 1.06 / 0.55 / 0.45 against
  6.99 / 3.96 / 1.45 without OT.
  - Without OT, the 1-step model also collapses onto a few modes (coverage KL 0.534 vs. 0.010 with OT).
  - By 9 steps the two are level, and beyond that no-OT is slightly ahead (0.33 vs. 0.37).
  - So straighter OT paths only pay off when sampling with very few steps.
- **Conditional, no CFG:**
  - Both models are accurate even at 1 step (100%, 0.33 each).
  - At 2–9 steps, no-OT reads 0.16–0.24 and OT 0.20–0.28. Both dip below the 0.219 floor, so this metric
    can't say which is better. The planned bias/spread split will.
- **Conditional, CFG at guidance 3.0:** both models overshoot badly with few steps (15.0 at 1 step, about 3.0 at
  3 steps) and recover by about 20 steps. This sweep holds guidance at 3.0, so CFG's few-step numbers mix the effect
  of guidance with the effect of step count.

### Guidance (CFG) sweep, 100 steps

Figure: [experiments/results/pose/epoch_50/cfg_sweep.png](experiments/results/pose/epoch_50/cfg_sweep.png).

| Variant | Guidance scale | Mode accuracy ↑ | Twist distance (data: 0.219) | Mode coverage KL ↓ |
| --- | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 1 | 1.000 ± 0.000 | 0.327 ± 0.005 | 0.0000 ± 0.0000 |
| cond · OT + CFG | 1.5 | 1.000 ± 0.000 | 0.320 ± 0.014 | 0.0000 ± 0.0000 |
| cond · OT + CFG | 2 | 1.000 ± 0.000 | 0.347 ± 0.027 | 0.0000 ± 0.0000 |
| cond · OT + CFG | 3 | 1.000 ± 0.000 | 0.461 ± 0.062 | 0.0000 ± 0.0000 |
| cond · OT + CFG | 5 | 1.000 ± 0.000 | 1.096 ± 0.227 | 0.0000 ± 0.0000 |
| cond · OT + CFG | 7 | 1.000 ± 0.000 | 2.741 ± 0.489 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 1 | 1.000 ± 0.000 | 0.318 ± 0.014 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 1.5 | 1.000 ± 0.000 | 0.307 ± 0.016 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 2 | 1.000 ± 0.000 | 0.328 ± 0.020 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 3 | 1.000 ± 0.000 | 0.423 ± 0.033 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 5 | 1.000 ± 0.000 | 0.979 ± 0.106 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 7 | 1.000 ± 0.000 | 2.519 ± 0.315 | 0.0000 ± 0.0000 |

- **Both CFG variants now behave alike.** Each is best at guidance 1.5 (0.320 with OT, 0.307 without), which only
  matches the unguided models.
- **Guidance beyond 2 hurts:** 0.42–0.46 at 3, 1.0–1.1 at 5 and 2.5–2.7 at 7.
- **Default guidance:** 3.0 is too high for this task.

### Training

| Variant | Training loss, epoch 50 | Training loss, epoch 100 | Minutes per 100-epoch run (6 in parallel) |
| --- | ---: | ---: | ---: |
| cond · OT + CFG | 0.6053 ± 0.0041 | 0.5604 ± 0.0063 | 8.7 |
| cond · OT, no CFG | 0.3936 ± 0.0020 | 0.3552 ± 0.0026 | 8.7 |
| cond · no OT + CFG | 0.7067 ± 0.0056 | 0.6497 ± 0.0079 | 6.3 |
| cond · no OT, no CFG | 0.4947 ± 0.0051 | 0.4433 ± 0.0048 | 6.3 |
| uncond · OT | 0.2844 ± 0.0046 | 0.2474 ± 0.0014 | 5.7 |
| uncond · no OT | 2.3307 ± 0.0165 | 2.2846 ± 0.0127 | 4.7 |

Loss isn't comparable across settings. OT pairing changes the regression target, and CFG adds unconditional
samples whose targets vary much more.
- **Within a setting:** within-condition OT lowers the loss somewhat (0.394 vs. 0.495 without CFG at epoch 50,
  0.605 vs. 0.707 with CFG).
- **Unconditional OT:** its ~8–9× lower loss is consistent with it removing crossing paths.
- **The buggy pairing:** it gave a *lower* loss (0.16 at epoch 50, no CFG), because it shortened paths by solving
  the wrong problem.
- **Run time:** the retrained runs took 8.7 minutes each, because all six parallel jobs were the heaviest variant.

## Reference: epoch-100 results

Main results for the final checkpoints, evaluated the same way:

| Variant | Guidance scale | Mode accuracy ↑ | Twist distance (data: 0.219) | Mode coverage KL ↓ |
| --- | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 3.0 | 1.000 ± 0.000 | 0.330 ± 0.054 | 0.0000 ± 0.0000 |
| cond · OT, no CFG | 1.0 (unguided) | 1.000 ± 0.000 | 0.289 ± 0.015 | 0.0000 ± 0.0000 |
| cond · no OT + CFG | 3.0 | 1.000 ± 0.000 | 0.346 ± 0.034 | 0.0000 ± 0.0000 |
| cond · no OT, no CFG | 1.0 (unguided) | 1.000 ± 0.000 | 0.288 ± 0.011 | 0.0000 ± 0.0000 |
| uncond · OT | n/a | — | 0.288 ± 0.006 | 0.0083 ± 0.0031 |
| uncond · no OT | n/a | — | 0.278 ± 0.009 | 0.0061 ± 0.0052 |

The full epoch-100 sweeps are in `experiments/results/pose/epoch_100/`.

## Known issues and caveats

- **The conditional task is too easy and several metrics saturate.** See [above](#why-the-task-is-too-easy). The
  CFG and conditional OT conclusions are specific to this task.
- **Twist distance has a floor.** Values below 0.219 mean the samples are too tightly clustered.
- **The steps sweep uses guidance 3.0 for CFG variants.** Their poor few-step results are mostly a guidance effect.
- **Pose training only sees 10 fixed time values** (multiples of 1/9). Sampling with 1, 3 or 9 steps queries only
  time values seen in training; other step counts interpolate in time.
- **Epochs 20 and 30 come from a pilot** with main metrics only, at 100 and 3 steps.
  `EVAL_EPOCH=20 ./pose_ablations.sh eval` produces the full set.
- **5 seeds.** The standard deviations are sample standard deviations over 5 runs. Read differences smaller than
  about 2 standard deviations as unresolved.

## MNIST status

Deferred; not trained. The pipeline is ready (`./mnist_ablations.sh`), and two code fixes also apply to it, though
neither has run in MNIST training yet:

- **No-CFG dropout:** No-CFG models no longer get ~10% unconditional dropout through MNIST's single conditioning
  token.
- **OT pairing:** conditional OT now pairs within each class (`f80364c`).

Other notes:

- **CFG dropout rate:** CFG MNIST models train with ~19% null-token dropout rather than the nominal 10%, because
  per-token and whole-sample dropout stack when there is only one token.
- **Learning rate:** the MNIST trainer uses a warmup + cosine schedule, so an intermediate MNIST checkpoint is *not*
  equivalent to a shorter training run.
- **Cost:** one 400-epoch run takes about 26 minutes, so 5 seeds × 6 variants is about 13 hours sequentially.

## Next steps

1. Tag this state as `pose-easy-v1`.
2. Add metrics that don't saturate, and validate them on the current models:
   - **Partial-conditioning evaluation:** give only one action token, so the target is bimodal.
   - **Bias and spread:** report them separately, with translation and rotation split.
   - **MMD against real goal samples.**
   - **Path straightness.**
   - **Tests:** add `tests/` for the pairing and masking invariants.
3. Generalize the hard-coded 4-mode task into a config. Confirm it reproduces this report's numbers exactly.
4. Train and evaluate a harder pose task in its own folders, with multimodal conditionals, more and closer modes,
   and rotation-differentiated modes. Keep this task as the easy baseline.

## Reproducing

```bash
conda activate FmT
./pose_ablations.sh                     # train 5 seeds × 6 variants (resumable), evaluate the epoch-100 checkpoints
EVAL_EPOCH=50 ./pose_ablations.sh eval  # evaluate the epoch-50 checkpoints (this report's main results)
```

Outputs (gitignored):

- `checkpoints/pose/seed_<N>/`: checkpoints, training configs and logs. Run logs:
  - `checkpoints/pose/pose_ablations_console.txt`: the original training round.
  - `checkpoints/pose/pose_ot_fix_console.txt`: the conditional-OT retrain and current evaluations.
- `experiments/results/pose/epoch_<E>/`: `metrics_summary.csv`, `steps_sweep_summary.csv`, `cfg_sweep_summary.csv`,
  `steps_sweep.png` and `cfg_sweep.png`.
- `experiments/results/pose/epoch_<E>/seed_<N>/`: per-seed `metrics.csv`, `steps_sweep.csv`, `cfg_sweep.csv`,
  `pose_grid.png` (sample trajectories) and `cfg_sweep.png`.
- `checkpoints/pose_before_ot_fix/` and `experiments/results/pose_before_ot_fix/`: the pre-fix conditional-OT runs
  and results.

## Appendix: per-seed results (epoch 50, 100 steps)

**Mode accuracy:**

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cond · OT, no CFG | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cond · no OT + CFG | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cond · no OT, no CFG | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**Twist distance** (real data: 0.219):

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 0.489 | 0.443 | 0.485 | 0.524 | 0.362 |
| cond · OT, no CFG | 0.305 | 0.308 | 0.363 | 0.316 | 0.302 |
| cond · no OT + CFG | 0.396 | 0.448 | 0.407 | 0.468 | 0.395 |
| cond · no OT, no CFG | 0.303 | 0.307 | 0.327 | 0.307 | 0.305 |
| uncond · OT | 0.399 | 0.371 | 0.392 | 0.351 | 0.339 |
| uncond · no OT | 0.361 | 0.313 | 0.330 | 0.324 | 0.302 |

**Mode coverage KL:**

| Variant | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| cond · OT, no CFG | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| cond · no OT + CFG | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| cond · no OT, no CFG | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| uncond · OT | 0.0075 | 0.0137 | 0.0105 | 0.0080 | 0.0084 |
| uncond · no OT | 0.0097 | 0.0137 | 0.0219 | 0.0179 | 0.0110 |
