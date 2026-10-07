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
- The original metrics were evaluated at `f80364c`.
- The [new metrics](#new-metrics) were evaluated at `0a1180a` (sweeps) and `af13ac3` (main metrics and the real-data
  row). The original metrics reproduce exactly at both.
- Result rows from now on record their commit in a `git_commit` column.
- Since `3ac6e51`, this task is defined in `configs/pose_tasks/four_corners.yaml` rather than hard-coded.
  Retraining two runs from the file reproduced their checkpoints bit for bit, and re-evaluating every epoch-50
  model matched all 10,445 result cells.

The study asks what two components add to the flow matching transformer:

- **OT**: minibatch optimal-transport pairing between noise samples and data samples during training.
- **CFG**: classifier-free guidance. Training randomly replaces the conditioning with a learned null token, and
  sampling extrapolates away from the unconditional prediction.

## Key findings (epoch 50)

1. **OT makes few-step sampling work for unconditional models.**
   - At 1 step, the OT model lands 1.06 from the nearest mode versus 6.99 without OT (energy distance to real data
     0.28 vs. 12.9). At 3 steps it's 0.45 vs. 1.45.
   - OT's sampling paths are straighter: path length ÷ start-to-end distance is 1.001 vs. 1.037 at 100 steps, and
     1.006 vs. 1.178 at 2 steps.
   - Both converge by about 9 steps. With many steps, no-OT is slightly ahead.
2. **With correct pairing, conditional OT matches no-OT at 100 steps, but stops few-step samples collapsing.**
   - At 100 steps, the two are within 1 seed standard deviation on every metric.
   - At 2–3 steps, samples without OT are too tightly clustered (spread 0.71–0.72× the data), while with OT the
     spread is right (0.94–1.03×), 3–4.5 standard deviations apart.
   - Twist distance had scored that collapse as an *improvement*. The large OT penalty in the first round was a
     pairing bug, now [fixed](#conditional-ot-pairing-fix).
3. **Guidance never helps on this task.**
   - Under energy distance, both CFG-trained models are best unguided (0.029–0.035), matching the no-CFG models
     (0.031–0.040). Every increase in guidance makes them worse: 0.18–0.24 at the default guidance of 3.0.
   - Twist distance had preferred guidance 1.5 only because guidance tightens the spread while adding bias.
4. **Guidance 3.0 breaks few-step sampling.** At 1 step, both CFG variants land 15.0 from their target. That is
   close to what a perfect model would do on this task (see [below](#why-the-task-is-too-easy)).
5. **The models are over-dispersed.** At 100 steps, samples spread 1.45–2.0× wider than the data in translation
   at epoch 50, and 1.2–1.4× at epoch 100. This fine-scale structure is the one part of the task that isn't
   saturated.
6. **Everything else about the task is too easy.**
   - Every conditional variant scores 100% mode accuracy.
   - One-token conditioning, which has a bimodal target, is handled perfectly (100% valid, 50/50 split at the noise
     floor).
   - One-step results match hand calculations for a perfect model.

   The next phase is a harder pose task ([Next steps](#next-steps)).

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
- **The original metrics have blind spots of their own:**
  - Mode accuracy only catches a model that ignores its condition.
  - Coverage KL for conditional models only echoes accuracy, because the evaluation batch is balanced.
  - Twist distance rewards collapse (floor 0.219) and adds translation and rotation in one norm.
  - Nothing measured path straightness directly.

  The [new metrics](#new-metrics) address each of these.
- **Bimodal conditionals are easy too.** With one action token hidden, each condition has two valid modes. All
  models put 100% of samples in a valid mode and split them 50/50 to within sampling noise, because the modes are
  still 10 units apart.

## New metrics

These were added to measure what the original metrics couldn't: distribution match, collapse versus
over-dispersion, path geometry, and multimodal conditioning. Each evaluation run also scores **real goal
samples** with the same code, giving a "real data" row: what a perfect sampler gets.

- **Energy distance** to real goal samples, computed within each mode (the target mode for conditional models,
  the nearest mode otherwise) and weighted by sample count. It is 0 when the distributions match and grows with any
  mismatch in location or spread, so collapse is penalized.
  - Poses are embedded as position plus rotation matrix / √2, so small rotation differences count about like
    radians.
  - It is computed per mode because over the whole mixture, the ~10-unit gaps between modes dilute within-mode
    errors about 4×. A unit test shows this.
- **Bias and spread**, split into translation (world units) and rotation (radians), relative to each mode center.
  Spread is reported as a ratio to the real data's: 1 matches, below 1 is collapse, above 1 is over-dispersion.
- **Path straightness**: sampling-path length divided by start-to-end distance, both as twist norms. Paths along
  the training interpolation score exactly 1.
- **Transport cost**: the mean start-to-end distance.
- **One-token conditioning:** one of the two action tokens is replaced by the null token, as in training, so two
  modes are valid. Three measures:
  - Valid-mode rate: the share of samples nearest a valid mode.
  - Imbalance KL: how unevenly samples split between the two valid modes. With 64 samples, a perfect random sampler
    scores about 0.007.
  - Energy distance within the valid modes.

The real-data reference samples use their own random-number generators, so adding these metrics changed none of
the original numbers. Unit tests in `tests/` check each metric against real data and against the failure it targets.

### Validation against known behavior

Before trusting the metrics, they were checked on behavior that is already understood, including the
[hand-calculated](#why-the-task-is-too-easy) one-step outcomes:

| Known behavior | Expected | Measured (epoch 50) |
| --- | ---: | ---: |
| Real goal samples (perfect sampler) | energy distance ≈ 0; spread ratio ≈ 1 | energy distance 0.0015; spread 0.98 / 0.97 (translation / rotation) |
| 1 step, unconditional, no OT: collapse to the mode average | bias ≈ √50 ≈ 7.07 | bias 6.90; energy distance 12.9 |
| 1 step, guidance 3.0: overshoot | bias ≈ 2 × 7.07 ≈ 14.1 | bias 14.49 (OT) / 14.53 (no OT) |
| 2–3 steps, plain conditional: too tight (twist distance 0.16, below the 0.219 floor) | spread < 1; worse than at 100 steps | spread 0.72 / 0.71; energy distance 0.074 vs. 0.031 at 100 steps |
| OT should straighten unconditional paths | straightness closer to 1 with OT | 1.001 vs. 1.037 at 100 steps; 1.006 vs. 1.178 at 2 steps |
| Nothing to straighten within one condition | straightness ≈ 1 with or without OT | 1.001 (OT) / 1.002 (no OT) |

### Results (epoch 50, 100 steps)

| Variant | Energy distance ↓ | Bias, translation ↓ | Bias, rotation (rad) ↓ | Spread ÷ data, translation | Spread ÷ data, rotation | Path straightness | Transport cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 0.239 ± 0.092 | 0.345 ± 0.092 | 0.077 ± 0.009 | 2.03 ± 0.16 | 1.27 ± 0.11 | 1.020 ± 0.001 | 9.39 ± 0.08 |
| cond · OT, no CFG | 0.040 ± 0.027 | 0.091 ± 0.045 | 0.052 ± 0.010 | 1.54 ± 0.06 | 1.26 ± 0.05 | 1.001 ± 0.000 | 9.11 ± 0.05 |
| cond · no OT + CFG | 0.182 ± 0.046 | 0.285 ± 0.048 | 0.089 ± 0.016 | 1.80 ± 0.08 | 1.63 ± 0.12 | 1.021 ± 0.001 | 9.31 ± 0.05 |
| cond · no OT, no CFG | 0.031 ± 0.009 | 0.082 ± 0.023 | 0.053 ± 0.014 | 1.45 ± 0.06 | 1.45 ± 0.04 | 1.002 ± 0.000 | 9.09 ± 0.05 |
| uncond · OT | 0.076 ± 0.027 | 0.137 ± 0.036 | 0.059 ± 0.003 | 1.81 ± 0.07 | 1.17 ± 0.09 | 1.001 ± 0.000 | 8.23 ± 0.07 |
| uncond · no OT | 0.050 ± 0.010 | 0.105 ± 0.010 | 0.056 ± 0.014 | 1.49 ± 0.09 | 1.26 ± 0.10 | 1.037 ± 0.002 | 8.24 ± 0.01 |
| real data | 0.001 ± 0.000 | 0.026 ± 0.000 | 0.023 ± 0.000 | 0.98 ± 0.00 | 0.97 ± 0.00 | — | — |

- **Over-dispersion:** every model's samples are too wide, by 1.45–2.0× in translation and 1.2–1.6× in rotation.
  Guidance 3.0 makes it worse and adds bias, which is why the CFG variants have the largest energy distances.
- **Conditional, no CFG:** OT and no-OT are level (0.040 ± 0.027 vs. 0.031 ± 0.009).
- **Unconditional:** no-OT is slightly closer than OT at this many steps (0.050 vs. 0.076, about 1.3 seed SDs). At
  epoch 100 they are level.
- **Transport cost doesn't move:** 8.23 with OT vs. 8.24 without. All four modes are almost equally far from the
  start distribution, so pairing can't shorten paths much. OT's benefit shows up as straighter, non-crossing paths
  instead.

**One-token conditioning (bimodal target):**

| Variant | Valid-mode rate ↑ | Imbalance KL (noise floor ≈ 0.007) | Energy distance ↓ |
| --- | ---: | ---: | ---: |
| cond · OT + CFG | 1.000 ± 0.000 | 0.016 ± 0.012 | 0.143 ± 0.031 |
| cond · OT, no CFG | 1.000 ± 0.000 | 0.009 ± 0.002 | 0.046 ± 0.026 |
| cond · no OT + CFG | 1.000 ± 0.000 | 0.012 ± 0.008 | 0.107 ± 0.029 |
| cond · no OT, no CFG | 1.000 ± 0.000 | 0.009 ± 0.005 | 0.046 ± 0.013 |
| real data | 1.000 ± 0.000 | 0.007 ± 0.000 | 0.001 ± 0.000 |

Every model puts all samples in a valid mode and splits them evenly. The no-CFG models sit at the noise floor; the
CFG models are slightly above it. Guidance again hurts within-mode quality (energy distance 0.11–0.14 vs. 0.05).

### New metrics vs. integration steps (epoch 50)

**Energy distance** (real data: 0.001):

| Variant | 1 step | 2 steps | 3 steps | 5 steps | 9 steps | 20 steps | 50 steps | 100 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 26.538 ± 0.405 | 10.263 ± 0.344 | 4.845 ± 0.325 | 1.393 ± 0.223 | 0.514 ± 0.136 | 0.325 ± 0.111 | 0.258 ± 0.097 | 0.239 ± 0.092 |
| cond · OT, no CFG | 0.072 ± 0.024 | 0.045 ± 0.026 | 0.044 ± 0.028 | 0.037 ± 0.026 | 0.036 ± 0.026 | 0.038 ± 0.026 | 0.039 ± 0.026 | 0.040 ± 0.027 |
| cond · no OT + CFG | 26.622 ± 0.355 | 10.186 ± 0.288 | 4.631 ± 0.181 | 1.168 ± 0.128 | 0.381 ± 0.075 | 0.243 ± 0.056 | 0.195 ± 0.048 | 0.182 ± 0.046 |
| cond · no OT, no CFG | 0.093 ± 0.033 | 0.074 ± 0.017 | 0.076 ± 0.016 | 0.054 ± 0.016 | 0.033 ± 0.011 | 0.029 ± 0.010 | 0.030 ± 0.010 | 0.031 ± 0.009 |
| uncond · OT | 0.276 ± 0.038 | 0.109 ± 0.029 | 0.092 ± 0.033 | 0.085 ± 0.033 | 0.081 ± 0.031 | 0.078 ± 0.029 | 0.077 ± 0.028 | 0.076 ± 0.027 |
| uncond · no OT | 12.944 ± 0.062 | 4.935 ± 0.523 | 0.943 ± 0.070 | 0.252 ± 0.016 | 0.114 ± 0.013 | 0.067 ± 0.008 | 0.053 ± 0.009 | 0.050 ± 0.010 |

**Translation spread ÷ data** (1 = matches data, < 1 = collapsed):

| Variant | 1 step | 2 steps | 3 steps | 5 steps | 9 steps | 20 steps | 50 steps | 100 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 8.36 ± 1.25 | 3.66 ± 0.20 | 3.57 ± 0.21 | 3.26 ± 0.15 | 2.50 ± 0.18 | 2.18 ± 0.17 | 2.06 ± 0.17 | 2.03 ± 0.16 |
| cond · OT, no CFG | 1.32 ± 0.07 | 0.94 ± 0.08 | 1.03 ± 0.08 | 1.21 ± 0.07 | 1.36 ± 0.07 | 1.47 ± 0.06 | 1.52 ± 0.06 | 1.54 ± 0.06 |
| cond · no OT + CFG | 8.35 ± 1.07 | 3.79 ± 0.20 | 3.65 ± 0.19 | 3.22 ± 0.15 | 2.29 ± 0.12 | 1.94 ± 0.10 | 1.83 ± 0.09 | 1.80 ± 0.08 |
| cond · no OT, no CFG | 1.24 ± 0.06 | 0.72 ± 0.05 | 0.71 ± 0.06 | 0.92 ± 0.08 | 1.17 ± 0.08 | 1.34 ± 0.07 | 1.42 ± 0.06 | 1.45 ± 0.06 |
| uncond · OT | 8.03 ± 0.22 | 4.52 ± 0.60 | 3.31 ± 0.69 | 2.48 ± 0.44 | 2.04 ± 0.22 | 1.85 ± 0.12 | 1.82 ± 0.08 | 1.81 ± 0.07 |
| uncond · no OT | 1.26 ± 0.19 | 11.19 ± 0.24 | 7.98 ± 0.57 | 3.77 ± 0.51 | 1.91 ± 0.22 | 1.53 ± 0.10 | 1.50 ± 0.11 | 1.49 ± 0.09 |

**Path straightness** (1 = straight):

| Variant | 1 step | 2 steps | 3 steps | 5 steps | 9 steps | 20 steps | 50 steps | 100 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 1.000 ± 0.000 | 1.065 ± 0.003 | 1.072 ± 0.003 | 1.051 ± 0.002 | 1.033 ± 0.002 | 1.025 ± 0.001 | 1.021 ± 0.001 | 1.020 ± 0.001 |
| cond · OT, no CFG | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.001 ± 0.000 | 1.001 ± 0.000 |
| cond · no OT + CFG | 1.000 ± 0.000 | 1.063 ± 0.004 | 1.071 ± 0.003 | 1.053 ± 0.002 | 1.034 ± 0.002 | 1.026 ± 0.001 | 1.022 ± 0.001 | 1.021 ± 0.001 |
| cond · no OT, no CFG | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.001 ± 0.000 | 1.001 ± 0.000 | 1.002 ± 0.000 | 1.002 ± 0.000 | 1.002 ± 0.000 |
| uncond · OT | 1.000 ± 0.000 | 1.006 ± 0.000 | 1.005 ± 0.000 | 1.004 ± 0.000 | 1.003 ± 0.000 | 1.002 ± 0.000 | 1.001 ± 0.000 | 1.001 ± 0.000 |
| uncond · no OT | 1.000 ± 0.000 | 1.178 ± 0.010 | 1.152 ± 0.006 | 1.099 ± 0.003 | 1.064 ± 0.002 | 1.047 ± 0.002 | 1.039 ± 0.002 | 1.037 ± 0.002 |

- **Few-step collapse without OT (conditional):** at 2–3 steps the plain conditional model collapses to 0.71–0.72×
  the data's spread, and its energy distance (0.074–0.076) is worse than at 100 steps.
  - With OT the spread stays at 0.94–1.03×, and energy distance is lower (0.044–0.045, about 1.4 SD).
  - Without OT, a few Euler steps average over the condition's goals and land near their mean. An OT coupling ties
    each start to a specific goal, so the within-mode structure survives even a single large step.
- **Unconditional:** OT beats no-OT on energy distance up to 9 steps (0.081 vs. 0.114 at 9), and no-OT is slightly
  ahead from 20 steps on.
  - Without OT, paths curve most at 2–3 steps (straightness 1.15–1.18), where samples are still between modes.
- **Spread vs. step count:** for the conditional no-CFG models, spread grows from 0.7–1.0× at 2–3 steps to
  1.4–1.5× at 100 steps. The unconditional models instead start very wide at 1–2 steps (samples between modes)
  and narrow to 1.5–1.8×.

### Guidance sweep with the new metrics (100 steps)

| Variant | Guidance scale | Energy distance ↓ | Bias, translation ↓ | Spread ÷ data, translation | One-token energy distance ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 1 | 0.035 ± 0.013 | 0.076 ± 0.024 | 1.59 ± 0.03 | 0.038 ± 0.012 |
| cond · OT + CFG | 1.5 | 0.060 ± 0.026 | 0.132 ± 0.043 | 1.47 ± 0.04 | 0.050 ± 0.018 |
| cond · OT + CFG | 2 | 0.106 ± 0.044 | 0.195 ± 0.062 | 1.55 ± 0.07 | 0.075 ± 0.022 |
| cond · OT + CFG | 3 | 0.239 ± 0.092 | 0.345 ± 0.092 | 2.03 ± 0.16 | 0.143 ± 0.031 |
| cond · OT + CFG | 5 | 0.914 ± 0.326 | 0.984 ± 0.232 | 4.71 ± 0.68 | 0.405 ± 0.047 |
| cond · OT + CFG | 7 | 3.196 ± 0.898 | 2.564 ± 0.483 | 8.39 ± 1.15 | 1.030 ± 0.153 |
| cond · no OT + CFG | 1 | 0.029 ± 0.010 | 0.072 ± 0.008 | 1.51 ± 0.05 | 0.028 ± 0.011 |
| cond · no OT + CFG | 1.5 | 0.046 ± 0.015 | 0.112 ± 0.024 | 1.35 ± 0.04 | 0.036 ± 0.012 |
| cond · no OT + CFG | 2 | 0.080 ± 0.024 | 0.164 ± 0.033 | 1.39 ± 0.03 | 0.054 ± 0.016 |
| cond · no OT + CFG | 3 | 0.182 ± 0.046 | 0.285 ± 0.048 | 1.80 ± 0.08 | 0.107 ± 0.029 |
| cond · no OT + CFG | 5 | 0.720 ± 0.134 | 0.841 ± 0.107 | 4.31 ± 0.52 | 0.310 ± 0.072 |
| cond · no OT + CFG | 7 | 2.808 ± 0.436 | 2.324 ± 0.309 | 7.79 ± 0.88 | 0.835 ± 0.154 |

Both CFG models are best at guidance 1 (no guidance) on every distribution metric. Guidance tightens spread slightly
up to 1.5 but adds bias faster, so energy distance rises steadily. Unguided, the CFG-trained models (0.029–0.035)
match the no-CFG models (0.031–0.040), so CFG *training* costs nothing here; only guidance does.

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
  unconditional models only (see [above](#why-the-task-is-too-easy)). A perfect random sampler scores about 0.004
  with 256 samples.
- **Distribution-level metrics** (energy distance, bias/spread, path straightness, one-token conditioning): see
  [New metrics](#new-metrics).

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

**New metrics at epoch 100 (100 steps):**

| Variant | Energy distance ↓ | Bias, translation ↓ | Bias, rotation (rad) ↓ | Spread ÷ data, translation | Spread ÷ data, rotation | Path straightness | Transport cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cond · OT + CFG | 0.137 ± 0.077 | 0.216 ± 0.087 | 0.063 ± 0.017 | 1.32 ± 0.06 | 1.02 ± 0.08 | 1.020 ± 0.001 | 9.25 ± 0.09 |
| cond · OT, no CFG | 0.029 ± 0.019 | 0.086 ± 0.032 | 0.043 ± 0.011 | 1.31 ± 0.02 | 1.19 ± 0.03 | 1.000 ± 0.000 | 9.12 ± 0.02 |
| cond · no OT + CFG | 0.156 ± 0.045 | 0.236 ± 0.051 | 0.065 ± 0.025 | 1.19 ± 0.12 | 1.18 ± 0.08 | 1.022 ± 0.001 | 9.28 ± 0.04 |
| cond · no OT, no CFG | 0.021 ± 0.010 | 0.072 ± 0.020 | 0.043 ± 0.013 | 1.26 ± 0.06 | 1.38 ± 0.04 | 1.002 ± 0.000 | 9.09 ± 0.04 |
| uncond · OT | 0.025 ± 0.009 | 0.069 ± 0.013 | 0.043 ± 0.004 | 1.40 ± 0.03 | 1.07 ± 0.01 | 1.001 ± 0.000 | 8.27 ± 0.03 |
| uncond · no OT | 0.023 ± 0.005 | 0.077 ± 0.008 | 0.040 ± 0.009 | 1.22 ± 0.04 | 1.21 ± 0.05 | 1.038 ± 0.002 | 8.28 ± 0.03 |
| real data | 0.001 ± 0.000 | 0.026 ± 0.000 | 0.023 ± 0.000 | 0.98 ± 0.00 | 0.97 ± 0.00 | — | — |

From epoch 50 to 100, over-dispersion shrinks (translation spread 1.45–2.0× → 1.2–1.4×) and energy distance
falls by 30–70% for the unguided models. So training is still refining the fine-scale structure at epoch 100.

The full epoch-100 sweeps are in `experiments/results/pose/epoch_100/`.

## Known issues and caveats

- **The conditional task is too easy and several metrics saturate.** See [above](#why-the-task-is-too-easy). The
  CFG and conditional OT conclusions are specific to this task.
- **Twist distance has a floor.** Values below 0.219 mean the samples are too tightly clustered. Prefer energy
  distance and the spread ratio.
- **Small samples for one-token conditioning.** There are 64 samples per one-token condition, so imbalance KL
  values below about 0.01 are within sampling noise.
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

1. ~~Verify the conditional-OT fix~~: done (tag `pose-easy-v1`).
2. ~~Add and validate metrics that don't saturate~~: done ([New metrics](#new-metrics), tag `pose-easy-v2`).
3. ~~Generalize the hard-coded 4-mode task into a config~~: done (tag `pose-task-config-v1`).
   - Tasks are YAML files in `configs/pose_tasks/`: start distribution, token vocabulary, and goal modes named by
     token lists. Modes that share a token list form one multimodal condition.
   - Evaluation reads each run's conditions from its training config.
   - The four-corners file reproduces this report exactly (bit-identical checkpoints, all result cells equal).
4. Train and evaluate a harder pose task in its own folders, keeping this task as the easy baseline. What the new
   metrics suggest it needs:
   - **Closer modes** (a few σ apart), so one-token conditioning and coverage stop being trivial.
   - **Multimodal conditionals**, where guidance has something to sharpen.
   - **Rotation-differentiated modes.**
   - **Few-step evaluation as a primary regime**, where OT matters.

## Reproducing

```bash
conda activate FmT
./pose_ablations.sh                     # train 5 seeds × 6 variants (resumable), evaluate the epoch-100 checkpoints
EVAL_EPOCH=50 ./pose_ablations.sh eval  # evaluate the epoch-50 checkpoints (this report's main results)
python -m unittest discover -s tests -t .   # pairing, masking, metric and task-file tests

# Another task: its own checkpoints/pose_<task>/ and experiments/results/pose_<task>/ folders
TASK_CONFIG=configs/pose_tasks/<task>.yaml ./pose_ablations.sh
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
