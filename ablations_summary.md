# What gives the best flow matching models? Ablations on pose generation

**Question:** what do I need to do to get the best samples from the flow matching transformer?

**Scope:** SE(3) pose generation with **discrete conditioning**: each sample is conditioned on action tokens
that name a goal region. Everything below comes from that setting. MNIST (image generation, class
conditioning) hasn't been run yet, so none of this is verified for images.

## Recommendation

**Best configuration on this data:** a conditional model trained with **per-condition minibatch OT**, sampled
at **guidance 1** with **5–9 Euler steps**.

- **Pair noise to data with minibatch OT, within each condition.** At 3 Euler steps it gives 4–8× lower energy
  distance; at 100 steps it's level. It costs ~40% more training time. [Details](#1-ot-fast-sampling-for-40-more-training-time).
- **Don't guide: sample at guidance 1.** Every guidance above 1 makes samples less like the data, and guidance
  costs a second forward pass per step. [Details](#2-guidance-condition-separation-at-the-cost-of-fidelity).
- **CFG training is optional.** Null-token dropout doesn't hurt: at guidance 1, a CFG-trained model matches one
  trained without it. So it only keeps the option of guidance open.
- **With OT, 5–9 Euler steps are enough** (3 if speed matters most). Without OT, the model needs ~20.
- **Train for at least 100 epochs, and probably longer.** Energy distance flattens by epoch 30–50, but on the
  two-orientations task the excess spread is still shrinking at epoch 100. [Details](#3-training).

**What no setting fixes yet:** every conditional model spreads its samples **about 1.15–1.5× wider than the
data**. That's the main visible reason the best energy distance is still 10–40× the real-data floor. See
[Open problem](#open-problem-over-dispersion).

## The evidence at a glance

![Energy distance per configuration](docs/ablations/overview.png)

*Energy distance to real data (log scale, lower is better) for the four conditional configurations, at 3 and 100
sampling steps. One dot per seed; the bar is the mean. CFG-trained models are sampled at the default guidance of 3.
Epoch 100.*

- **OT, no CFG wins in every panel.** It matters most at 3 steps: 0.029 vs. 0.112 without OT (two orientations)
  and 0.015 vs. 0.118 (within 3σ).
- **At 100 steps, OT and no OT are level** (0.024 vs. 0.027; 0.008 vs. 0.012).
- **Guidance 3 is the worst choice everywhere:** 4–10× worse than unguided at 100 steps, and catastrophic at 3
  steps on the first task (3.1–3.7).

## Tasks and setup

Two pose tasks, both with **bimodal conditions**: the tokens name a corner, and each corner holds two goal
orientations. A good model must keep both orientations rather than averaging them.

- **Two orientations per corner** (`configs/pose_tasks/corners_two_orientations.yaml`): four corners 10 units apart,
  each with two orientations rotated ±0.25 rad about z (5σ apart). Picking the corner is easy; keeping both
  orientations isn't.
- **Every mode within 3σ** (`configs/pose_tasks/corners_3sigma.yaml`): the same 8 modes packed so every pair is
  within 3σ (σ = 0.1) in position and orientation. The corners overlap, so even real data can't be sorted
  perfectly: only **73.4%** of real samples are nearest to a mode of their own corner.

The original four-corners task is left out: its modes are far apart and every model solves it. Its results, and
the history of this study (including a conditional-OT pairing bug that was found and fixed), are in the previous
version of this file at tag `pose-two-orientations-v1`.

**Variants:** 2 × 2 conditional models (OT or not × CFG or not), plus unconditional models with and without OT.
Each has 5 seeds and trains for 100 epochs with identical hyperparameters:
- batch 128;
- AdamW at a constant LR of 1e-4;
- for CFG models, 10% null-token dropout per sample plus 10% per token.

Samples use Euler integration from a unit Gaussian in twist space. The evaluation uses 256 samples per model
(64 per condition), with the same start noise for every model.

**Metrics:**
- **Energy distance** (primary): distance between the sample distribution and real data, per goal mode. Real
  data scores about 0.001. It catches collapse, excess spread and bias alike.
- **Corner accuracy:** the share of samples nearest to a mode of the requested corner. On the 3σ task it has a
  ceiling of 0.734, and **higher is not better**: a sampler above it separates the corners more than the data
  does.
- **Orientation split:** how each condition's samples divide between its two orientations (real data: 50/50 ±
  sampling noise).
- **Spread ratio:** the samples' spread around their mode, relative to real data's.

## 1. OT: fast sampling for ~40% more training time

![Energy distance vs. sampling steps](docs/ablations/steps_sweep.png)

*Energy distance vs. number of Euler steps (both axes log). Mean over 5 seeds; shading = seed range. Top:
conditional models (dashed = CFG-trained, sampled at guidance 3). Bottom: unconditional models.*

- **With OT, the model converges in few steps.** On the two-orientations task, 3 steps get within 1.2× of its
  100-step quality and 9 steps match it. On the 3σ task, 3 steps are within 2× and 9 steps within 1.3×.
- **Without OT it needs about 20 steps** before it catches up.
- **Unconditional models show the same pattern:** at 3 steps, energy distance is 0.046 vs. 0.69 and 0.014 vs.
  0.14.
- **The guided models' few-step failure is caused by guidance, not by OT:** both OT and no-OT CFG models are bad.

![Orientation per sample](docs/ablations/orientation_violins.png)

*Each sample's rotation about z relative to the midpoint between its corner's two orientations (dotted lines =
the two targets). Models trained without CFG; 128 samples per condition × 4 conditions × 5 seeds per violin.*

**Why OT matters here:**
- Without OT, the model learns the *average* velocity toward the two orientations, so few-step samples land
  *between* them. At 1–3 steps, the no-OT violins are single narrow lumps at 0, while real data is bimodal.
- OT pairs each noise sample with a specific orientation during training: noise with positive rotation about z goes
  to the positive orientation. The learned flow therefore splits early, and even 1-step samples keep both
  orientations.
- At 100 steps the two are the same; both are wider than the data (see
  [Open problem](#open-problem-over-dispersion)).

**Costs:**
- **Training time:** OT computes a Hungarian assignment within each condition for every minibatch. Conditional
  runs took ~700 s with OT vs. ~500 s without (6 runs in parallel on one RTX 4090).
- **Sampling:** no cost. OT changes only training.
- **Implementation:** pair within each condition. Pairing across the whole mixed-condition batch biases each
  condition's noise toward its own goal, and it hurt conditional models in an earlier round.

## 2. Guidance: condition separation at the cost of fidelity

![Guidance sweep](docs/ablations/guidance_sweep.png)

*CFG-trained models sampled at guidance 1–7, 100 steps. Lines = mean over seeds (shading = range). The separate
markers at guidance 1 are the models trained without CFG. Right: each condition's split between its two
orientations, one dot per condition × seed; grey band = the 95% range for a perfect sampler with 64 samples.*

![Where guidance puts samples](docs/ablations/guidance_violins.png)

*Each sample's position along its corner's outward direction (away from the other corners), relative to the
corner. OT + CFG models, 128 samples per condition × 4 conditions × 5 seeds per violin.*

Guidance extrapolates away from the unconditional prediction. Here that means pushing each sample away from the
other conditions:

- **Samples move outward, past their corner.**
  - On the 3σ task, the median sample sits 0.15 beyond its corner at guidance 3, as far again as the corner is
    from the cluster centre, and 0.27 beyond at guidance 7.
  - On the two-orientations task, the median sits 0.18 beyond at guidance 3 and 0.77 beyond at guidance 7, where
    4% of samples land more than 3 units out.
- **Corner accuracy rises above what real data achieves:** 0.74 → 0.91 → 0.98 → 0.995 at guidance 1 / 1.5 / 2 / 3
  on the 3σ task, against 0.734 for the data. The samples become *easier to classify than real data*. That is a
  distortion, not an improvement.
- **The orientation split gets lopsided, where orientations differ between conditions.**
  - On the two-orientations task, each corner's orientations are its own, and at guidance 3 several conditions
    split 25/75.
  - On the 3σ task every corner shares the same two orientations, so there's nothing in rotation for guidance to
    amplify, and the split stays balanced.
- **Energy distance gets steadily worse with guidance on both tasks.** At guidance 1, a CFG-trained model matches
  one trained without CFG: 0.022–0.033 vs. 0.024–0.027, and 0.007–0.012 vs. 0.008–0.012.

**When guidance could still be worth it:** when you want samples that are unambiguous about their condition
more than samples that match the data. Even then, 1.5 is the most to try. Guidance also doubles the per-step
compute, because it needs a second, unconditional pass.

## 3. Training

![Training curves](docs/ablations/training_curves.png)

*Left: training loss per epoch. Then energy distance at 100 and 3 sampling steps, and position spread relative
to real data, evaluated every 10 epochs. Mean over 5 seeds; shading = seed range. CFG-trained models are sampled
at guidance 3.*

- **Energy distance converges early.**
  - On the 3σ task it is flat from about epoch 30, and the epoch-to-epoch wobble is seed noise.
  - On the two-orientations task it falls until about epoch 40–50, then improves only slowly.
- **The few-step advantage of OT appears early:** by epoch 10 on the 3σ task and by epoch 30–40 on the
  two-orientations task. It holds through epoch 100.
- **Over-dispersion is still shrinking at epoch 100 on the two-orientations task:** position spread falls from
  4.6× at epoch 10 to 1.3× at epoch 100 and hasn't levelled off. On the 3σ task it settles at 1.0–1.15× by
  epoch 20.
- **OT's training loss is lower,** but that doesn't mean its model is better. OT pairing makes the regression
  target less noisy, so the losses of OT and no-OT runs aren't comparable. Only the shape of each curve is
  informative.
- **Guidance 3 shrinks the spread below the data's** on the 3σ task (0.75–0.82×), the other side of its outward
  push.

Practical reading: 100 epochs is enough for energy distance on both tasks. The remaining spread error on the
two-orientations task looks like a training-length problem, so longer training is the cheapest next
experiment.

## 4. What the samples look like

![3σ task: start-to-goal mappings](docs/ablations/mappings_3d_3sigma.png)

*3σ task, seed 1, epoch 100: start → goal paths near the goal region for each model, at 3 and 100 steps.
Every model gets the same start poses. At guidance 3 (middle row), each corner's samples sit outside the corner,
away from the other three. Without OT at 3 steps (top row, second panel), samples bunch tightly on the corners
instead of spreading like the data.*

![Two orientations: rotation paths](docs/ablations/rotation_paths_3d_two_orientations.png)

*Two-orientations task, seed 1, epoch 100: the final approach of one condition's sampling paths in rotation space,
relative to the midpoint between its two orientations (stars). Without OT at 3 steps, 44% of the endpoints end
between the stars, vs. 16% with OT and 9% of real samples. At 100 steps it's 14–23% for every model.*

## Open problem: over-dispersion

Every conditional model spreads its samples too widely at 100 steps:
- **Two orientations:** 1.28–1.31× in position and 1.33–1.45× in rotation, against the data's 0.91 / 0.96.
- **Within 3σ:** 1.04–1.15× / 1.06–1.17×, against 0.91 / 0.92.

This is the main visible reason the best energy distance is 10–40× the real-data floor, and none of the ablated
components fixes it. Candidates to try next, cheapest first:

- **Longer training.** On the two-orientations task, the spread is still shrinking at epoch 100 (see
  [Training](#3-training)).
- **Learning-rate decay or EMA weights.** The pose trainer uses a constant LR of 1e-4 with no EMA, so the final
  weights carry optimisation noise. Both are standard in flow matching.
- **Continuous training times.** Training only sees 10 fixed time values (multiples of 1/9), so the velocity
  field between them is interpolated.

## Caveats

- **5 seeds per configuration.** Treat differences inside the seed range (the shading and dot spread) as
  unresolved.
- **Small per-condition samples:** 64 samples per condition, so orientation splits within about ±0.12 of 50/50
  are sampling noise.
- **The steps sweep samples CFG-trained models at guidance 3.** Their few-step results mostly reflect guidance;
  at guidance 1 they behave like the models trained without CFG.
- **The training times are rough:** they were measured with 6 runs sharing one GPU.

## Reproducing

```bash
conda activate FmT
for t in corners_two_orientations corners_3sigma; do
  TASK_CONFIG=configs/pose_tasks/$t.yaml ./pose_ablations.sh          # train 5 seeds × 6 variants, evaluate epoch 100
  TASK_CONFIG=configs/pose_tasks/$t.yaml ./pose_ablations.sh curves   # metrics at every saved epoch
  python pose_gen_inference.py --visualize_task -CP checkpoints/pose_$t/seed_1/ -CE 100 -N 12   # 3-D figures
done
python experiments/summary_figures.py     # the figures in this file -> docs/ablations/
python -m unittest discover -s tests -t . # pairing, masking, metric and task-file tests
```

Results land in `experiments/results/pose_<task>/` (gitignored): `epoch_<E>/seed_<N>/` holds the per-seed metrics
and sweeps, and `training_curves/` the per-epoch metrics. Checkpoints are in `checkpoints/pose_<task>/seed_<N>/`.
The two tasks were trained at `5f77177` (two orientations) and `af9cf5d` (3σ); every result row records the commit
it was evaluated at.
