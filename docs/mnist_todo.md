# MNIST runs to do

Everything below is deferred: the code supports it, but nothing here has been run on the current
model and training framework (adaLN-Zero, RMSNorm + QK-norm, EMA, gradient clipping, continuous
time; readme section 8). The pose results these mirror are in
[ablations_summary.md](../ablations_summary.md).

Checkpoints from before this framework (`checkpoints/mnist/`, `checkpoints/pre_ablation/`) can't be
loaded by the current code. Check out tag `pose-continuous-v1` to evaluate them.

## Cost

- One 400-epoch run (the driver's default) took ~26 min on the RTX 4090 with the old model, run
  sequentially. The new model is about the same size.
- A 5-seed × 6-variant grid is therefore ~13 h at `JOBS=1`.
- **Measure the parallel speedup first** (`JOBS=2`, `JOBS=3` on one seed). For pose, `JOBS=6` gave
  ~2.7× the sequential throughput; MNIST's speedup has never been measured.

## 1. OT × CFG grid on the new framework

```bash
conda activate FmT
./mnist_ablations.sh            # trains the classifier oracle if missing, then 6 variants x 5 seeds, then evaluates
```

- Trains the evaluation classifier (`eval_assets/mnist_cnn.pt`) first. It doesn't exist yet.
- Output: `checkpoints/sota/mnist/seed_<N>/` and `experiments/results/sota/mnist/epoch_400/`.
- MNIST uses a warmup + cosine LR, so intermediate checkpoints are **not** equivalent to shorter
  runs. Evaluate at the final epoch only.
- EMA decay is 0.9999, sized for the 80k optimizer steps of a 400-epoch run (200 batches per epoch).
  Shorter runs need a smaller decay (horizon ≈ 1 / (1 − decay) steps).

**Before relying on the results:** the MNIST metrics are class accuracy and the class-marginal KL
from the classifier oracle. Neither measures sample quality within a class. Consider adding a
distribution metric (e.g. a Fréchet distance on the classifier's penultimate features) first; the
pose study's main metric is a distribution distance (energy distance).

## 2. Single-axis ablations

Pick the baseline from step 1 with the pose study's rule: among the conditional no-CFG variants,
the best on the main metric. Then:

```bash
ABLATE_BASE="cond_image_flow_matching_model_OT_NOCFG|--conditional --no_cfg" ./mnist_ablations.sh ablate
```

Arms (one change each, 5 seeds, under `checkpoints/sota/mnist/ablate/<arm>/`):

| Arm | Change |
|---|---|
| `t_logit_normal`, `t_beta` | training-time density (uniform is the baseline) |
| `time_x1000`, `time_fourier` | flow-time featurisation (sinusoids of t is the baseline) |
| `cond_cross_attn`, `cond_joint` | class-token pathway (adaLN is the baseline) |

The `cond_cross_attn` arm is expected to lose. With one class token, cross-attention collapses to
adding one vector per sample, because attention over one key always has weight 1.

~13 h at `JOBS=1` for the six arms.

## 3. Eval-only sweeps (no retraining)

```bash
./mnist_ablations.sh solver     # Euler vs midpoint vs Heun at 2-100 network evaluations, unguided
./mnist_ablations.sh guidance   # guidance 1 / 1.5 / 3 over the full path or within [0, 0.5), [0.25, 0.75), [0.5, 1)
```

`guidance` runs on the CFG-trained variants of step 1.

## 4. Not implemented yet

- **Positional embedding ablation:** 1D learned vs 2D sin-cos (the current default) vs 2D RoPE.
  - `pos_emb_type='1d_learned'` exists in the model.
  - 2D RoPE needs implementing in the attention modules (`models/support_models.py`).
  - `image_gen_trainer.py` needs a `--pos_emb_type` flag, and `mnist_ablations.sh` the arms.
  - On pose this axis is a no-op (`seq_len=1`), which is why it was left for MNIST.
- **Dropout:** MNIST trains with dropout 0.1, pose with 0. With 60k training images it's a tuning
  knob, not a settled choice. Needs a `--dropout` flag and a `dropout_0` arm.
