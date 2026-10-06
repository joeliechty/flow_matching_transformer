---
license: mit
tags:
  - flow-matching
  - generative
  - se3
  - mnist
  - ablation-study
library_name: pytorch
---

# Flow Matching Transformer — Ablation Study

This repository hosts trained checkpoints for an ablation study of a flow matching
transformer applied to two tasks:

- **SE(3) pose generation** — generating 3D poses (position + orientation) by
  predicting twists on the SE(3) manifold.
- **MNIST image generation** — class-conditional digit generation over patch
  tokens with 2D sin/cos positional embeddings.

Each task has six ablation variants (4 conditional + 2 unconditional
baselines), each trained with 5 seeds, so the contribution of each
"state-of-the-art" component can be measured in isolation:

- **OT** — Geodesic Optimal Transport pairing between noise and data
- **CFG** — Classifier-Free Guidance (with a learnable null token)

Source code: {{GITHUB_URL}}

## Ablation results

Metrics computed by `experiments/evaluate_all.py` on 256 samples per model
(100 integration steps), reported as mean ± std over seeds. Pose metrics are computed in **SE(3) Lie-algebra (twist) space** — a
sample is assigned to the mode whose pose is closest to it under
`compute_twist_between_poses`. MNIST metrics use a small CNN classifier
(`classifier/mnist_cnn.pt`) as an oracle. CFG-trained conditional variants are
sampled at `cfg_scale=3.0`; `NOCFG` variants are sampled unguided
(`cfg_scale=1.0`). The `cfg` column is blank for unconditional models, where
guidance does not apply.

{{METRICS_TABLE}}

- `mode_accuracy` — fraction of conditional samples whose nearest SE(3) mode
  matches the conditioning action token. Higher is better.
- `mode_distance` — mean twist-space distance from each sample to its target
  mode (nearest mode for unconditional models). It keeps separating variants
  after mode accuracy saturates. Lower is better only down to the spread of
  the real goal distribution (≈0.22 for the default σ=0.1 goals); values below
  that mean samples are collapsing onto the mode centre.
- `mode_coverage_kl` — KL of the empirical mode histogram against uniform.
  Lower = better coverage of all 4 modes (relevant for unconditional models).
- `class_accuracy` — classifier accuracy on conditional MNIST samples against
  the conditioning label. Higher is better.
- `class_marginal_kl` — KL of the predicted-class histogram against uniform
  (relevant for unconditional MNIST).

## Figures

- `results/<task>/steps_sweep.png` — every metric vs the number of ODE
  integration steps used at sampling time (1–100), mean ± std over seeds.
- `results/<task>/cfg_sweep.png` — every metric vs `cfg_scale` for the
  CFG-trained conditional variants, mean ± std over seeds.
- `results/pose/seed_<N>/pose_grid.png` — 2×2 trajectory plot, one panel per
  conditional pose ablation variant.
- `results/mnist/seed_<N>/mnist_grid.png` — 4×10 image grid, rows = ablation
  variants, columns = digits 0-9.

## Loading a checkpoint

```python
from huggingface_hub import hf_hub_download
from models.conditional_flow_matching_transformer import ConditionalFlowMatchingTransformerModel
from omegaconf import OmegaConf

repo_id = "{{REPO_ID}}"
ckpt = hf_hub_download(repo_id, "pose/seed_1/cond_pose_flow_matching_model_OT_CFG_epoch_100.pt")
config_path = hf_hub_download(repo_id, "pose/seed_1/cond_pose_flow_matching_model_OT_CFG_training_config.yaml")

config = OmegaConf.load(config_path)
model_config = OmegaConf.to_container(config.model, resolve=True)
model, _ = ConditionalFlowMatchingTransformerModel.load_checkpoint(
    ckpt, device="cpu", model_config=model_config,
)
model.eval()
```

## Reproduction

```bash
# Train + evaluate each task's 6 variants for seeds 1-5 (metrics, CFG sweep,
# sampling-steps sweep, sample grids, mean ± std summaries).
./pose_ablations.sh
./mnist_ablations.sh

# Upload (after huggingface-cli login).
python huggingface/upload.py --repo_id {{REPO_ID}}
```

Hyperparameters: pose models trained for 100 epochs × 100 batches × 128 samples;
MNIST models trained for 400 epochs × 200 batches × 128 samples. All variants
share the same architecture, optimizer, and seeds (1–5) — only the OT / CFG /
conditional flags change. See `pose/seed_<N>/*.yaml` and `mnist/seed_<N>/*.yaml`
for full configs.
