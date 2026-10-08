
# On-Manifold Flow-Matching Transformers for Generative Modeling

This repository serves as a practical tutorial and reference implementation for Continuous-Time Generative Modeling on the different manifolds. It demonstrates how to train a Flow Matching model using a Transformer backbone to generate samples from different goal distributions (3D poses on the SE(3) manifold, images on the Euclidean manifold, etc.).

Standard diffusion or flow matching models operate in flat Euclidean space for tasks like image generation. For robotics, flow matching can be done in either a latent Eucldean space or on the SE(3) manifold. This repo explains how to do the latter. It covers how to implement the transformer architecture in a flow matching model as well as some tips and tricks that researchers have discovered make these models perform better including optimal transport, classifier free guidance, and using adaptive layer norms.

## Core Concepts and Tutorial Overview

This repository is designed to teach several advanced concepts in generative modeling and manifold mathematics.

### 1. Flow Matching on SE(3) [1,4]
Standard flow matching learns a vector field that transports a simple base distribution (e.g., a standard Gaussian) to a complex data distribution. In this repository, our "data" consists of 3D poses.
* State Representation: The integrator state is a pose (position + quaternion). The network sees it as a position and an Ortho6D rotation (section 2).
* Network Output: The network predicts Twists (v in R^6), representing linear and angular velocities.
* Integration: ODE integration uses the twist exponential map (`add_twist_to_pose` in [utils/tf_utils.py](utils/tf_utils.py)) to ensure the generated samples stay strictly on the SE(3) manifold.

### 2. SE(3) Lie Algebra: Ortho6D and the Twist Exponential Map [8]
The SE(3) flow keeps three representations apart: what the network reads, what it predicts, and the state the integrator carries.
* Network input uses **Ortho6D** (two rows of the rotation matrix) alongside the position. Ortho6D is continuous and the same for q and −q, whereas a quaternion input shows the network every rotation twice (q and −q), with a jump between them. The model converts each pose on the fly (`_quat_to_ortho6d` in [utils/tf_utils.py](utils/tf_utils.py)), so `forward` still takes 7-D poses.
* Network output is a **twist** (a tangent vector), the standard target on Lie groups [1].
* Manifold state stays as **quaternions + position** (7D). Integration uses the axis-angle exponential map: the angular velocity ω is converted to a quaternion via `[cos(½‖ω‖dt), (ω/‖ω‖) sin(½‖ω‖dt)]` and composed with the current orientation, while linear velocity integrates additively (`add_twist_to_pose` in [utils/tf_utils.py](utils/tf_utils.py)).

### 3. Tokenized Flow Matching (Action Chunks) [7]
To support continuous trajectories or multi-joint systems, this repository natively supports Tokenized Flow Matching (inspired by architectures like pi0). Instead of flattening temporal or spatial dimensions, the model processes inputs as sequences `[batch_size, seq_len, dim]`. 
* Joint Denoising: The joint distribution and kinematic constraints of the action chunk are learned implicitly via the Transformer's self-attention mechanism.
* Independent Integration: ODE integration is applied to each token/slot independently using standard SE(3) algebra.

### 4. Geodesic Optimal Transport (OT) [6]
To make learning efficient, flow matching pairs noise samples with target data samples. Rather than pairing them randomly, this implementation uses Geodesic Optimal Transport (`geodesic_optimal_transport_pairing` in [utils/train_utils.py](utils/train_utils.py)). It computes the exact pairwise geodesic distances (the magnitude of the twist required to move between poses) and solves the linear sum assignment problem (Hungarian algorithm) to find the shortest paths on the manifold. For sequence data, OT is applied independently per slot across the batch.

### 5. OT Pairing Modes: Per-Frame vs Flat [6]
Geodesic OT can be applied at different granularities depending on whether the sequence dimension carries spatial structure.
* **Per-frame** (`sequence_ot_pairing` in [utils/train_utils.py](utils/train_utils.py)) computes an independent Hungarian assignment for each slot in the action chunk. Used by the SE(3) pose trainer, where each slot is a separate pose with no spatial neighbor relationship.
* **Flat** (`flat_ot_pairing` in [utils/train_utils.py](utils/train_utils.py)) flattens `[B, S*D]` and computes a single permutation across the whole image. Used by the MNIST trainer, because independent per-patch OT would scramble spatial coherence and break the image structure of the noise samples.

### 6. OT Pairing with Continuous Conditions [10–15]
Minibatch OT and conditioning interact. A conditional model is sampled from the full noise distribution under every condition, so during training each condition has to see all of the noise distribution too.
* **Plain (global) OT skews the prior.** Pairing across a mixed-condition batch hands each condition only the noise samples nearest its own goals. The model never learns what to do with the rest of the noise, but sampling draws from all of it [10]. In this repo's pose ablations, this made conditional models worse than no OT at all.
* **Per-condition OT fixes this for discrete conditions.** `_pair_within_conditions` in [utils/train_utils.py](utils/train_utils.py) runs the Hungarian assignment separately within each condition. It is the pose trainer's default (`--pairing ot`).
* **Per-condition OT breaks down for continuous conditions.** When every condition is slightly different (a continuous goal, a noisy observation), each group holds a single sample, so per-condition pairing is just random pairing (I-CFM) and the few-step benefit of OT is lost.

The way out is to let noise move only between *nearby* conditions. If the conditional distribution p(x | c) changes smoothly with c, samples with nearby conditions are nearly samples of the same conditional, and pairing among them keeps each condition's noise close to the full prior. Three ways of doing this are implemented (`condition_aware_pairing` and `Pairer` in [utils/train_utils.py](utils/train_utils.py), selected with `--pairing`). In each, noise sample i starts out carrying the condition c_i of the data sample it was drawn with:
* **Soft condition penalty with an adaptive weight (`c2ot`) [10].** One Hungarian assignment over the whole batch, with cost `C[i,j] = cost(x0_i, x1_j) + w · ‖c_i − c_j‖²`. At w = 0 this is plain OT; as w → ∞ every noise sample stays with its own condition (per-condition OT for discrete conditions, random pairing for continuous ones). C²OT sets w per batch by bisection, so that a target share `r_tar` (default 0.01) of all pairs is admissible: they cost no more than the noise sample's original random partner. That removes any dependence on the scale of the condition metric.
* **Fixed large weight (`c2ot_fixed`) [12, 13].** The same cost with one large w, fixed at the start of training. This is the minibatch form of dynamic conditional OT (COT-FM) and Bayesian OT flow matching; their theory shows that the weighted problem recovers the conditional OT plan as w → ∞.
* **Cluster, then match (`cluster`) [11].** COT Policy quantizes the conditions with K-means (K about the batch size, with PCA first for image observations) and adds `γ · ‖c̄_i − c̄_j‖²` on the cluster centroids. γ is set per batch so the condition term outweighs the sample term about 10× (`--cond_scale`, default 10). The network still sees the raw condition.

Three practical points, from this repo's experiments ([ablations_summary.md](ablations_summary.md#part-2-continuous-conditioning)):
* **Calibrate the knob per task.** The published defaults (`--r_tar 0.01`, `--cond_scale 10`) work on 2-D toys, but on the pose tasks they barely change the pairing. There, a 5-unit translation shared by every pair dominates the cost. `experiments/pairing_diagnostics.py --calibrate` picks the loosest setting whose prior skew stays within a bound, without training.
* **The OT batch size sets a trade-off.** C²OT solves one assignment over an OT batch several times the network batch and splits it into network batches (`--ot_batch_mult`). That gives each sample more partners with nearby conditions and makes one-step samples better. Here it also made 20–100-step samples worse than random pairing. With the OT batch equal to the network batch, the pairing beat random pairing at 1–5 steps and matched it beyond.
* **Beyond minibatches.** Semidiscrete flow matching (SD-FM) [14] pairs fresh noise with the whole dataset through a learned dual potential, which removes the minibatch limit. Conditional variable flow matching [15] amortizes conditional OT across continuous conditions. Neither is implemented here.

The comparison of these pairings on continuous-condition pose tasks is in [ablations_summary.md](ablations_summary.md#part-2-continuous-conditioning).

### 7. Time Sampling [4, 16, 17]
Training draws the flow time `t ∈ [0, 1]` (t = 0 is noise, t = 1 data) continuously, so the network sees every time it is queried at during sampling (`_run_flow_matching_step` in [utils/train_utils.py](utils/train_utils.py)).
* **SE(3)** draws `--n_steps` times per (noise, goal) pair and fans the batch out to `[B*n_steps, S, D]`. Each geodesic is reused for several times, and the batch, OT batch and compute match the fixed 10-point grid this repository used to train on.
* **Euclidean** (images) draws one time per sample.
* **The density** is a contested choice, so it is a flag (`--t_dist`, `sample_times`): `uniform` (Rectified Flow / I-CFM, the default), `logit_normal` (weights the middle of the path; SD3's choice [16]) or `beta` (π0's, weights the noisy end [17]).

### 8. Transformer Backbone: adaLN-Zero and the Training Recipe [5, 16]
The core architecture ([models/flow_transformer_base.py](models/flow_transformer_base.py), shared by both models) is a DiT-style sequence-to-sequence Transformer. These choices are settled in current flow-matching and diffusion models, so they have no flags:
* **adaLN-Zero time conditioning.** The flow time is embedded and regresses, for every branch of every block, a shift, a scale and a gate: `x + gate · branch(modulate(norm(x)))`. The regressing layers start at zero, so every block starts as the identity and the output layer at zero [5].
* **RMSNorm and QK-norm.** Blocks normalise with RMSNorm, and attention RMS-normalises each head's queries and keys, which bounds the attention logits [16]. Attention runs through `F.scaled_dot_product_attention`.
* **Normalisation.** On SE(3) the network sees standardised positions and predicts a standardised twist; the trainer measures the statistics from the task (`pose_normalizer_stats`), and the loss is computed in standardised units. Without it, the translation to the goals (about 5 units) dominates the loss over the rotation.
* **EMA weights and gradient clipping.** Training keeps an exponential moving average of the weights (decay 0.999 for pose, 0.9999 for MNIST), checkpoints save it, and the loaders sample from it. Gradients are clipped at norm 1.
* **Solvers.** Sampling integrates with Euler, midpoint or Heun steps (`method`). On SE(3), midpoint and Heun apply their twists through the exponential map, so they stay on the manifold.

The flow time's featurisation is a contested choice, so it is a flag (`--time_emb`): sinusoids of t (the default), of 1000·t (DiT's training scale, used by SD3 and Flux), or Gaussian Fourier features [19].

Models from before this framework (tag `pose-continuous-v1` and earlier) can't be loaded by the current code; check out that tag to evaluate or plot them.

### 9. 2D Sin-Cos Positional Embeddings and Learned Null Tokens [2]
Two small architectural details are worth calling out because they materially affect conditional image generation.
* **2D sin-cos positional embeddings** (`get_2d_sincos_pos_embed` in [models/support_models.py](models/support_models.py)) give each of the 49 MNIST patches a position encoding that splits row and column into separate sinusoidal halves. This DiT-style spatial inductive bias outperforms a single learned 1D embedding when the token grid has a known 2D layout.
* **Learned null token** ([models/conditional_flow_matching_transformer.py:79](models/conditional_flow_matching_transformer.py#L79)) is a trainable `[1, 1, hidden_dim]` parameter that replaces observation embeddings on the unconditional path (training dropout and CFG inference). Unlike zero-masking, the network learns an explicit representation of "no condition," which is what makes the double-pass CFG extrapolation in section 11 numerically well-behaved.

### 10. Conditioning Pathways [5, 9, 16]
The `ConditionalFlowMatchingTransformerModel` extends the architecture to support goal-directed generation. Discrete actions or observations (e.g., "top", "left") are embedded as condition tokens, each with a learned embedding for its slot, so the model also knows which slots are nulled. How the tokens reach the transformer is a contested choice, so it is a flag (`--cond_mode`, `COND_MODES`):
* `adaln` (default): the tokens are flattened, projected and added to the time embedding, so they drive every block's adaLN modulation. This is how DiT conditions on class labels [5].
* `cross_attn`: every block gets a cross-attention branch to the tokens [9]. With a single token this collapses to adding one vector per sample, because attention over one key always has weight 1.
* `joint`: the tokens join the sequence, so self-attention mixes them in (DiT's in-context conditioning; SD3 [16] and π0 [17] attend jointly too, with separate weights per stream).

### 11. Classifier-Free Guidance (CFG) [3]
Classifier-free guidance lets a single conditional model trade off sample diversity for stronger adherence to its conditioning at inference time, without training a separate classifier.
* Training: With probability ~10% (see `_run_flow_matching_step` in [utils/train_utils.py](utils/train_utils.py)), conditioning tokens are replaced with a learned null embedding. The model therefore learns both the conditional vector field v(x, t | c) and the unconditional vector field v(x, t | ∅) simultaneously.
* Inference: At each ODE step, two forward passes are run — one with the real condition, one with the null condition — and the result is extrapolated as `v = v_uncond + cfg_scale * (v_cond - v_uncond)` (see `inference` in [models/conditional_flow_matching_transformer.py](models/conditional_flow_matching_transformer.py)). `cfg_scale = 1.0` recovers the standard conditional flow; higher values push the trajectory more aggressively toward the conditioned mode at the cost of diversity.
* Guidance intervals: `cfg_interval=(lo, hi)` applies guidance only at flow times `lo <= t < hi` and the plain conditional velocity elsewhere [18].

## Installation

The repository uses Conda to manage dependencies and isolate the environment.

For Linux/Windows (CUDA):
```bash
git clone <your-repo-url>
cd flow_matching_transformer
bash setup_conda.sh
```

For macOS (Apple Silicon/M-Series):
```bash
git clone <your-repo-url>
cd flow_matching_transformer
bash setup_conda_mac.sh
```

This will create and activate a conda environment named `FmT` with Python 3.12, PyTorch, and all required math and visualization libraries.

## Usage

The repository ships two reference applications that exercise the same flow matching core:

* **Pose generation** (`pose_gen_trainer.py` / `pose_gen_inference.py`) — a built-in toy dataset of start poses at the origin and multimodal goal poses (rotated corners) on the SE(3) manifold.
* **Image generation** (`image_gen_trainer.py` / `image_gen_inference.py`) — MNIST tokenized as a 7×7 grid of 4×4 patches (`seq_len=49`, `patch_dim=16`), demonstrating Euclidean flow matching with the same tokenized transformer backbone.

### 1. Training a Pose Model

`pose_gen_trainer.py` trains on the built-in multimodal goal distribution.

The model and training recipe are fixed (section 8); only the contested choices have flags.

Train an Unconditional Pose Model (with Optimal Transport and CFG):
```bash
python pose_gen_trainer.py --num_epochs 50 --batch_size 128
```

Train a Conditional Pose Model (Action-Directed):
This trains the model to associate specific target modes with specific conditioning tokens (e.g., `top`/`bottom`/`left`/`right`).
```bash
python pose_gen_trainer.py --conditional --num_epochs 50
```

Useful Flags:
* `--conditional` / `-C`: Trains the conditional model variant with action-token conditioning; omit for the unconditional model.
* `--no_ot` / `-NOOT`: Disables Optimal Transport pairing (useful for seeing how OT improves flow straightness).
* `--no_cfg` / `-NOCFG`: Disables classifier-free guidance (no unconditional dropout during training).
* `--pairing`: How noise is paired with data (section 6). `ot` (default) runs OT within each condition, or over the whole batch for unconditional models; `independent` is the same as `--no_ot`; `global`, `c2ot`, `c2ot_fixed` and `cluster` are the pairings for continuous conditions. The checkpoint suffix follows the pairing (`_OT`, `_NOOT`, `_GOT`, `_C2OT`, `_C2OTFIX`, `_CLUSTER`).
* `--r_tar`: For `c2ot`, the target share of admissible pairs that sets the condition weight (default 0.01).
* `--ot_batch_mult`: Pairs over this many network batches at once, then splits them (default 1).
* `--num_clusters`: For `cluster`, the number of K-means clusters (default: the OT batch size).
* `--task_config`: Pose task file in [configs/pose_tasks/](configs/pose_tasks/) (goal modes and conditioning).
* `--n_steps`: Training times drawn per (noise, goal) pair (section 7).
* `--t_dist`: Training-time density: `uniform` (default), `logit_normal` or `beta` (section 7).
* `--time_emb`: Flow-time featurisation: `sinusoidal` (default), `sinusoidal_x1000` or `fourier` (section 8).
* `--cond_mode`: How the condition tokens enter: `adaln` (default), `cross_attn` or `joint` (section 10).
* `--seq_len` / `-S`: Sequence length per trajectory (action chunk size).
* `--save_path`: Directory to save `.pt` checkpoints and `.yaml` config files.

Checkpoints and configs are saved under `--save_path` with a suffix encoding the training mode, e.g.
`checkpoints/cond_pose_flow_matching_model_OT_CFG_epoch_10.pt` and the matching `_training_config.yaml`.

### 2. Running Pose Inference and Visualization

`pose_gen_inference.py` loads a trained checkpoint, integrates the learned ODE vector field, and visualizes the resulting SE(3) trajectories using Matplotlib.

Unconditional Inference:
```bash
python pose_gen_inference.py --checkpoint_epoch 50 --num_samples 10 --return_trajectory
```

Conditional Inference (Guiding the Flow):
If you trained a conditional model, you can force the flow toward specific modes by combining action tokens.
```bash
# Force the flow to the top-right mode
python pose_gen_inference.py --conditional --actions top right --num_samples 5 --return_trajectory
```

Additional Inference Flags:
* `--conditional` / `-C`: Loads the conditional model variant; must match the trained checkpoint.
* `--actions` / `-A`: One or more action tokens (`top`, `bottom`, `left`, `right`) to condition on. Omit on a conditional model to use the null token (unconditional path).
* `--cfg_scale` / `-CFG`: Classifier-free guidance scale (default `3.0`; `1.0` disables CFG).
* `--no_cfg` / `-NOCFG`: Forces `cfg_scale=1.0` (matches a model trained with `--no_cfg`).
* `--no_ot` / `-NOOT`: Loads a checkpoint that was trained without OT pairing.
* `--checkpoint_path` / `-CP`: Directory containing the checkpoint and config (defaults to `checkpoints/`).

The inference script automatically generates a 3D plot showing positional paths and coordinate frame axes (RGB = XYZ) evolving over time.

### 3. Training an Image (MNIST) Model

`image_gen_trainer.py` trains a flow matching transformer on MNIST patches. The 28×28 images are split into 49 patches of dimension 16; flow matching runs in Euclidean space over those patch tokens.

Train an Unconditional MNIST Model:
```bash
python image_gen_trainer.py --num_epochs 10 --batch_size 128
```

Train a Class-Conditional MNIST Model (digit labels 0–9):
```bash
python image_gen_trainer.py --conditional --num_epochs 10
```

Useful Flags:
* `--conditional` / `-C`: Trains a class-conditional model on digit labels 0–9; omit for the unconditional model.
* `--no_ot` / `-NOOT`, `--no_cfg` / `-NOCFG`, `--save_path`: Same semantics as the pose trainer.
* `--t_dist`, `--time_emb`, `--cond_mode`: The contested choices, as in the pose trainer.
* `--num_batches_per_epoch` / `-BE`: Minibatches drawn per epoch.
* `--data_root`: MNIST download/cache location (default `./data`).
* `--num_workers`: DataLoader worker count.

### 4. Running Image Inference and Visualization

`image_gen_inference.py` integrates the learned vector field from Gaussian noise back to image patches, then decodes patches into 28×28 images.

Unconditional Sampling:
```bash
python image_gen_inference.py --checkpoint_epoch 10 --num_samples 8 --return_trajectory
```

Class-Conditional Sampling (pick a digit 0–9):
```bash
python image_gen_inference.py --conditional --digit 7 --num_samples 8 --return_trajectory
```

Useful Inference Flags:
* `--conditional` / `-C`: Loads the class-conditional model variant; must match the trained checkpoint.
* `--digit` / `-D`: Digit class (0–9) for conditional sampling. Omit on a conditional model to fall back to the null token (unconditional path).
* `--cfg_scale` / `-CFG`, `--no_cfg` / `-NOCFG`, `--no_ot` / `-NOOT`: Behave the same as in pose inference.
* `--num_steps` / `-STEPS`: ODE integration steps (default 100).
* `--save_path`: Optional path to save the trajectory tile figure (noise → denoised image grid).

With `--return_trajectory`, the script tiles intermediate timesteps so you can see the noise denoise into MNIST digits.

### 5. Pairing Experiments with Continuous Conditions

These experiments compare the pairings of section 6. The results are in [ablations_summary.md](ablations_summary.md#part-2-continuous-conditioning).

**Toys first.** `toy_gen_trainer.py` trains a conditional flow on one of two 2-D toys (`utils/toy_tasks.py`). `moons` is C²OT's 8 Gaussians → moons, conditioned on the target's x-coordinate. `fork` is COT Policy's fork: y given x, with two branches for x > 0. Hyperparameters follow C²OT's toy setup.
```bash
python toy_gen_trainer.py --toy moons --pairing c2ot --seed 1 --save_path checkpoints/toy_moons/seed_1/
./toy_ablations.sh                    # every pairing x toy x seed, then W₂² at 1-100 Euler steps and RK45
./toy_ablations.sh diagnostics        # pairing statistics only, no training
```

**Pose tasks with continuous conditions.** `configs/pose_tasks/continuous_goals.yaml` puts each goal anywhere on a disk, so no two samples share a condition. `configs/pose_tasks/corners_two_orientations_jitter.yaml` adds noise to the corner tokens. `pose_ablations.sh` detects the kind of task and trains one conditional model per pairing:
```bash
export TASK_CONFIG=configs/pose_tasks/continuous_goals.yaml
./pose_ablations.sh diagnostics   # cost ratio, condition shift, prior skew, branch agreement (no training)
./pose_ablations.sh calibrate     # pick each pairing's knob: the loosest setting with prior skew <= 0.02
./pose_ablations.sh               # train every pairing x 5 seeds with the calibrated knobs, then evaluate
SENS_PAIRING=c2ot ./pose_ablations.sh sensitivity   # retrain one pairing with its knob and OT batch varied
```

**Single-axis ablations.** `ablate` retrains one baseline variant with one contested choice changed per arm (t density, time embedding, conditioning pathway, or CFG training), and `solver` and `guidance` compare ODE solvers at equal network evaluations and guidance intervals without retraining:
```bash
export TASK_CONFIG=configs/pose_tasks/corners_3sigma.yaml
ABLATE_BASE="cond_pose_flow_matching_model_OT_NOCFG|--conditional --no_cfg" ./pose_ablations.sh ablate
./pose_ablations.sh solver
./pose_ablations.sh guidance      # the CFG-trained models in CKPT_ROOT
```
Run `calibrate` before training. The published defaults (`--r_tar 0.01`, `--cond_scale 10`) barely change the pairing when one large offset (here the 5-unit translation to the goals) dominates every pairing cost. `experiments/pairing_diagnostics.py` measures this without training.

## Code Structure

* `models/`: Neural network architectures.
  * `support_models.py`: Core components (QK-normed self- and cross-attention, the adaLN-Zero block and final layer, time embeddings, FeedForward).
  * `flow_transformer_base.py`: What both models share: input embedding and normalisation, the transformer trunk, checkpoints, and the ODE solvers.
  * `flow_matching_transformer.py`: The unconditional base model natively supporting temporal sequences.
  * `conditional_flow_matching_transformer.py`: The action-conditioned sequence model.
* `utils/`: Mathematical and operational utilities.
  * `tf_utils.py`: High-performance, batched SE(3) operations. Handles conversions between quaternions, rotation matrices, Ortho6D, and computing or applying twists via exponential maps.
  * `train_utils.py`: Data generation, geodesic interpolation, and slot-wise Optimal Transport logic.
  * `visualization_utils.py`: 3D plotting utilities for visualizing SE(3) pose trajectories over time.
  * `logging_utils.py`: Utilities for routing output streams, formatting console logs, and tracking metrics.
  * `pose_task.py`: Pose task files (`configs/pose_tasks/`): discrete goal modes named by tokens, or goals that vary continuously with the condition (`ContinuousGoalTask`).
  * `toy_tasks.py`: The 2-D toys (moons, fork) used to check the continuous-condition pairings.
* `pose_gen_trainer.py`: SE(3) pose-generation training loop with the built-in multimodal goal distribution, minibatch sequence formatting, loss computation, and checkpointing.
* `pose_gen_inference.py`: ODE sampling (Euler, midpoint or Heun) for SE(3) pose action chunks from the trained vector field, plus 3D trajectory visualization.
* `image_gen_trainer.py`: MNIST training entrypoint. Tokenizes images into 7×7 patch grids and reuses the shared training loop for Euclidean flow matching.
* `image_gen_inference.py`: MNIST sampling entrypoint. Integrates patch-space noise back to images and tiles the noise → denoised trajectory.
* `toy_gen_trainer.py`: Trains a conditional flow on a 2-D toy with one pairing.
* `pose_ablations.sh`, `toy_ablations.sh`, `mnist_ablations.sh`: Ablation drivers (train every variant and seed, then evaluate).
* `experiments/`: Evaluation and figures. `evaluate_all.py` (pose and MNIST metrics), `steps_sweep.py`, `cfg_sweep.py`, `evaluate_toys.py`, `pairing_diagnostics.py` (pairing statistics and knob calibration, no training), `aggregate_seeds.py`, and `summary_figures.py` (the figures in ablations_summary.md).
* `pyproject.toml`: Project metadata and build configuration, allowing the repository to be installed as a standard Python package.


## References
[1] Chen, R. T. Q., & Lipman, Y. (2023). Flow Matching on General Geometries. ICLR 2024. https://doi.org/10.48550/arxiv.2302.03660

[2] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv. https://doi.org/10.48550/arxiv.2010.11929

[3] Ho, J., & Salimans, T. (2022). Classifier-Free Diffusion Guidance. arXiv. https://doi.org/10.48550/arxiv.2207.12598

[4] Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2022). Flow Matching for Generative Modeling. arXiv. https://doi.org/10.48550/arxiv.2210.02747

[5] Peebles, W., & Xie, S. (2023). Scalable Diffusion Models with Transformers. 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 4172-4182. https://doi.org/10.1109/iccv51070.2023.00387

[6] Tong, A., Fatras, K., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Wolf, G., & Bengio, Y. (2023). Improving and generalizing flow-based generative models with minibatch optimal transport. arXiv. https://doi.org/10.48550/arxiv.2302.00482

[7] Zhao, T., Kumar, V., Levine, S., & Finn, C. (2023). Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. Robotics: Science and Systems XIX. https://doi.org/10.15607/rss.2023.xix.016

[8] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2018). On the Continuity of Rotation Representations in Neural Networks. arXiv. https://doi.org/10.48550/arxiv.1812.07035

[9] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems. https://arxiv.org/abs/1706.03762

[10] Cheng, H. K., & Schwing, A. (2025). The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow-Based Generation. IEEE/CVF International Conference on Computer Vision (ICCV 2025). https://arxiv.org/abs/2503.10636

[11] Sochopoulos, A., Malkin, N., Tsagkas, N., Moura, J., Gienger, M., & Vijayakumar, S. (2025). Fast Flow-based Visuomotor Policies via Conditional Optimal Transport Couplings. Proceedings of the 9th Conference on Robot Learning (CoRL), PMLR 305, 3357-3377. https://arxiv.org/abs/2505.01179

[12] Kerrigan, G., Migliorini, G., & Smyth, P. (2024). Dynamic Conditional Optimal Transport through Simulation-Free Flows. Advances in Neural Information Processing Systems 37 (NeurIPS 2024). https://arxiv.org/abs/2404.04240

[13] Chemseddine, J., Hagemann, P., Steidl, G., & Wald, C. (2025). Conditional Wasserstein Distances with Applications in Bayesian OT Flow Matching. Journal of Machine Learning Research, 26(141). https://arxiv.org/abs/2403.18705

[14] Mousavi-Hosseini, A., Zhang, S. Y., Klein, M., & Cuturi, M. (2026). Flow Matching with Semidiscrete Couplings. International Conference on Learning Representations (ICLR 2026). https://arxiv.org/abs/2509.25519

[15] Generale, A. P., Robertson, A. E., & Kalidindi, S. R. (2024). Conditional Variable Flow Matching: Transforming Conditional Densities with Amortized Conditional Optimal Transport. arXiv. https://arxiv.org/abs/2411.08314

[16] Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., Podell, D., Dockhorn, T., English, Z., Lacey, K., Goodwin, A., Marek, Y., & Rombach, R. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. International Conference on Machine Learning (ICML 2024). https://arxiv.org/abs/2403.03206

[17] Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., et al. (2024). π0: A Vision-Language-Action Flow Model for General Robot Control. arXiv. https://arxiv.org/abs/2410.24164

[18] Kynkäänniemi, T., Aittala, M., Karras, T., Laine, S., Aila, T., & Lehtinen, J. (2024). Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models. Advances in Neural Information Processing Systems 37 (NeurIPS 2024). https://arxiv.org/abs/2404.07724

[19] Tancik, M., Srinivasan, P. P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J. T., & Ng, R. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. Advances in Neural Information Processing Systems 33 (NeurIPS 2020). https://arxiv.org/abs/2006.10739
