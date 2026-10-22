
# On-Manifold Flow-Matching Transformers for Generative Modeling

This repository serves as a practical tutorial and reference implementation for Continuous-Time Generative Modeling on the different manifolds. It demonstrates how to train a Flow Matching model using a Transformer backbone to generate samples from different goal distributions (3D poses on the SE(3) manifold, images on the Euclidean manifold, etc.).

Standard diffusion or flow matching models operate in flat Euclidean space for tasks like image generation. For robotics, flow matching can be done in either a latent Eucldean space or on the SE(3) manifold. This repo explains how to do the latter. It covers how to implement the transformer architecture in a flow matching model as well as some tips and tricks that researchers have discovered make these models perform better including optimal transport, classifier free guidance, and using adaptive layer norms.

Which of these tricks actually help was tested on synthetic SE(3) tasks and on real robot grasps (ACRONYM). The recipe and the evidence are in [Findings: Pose Generation](#findings-pose-generation).

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

Three practical points, from this repo's experiments (the calibration study is in `ablations_summary.md` at tag `pose-3sigma-geo-v1`; the current recommendations are in [ablations_continuous.md](ablations_continuous.md)):
* **Calibrate the knob per task.** The published defaults (`--r_tar 0.01`, `--cond_scale 10`) work on 2-D toys, but on the pose tasks they barely change the pairing. There, a 5-unit translation shared by every pair dominates the cost. `experiments/pairing_diagnostics.py --calibrate` picks the loosest setting whose prior skew stays within a bound, without training.
* **The OT batch size sets a trade-off.** C²OT solves one assignment over an OT batch several times the network batch and splits it into network batches (`--ot_batch_mult`). That gives each sample more partners with nearby conditions and makes one-step samples better. Here it also made 20–100-step samples worse than random pairing. With the OT batch equal to the network batch, the pairing beat random pairing at 1–5 steps and matched it beyond.
* **Beyond minibatches.** Semidiscrete flow matching (SD-FM) [14] pairs fresh noise with the whole dataset through a learned dual potential, which removes the minibatch limit. Conditional variable flow matching [15] amortizes conditional OT across continuous conditions. Neither is implemented here.

How these pairings compare on continuous-condition pose tasks, synthetic and real, is in [ablations_continuous.md](ablations_continuous.md#pairing-noise-with-data).

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
* **Solvers.** Sampling integrates with Euler, midpoint or Heun steps (`method`). On SE(3), midpoint and Heun apply their twists through the exponential map, so they stay on the manifold. For pose, 5 midpoint steps matched 100 Euler steps, and Heun was the worst at low budgets ([Findings](#sampling-solver-and-number-of-steps)).

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

For clean discrete labels the three pathways performed about the same in the pose experiments. For a point cloud, adaLN on the pooled feature generalised worse to unseen objects than attention did ([Findings](#how-the-condition-enters-the-network)).

What a token is made from is set by the task (`obs_encoder`, `OBS_ENCODERS`):
* `mlp`: continuous token vectors, for the synthetic tasks and MNIST.
* `embedding`: integer ids, here `[category, object]` for ACRONYM grasps.
* `point_patch`: a point cloud, tokenised the Point-BERT / Point-MAE way [21] into 64 patches. Farthest-point sampling picks the centres, each centre's 32 nearest points go through a mini-PointNet, and 4 transformer layers mix the patches. A batch holds many grasps of few objects, so each cloud is encoded once and cross-attention computes its keys and values once (`SharedTokens`). adaLN pools the patches.

### 11. Classifier-Free Guidance (CFG) [3]
Classifier-free guidance lets a single conditional model trade off sample diversity for stronger adherence to its conditioning at inference time, without training a separate classifier.
* Training: With probability ~10% (see `_run_flow_matching_step` in [utils/train_utils.py](utils/train_utils.py)), conditioning tokens are replaced with a learned null embedding. The model therefore learns both the conditional vector field v(x, t | c) and the unconditional vector field v(x, t | ∅) simultaneously.
* Inference: At each ODE step, two forward passes are run — one with the real condition, one with the null condition — and the result is extrapolated as `v = v_uncond + cfg_scale * (v_cond - v_uncond)` (see `inference` in [models/conditional_flow_matching_transformer.py](models/conditional_flow_matching_transformer.py)). `cfg_scale = 1.0` recovers the standard conditional flow; higher values push the trajectory more aggressively toward the conditioned mode at the cost of diversity.
* Guidance intervals: `cfg_interval=(lo, hi)` applies guidance only at flow times `lo <= t < hi` and the plain conditional velocity elsewhere [18].
* In this repo's pose experiments, no guidance setting improved on unguided sampling; see [Guidance](#guidance) under Findings.

## Findings: Pose Generation

The pose model was put through a series of ablation studies, first on synthetic SE(3) goal distributions and then on real robot grasps. This section gives the recipe they support, what each choice did, the pitfalls found on the way, and what is still open. The full evidence is in two reports, one per kind of conditioning: [ablations_discrete.md](ablations_discrete.md) for a label that many samples share, and [ablations_continuous.md](ablations_continuous.md) for a signal that differs for every sample. MNIST hasn't been run under the current framework ([docs/mnist_todo.md](docs/mnist_todo.md)), so none of this is verified for images.

### What Was Tested
* **Synthetic goal distributions** ([configs/pose_tasks/](configs/pose_tasks/)), conditioned on tokens that name a goal corner:
  * `four_corners`: four far-apart modes. Every model solved it, so it was dropped after the first round.
  * `corners_two_orientations`: each corner holds two orientations 5σ apart, so each condition is bimodal. A model has to keep both orientations rather than average them.
  * `corners_3sigma`: the same eight modes packed so that every pair is within 3σ. The corners overlap: only 73.4% of real samples are nearest a mode of their own corner. This is the main synthetic task.
  * Continuous versions: `corners_3sigma_jitter` and `corners_two_orientations_jitter` add noise to every token, and `continuous_goals` puts each goal anywhere on a disk. No two samples share a condition.
  * Two 2-D toys (moons and fork, `toy_gen_trainer.py`) checked the continuous-condition pairings against their papers before the pose runs.
* **Real grasps from ACRONYM [20].** Successful Franka grasps on 10 object categories (606 training objects, 711k grasps), conditioned on the object's point cloud or on its `[category, object]` ids. Models are scored on held-out grasps of training objects, on 151 unseen objects, and on unseen categories (Cup, Camera, Guitar).
* **Metrics.** Every metric compares samples with real samples the model never trained on, and is reported next to the score of a real sample set of the same size (the data floor).
  * Synthetic: **energy distance** per goal mode (0 = matches the data). Corner accuracy and spread ratio show *how* the samples are wrong.
  * ACRONYM: **1-NN accuracy**, how well a nearest-neighbour test tells generated grasps from real ones (0.5 = it can't). The readable companions are recreation error (real grasp → nearest sample) and fidelity error (sample → nearest real grasp), in cm on the gripper's control points, with the jaw swap symmetrised.
* **Protocol.** 5 seeds per setting on the synthetic tasks, 3 on ACRONYM. Each alternative retrains the recommended setting with one choice changed, and differences are judged against the seed-to-seed standard deviation (SD).

Numbers below are energy distance for the synthetic tasks and 1-NN accuracy for ACRONYM, lower is better for both, sampled with Euler steps unless a solver is named.

### The Recommended Recipe
1. **Keep the framework of section 8 and its defaults.** No alternative beat uniform training times, sinusoids of t or, for grasp-like poses that can face any direction, a uniform-SO(3) rotation prior.
2. **Pair noise with data by minibatch OT within each condition** (`--pairing ot`, the default).
   * Put many samples of each condition in a batch. On ACRONYM, 16 objects × 64 grasps (`--grasps_per_object 64`) beat 64 objects × 16 grasps.
   * If no two samples share a condition, use C²OT with the OT batch equal to the network batch (`--pairing c2ot --ot_batch_mult 1`, with `--r_tar` from `./pose_ablations.sh calibrate`).
   * Never pair across conditions (`--pairing global`).
3. **Feed a rich condition through attention.** For a point cloud use `--cond_mode cross_attn`; for a few tokens `--cond_mode joint`. For clean discrete labels the pathway barely matters, and joint attention was slightly best.
4. **Train without CFG and sample unguided** (`--no_cfg`).
5. **Sample with 5 midpoint steps** (`method='midpoint'`, 10 network evaluations). With a budget of 2 evaluations, take 2 Euler steps. Avoid Heun.
6. **Size the backbone to the number of conditions.** The default 1.3M-parameter model learns the synthetic tasks fully. 606 object ids wanted the largest backbone tried, 80M parameters (`--hidden_dim 512 --num_layers 12 --num_heads 8`). Point clouds need 38M (`--hidden_dim 384 --num_layers 8 --num_heads 6`), and more doesn't help on unseen objects.
7. **Train longer only when there are many conditions.** The synthetic tasks reach the data floor within 10–20 of 100 epochs, and point-cloud models plateau halfway through their 30k steps. The id model was still improving at 30k steps.

On the 3σ task this recipe reaches energy distance 0.0004 at 5 midpoint steps and the metric's floor at 9:
```bash
python pose_gen_trainer.py --conditional --no_cfg --task_config configs/pose_tasks/corners_3sigma.yaml --num_epochs 100
```

![Every alternative against the recommended setting, discrete conditioning](docs/combined/discrete_scorecard.png)
![Every alternative against the recommended setting, continuous conditioning](docs/combined/continuous_scorecard.png)

*Each row swaps one recommended setting for an alternative, in both evaluations (circles: synthetic 3σ task; squares: ACRONYM). Each marker is how much worse (right) or better (left) that makes the samples, in seed SDs; the grey band is no clear difference. Left: 3 Euler steps; right: 100.*

### What Each Choice Did

#### Pairing Noise with Data
* **OT across a mixed-condition batch hurts conditional models.** The first round paired across all conditions at once, so each condition trained only on the noise nearest its own goals. Conditional models were then worse with OT than without (mean distance to the target mode 0.85 vs. 0.29 at 100 steps). Pairing within each condition fixed it. With continuous conditions, global OT fails the same way: 0.032 at 100 steps on the 3σ noisy-token task, and 0.719 vs. 0.621 on ACRONYM's unseen objects.
* **OT within each condition is what makes few-step sampling work, and it costs nothing at 100 steps.**
  * Synthetic 3σ task, 3 steps: 0.0057 vs. 0.087 for random pairing (15×).
  * ACRONYM ids: 0.812 vs. 0.983 at 1 step and 0.551 vs. 0.627 at 3. Point clouds, unseen objects at 3 steps: 0.653 vs. 0.710.
  * Random pairing catches up by about 20 steps and is level at 100. OT costs about 1.1× the training time and nothing at sampling.
* **Why it matters for multimodal conditions.** Without OT the network learns the average velocity towards a condition's modes, so few-step samples land between them. On the two-orientations task, 44% of 3-step samples ended between the two orientations without OT, against 16% with OT and 9% of real samples.
* **More samples of each condition per batch straighten the paths further** (ACRONYM). With 64 grasps of each of 16 objects, 1 step scores 0.653 (ids) and 0.722 (point clouds), against 0.812 and 0.836 with 16 grasps of each of 64 objects. With 4 grasps of each of 256 objects it is 0.95. All are level at 100 steps.
* **Continuous conditions need condition-aware pairing (section 6), calibrated per task.**
  * On the toys the implementation reproduces C²OT: one-step W₂² 0.071 on moons, against the paper's 0.077.
  * On poses the published knob defaults barely change the pairing, because the translation from the noise to the goals, shared by every pair, dominates the cost. `./pose_ablations.sh calibrate` picks the loosest setting whose prior skew stays within a bound, without training.
  * The OT batch sets a trade-off. An OT batch of 4 network batches gave better one-step samples, but was 1.5–2.3× worse than random pairing at 100 steps. With the OT batch equal to the network batch, the pairing was never worse than random.
  * On the 3σ noisy-token task, C²OT matches an oracle that pairs within the true corners: 0.0056 at 3 steps, against 0.085 for random pairing.
  * After calibrating, check how well the conditions separate, not only the skew bound. On overlapping modes every setting passed the bound, and the fixed-weight and cluster pairings at the loosest setting under-separated the corners (corner accuracy 0.61–0.62 vs. 0.73 for real data). C²OT, which sets its weight per batch, held up.

#### How the Condition Enters the Network
* **Clean discrete labels: the pathway barely matters.** On the synthetic task adaLN, cross-attention and joint attention are within seed noise at every step count. On ACRONYM ids joint attention is slightly best, 0.513 at 100 steps against 0.516 for cross-attention and 0.515 for adaLN, consistently across step counts.
* **Rich continuous conditions: use attention.**
  * adaLN on a pooled point-cloud feature memorises. It is better on seen objects, but worse on unseen objects (0.639 vs. 0.621) and unseen categories (0.716 vs. 0.696).
  * On the synthetic noisy tokens, attention separates the corners closer to the data (corner accuracy: joint 0.722, cross-attention 0.704, adaLN 0.682, real data 0.734), with energy distance tied.
  * Joint attention ties cross-attention on point clouds, but it puts all 64 patch tokens in every sample's sequence, so it trains 8× slower.

#### Guidance
* **Training with CFG doesn't hurt; guiding does.** Sampled unguided, CFG-trained models match those trained without CFG.
* **Guidance 3, the inference script's default, is the worst setting tested everywhere.**
  * Synthetic 3σ task: 0.072 at 100 steps, against the unguided floor. Samples are pushed outward past their corner, and corner accuracy rises to 1.00 against the data's 0.73: the samples are easier to classify than real data, which is a distortion.
  * Where each condition's modes are its own (two orientations), guidance also unbalances the split between them, as far as 25/75.
  * ACRONYM: 0.73 (ids) and 0.81 (point clouds), against 0.516 and 0.621 unguided.
* **Small scales don't help either.** At 1.1–1.3, where the original CFG paper found its best FID [3], nothing beats unguided sampling on ACRONYM. On the synthetic noisy-token task mild guidance gives a small, consistent gain (−0.0001 vs. 0.0002), because the unguided model blurs the noisy corners together (corner accuracy 0.685 vs. 0.734). That gain is smaller than the noise of one real draw.
* **Guidance sharpens the samples rather than correcting them.** On ACRONYM ids, scales 1.2–1.5 lower both nearest-neighbour errors (fidelity 1.97 → 1.76–1.85 cm, recreation 1.78 → as low as 1.73 cm) while 1-NN accuracy gets worse. The samples concentrate on the common grasps and under-sample the rare ones. This fits the literature: guidance trades diversity for fidelity, and its gains on distribution metrics come mostly from tightening a model that spreads its samples too widely. These models don't.
* **Where along the path it acts matters.** Guiding the whole path does the most damage. On the synthetic task the noisy half (t < 0.5) is by far the least harmful (0.0029 vs. 0.072 at scale 3), unlike images, where the middle of the path was best [18]. On ACRONYM the partial intervals are closer together.

#### Sampling: Solver and Number of Steps
* **Midpoint is the best solver from 4 network evaluations up.** 5 midpoint steps (10 evaluations) reach the 100-step quality: 0.0004 on the synthetic task, against 0.0015 for Euler at the same cost, and 0.621 on ACRONYM's unseen objects. On ACRONYM ids, 3 midpoint steps already match 100 Euler steps (0.517 vs. 0.516).
* **At 2 evaluations, two Euler steps beat one midpoint step** (ACRONYM ids 0.585 vs. 0.759; point clouds 0.677 vs. 0.801).
* **Heun is the worst at low budgets** and only catches up at about 18 evaluations. At 2 evaluations on the synthetic task it scores 0.20, 26× Euler, with 3.2× the data's rotation spread. A likely cause, not verified: Heun averages twists evaluated at orientations a full step apart without moving them into a common frame, whereas midpoint's second evaluation is only half a step away.
* **Midpoint narrows OT's advantage without removing it.** Random pairing gains the most from it, but OT is still 7–12× better at 4–6 evaluations on the synthetic task.

#### Training Times, Time Embedding and Prior
* **Training-time density.** Logit-normal (SD3's) helps the synthetic task at 1–3 Euler steps (0.0076 vs. 0.0116 at 1 step), but its one-step samples spread 2.5× too widely in rotation, against 1.6× with uniform times. It ties from 9 steps on, and ties everywhere on ACRONYM. π0's Beta ties with uniform.
* **Time embedding.** Sinusoids of 1000·t and Gaussian Fourier features are worse at 1–3 steps (one step: 0.018 and 0.029 vs. 0.012) and tie from 9 steps.
* **Rotation prior.** On ACRONYM a Gaussian-twist prior ties the uniform-SO(3) one (ids: 0.546 vs. 0.551 at 3 steps, level at 100).

#### Model Size and Training Length
* **Capacity helps on seen conditions, not on new ones.** ACRONYM ids at 10M / 30M / 80M parameters: 0.529 / 0.520 / 0.516, with diminishing returns. Point clouds from 14M to 93M: held-out grasps of training objects improve from 0.522 to 0.515, but unseen objects stay at 0.62 and unseen categories at 0.69–0.70.
* **The synthetic tasks are learned almost at once.** Energy distance is at the floor by epoch 10–20, although the spread keeps tightening towards the data's until epoch 100 (0.94 at epoch 40, 0.92 at 100; real data 0.91).
* **Few-step Euler sampling gets worse with longer training on the synthetic task** (3 steps: 0.0033 at epoch 10, 0.0057 at 100), as the samples contract towards the mode centres (position spread 0.99 → 0.67 of the data's). Midpoint sampling doesn't degrade, and neither does Euler on ACRONYM. If you'll sample with 1–3 Euler steps on a small task, use an early checkpoint.
* **ACRONYM:** unseen-object quality is flat from epoch 50 (0.622 at 50, 0.621 at 100), while the id model is still improving at the end (0.528 at 50, 0.516 at 100).

#### Generalising to New Objects (ACRONYM)
![Generalisation](docs/acronym/generalisation.png)

*Per-object OT baselines at epoch 100 on each held-out level; dotted lines are real data. Left: 1-NN accuracy; middle: recreation error; right: fidelity error.*

* **Conditioning on geometry costs nothing on known objects.** On held-out grasps of training objects, the point-cloud model matches the id model at every step count (0.516 at 100 steps for both).
* **New objects leave a gap.** Unseen objects score 0.62 and unseen categories 0.70, against 0.50 for real data. Model size, pairing, training times and the prior don't close it; the point-cloud encoder and the number of training objects are the untested levers.
* **The gap has a heavy tail.** The median unseen object's samples are close to its real grasps (energy distance 0.09 cm), but a few unusually large objects, such as tall bottles and toy figures of 0.25–0.34 m, fail outright (1-NN accuracy ≥ 0.97). That looks like extrapolation beyond the training objects' sizes.

#### The Current Framework vs. the Original Model
* **The original model** had plain adaLN with LayerNorm, a quaternion input, a fixed grid of 10 training times, dropout 0.1 and no EMA. Its best configurations stayed 10–40× above the energy-distance floor and spread their samples 1.04–1.15× (3σ task) to 1.3–1.5× (two orientations) as widely as the data.
* **The current framework (section 8)** reaches the floor: on the 3σ task with OT at 100 steps, 0.0082 → −0.0001. Few-step samples are 2.2–2.6× better, the spread is 0.93–0.95× (real data 0.91×), the seed range is about 10× narrower, and training converges by epoch 10–20 instead of wandering.
* **The settled practices were adopted together,** so the gain isn't attributed to any one of them. The original model's code and checkpoints load at tag `pose-continuous-v1`.

### Pitfalls Found Along the Way
* **Take the short way round in rotation.** q and −q are the same rotation, so the twist between two poses must use the quaternion difference with qw ≥ 0 (`compute_twist_between_poses` in [utils/tf_utils.py](utils/tf_utils.py)). Before this fix, about 10% of the 3σ task's training pairs rotated by more than π. Fixing it improved the synthetic baseline by 6–13 seed SDs at 2 or more steps and cut the one-step rotation over-spread from 2.5× to 1.6×. Assigning samples to modes needs the same care: the old evaluation mishandled quaternions with negative w.
* **Pair within conditions**, as above. Plain minibatch OT over a conditional batch silently skews each condition's prior.
* **Use metrics that collapse can't game.**
  * The mean distance to the target mode scored collapsed few-step samples as improvements, since real samples themselves sit 0.219 from their mode.
  * Compute energy distance per mode. Over the whole mixture, the gaps between modes dilute the within-mode errors by about 1/K.
  * On ACRONYM, recreation and fidelity error bottom out at 4–10 steps because they reward slight contraction. 1-NN accuracy falls monotonically with steps, which is why it is the primary metric there.
  * Averaged over objects, energy distance is dominated by the few objects a model fails on, so report the median as well.
  * Training losses aren't comparable across pairings: OT makes the regression target less noisy, so its loss is lower whether or not its model is better.
* **Prepare meshes without simplifying.** Manifold's `-s` flag and the `simplify` step, which ACRONYM used, corrupted 239 of the 852 ShapeNetSem meshes here. Plain `manifold` gives surfaces within 0.8–1.3 mm of ACRONYM's own.
* **Speed-ups that changed nothing measurable:** bf16 autocast, and encoding each point cloud once per object with shared keys and values.

### Still Open
* **Generalisation to new objects** is the real gap (above).
* **The best settings were tested one at a time, not combined**, e.g. 64 grasps per object with joint attention and midpoint sampling.
* **The ACRONYM id model isn't converged** at 30k steps, and might keep improving with a larger backbone.
* **At 100 steps the best synthetic models sit at the metric's floor**; 256 evaluation samples can't separate them.
* **Not tried:** a diagnosis of Heun's failure, C²OT and autoguidance on ACRONYM, and a grasp-success metric (these studies measure pose recreation only).

### Full Reports
* [ablations_discrete.md](ablations_discrete.md) and [ablations_continuous.md](ablations_continuous.md): the current recommendations, with one figure per topic (`python experiments/combined_figures.py --kind discrete` or `--kind continuous`).
* The per-study write-ups, with every arm's numbers and the commands to rerun them, are kept at git tags:
  * `pose-two-orientations-v1`: the four-corners task, the conditional-OT fix and the validation of the metrics.
  * `pose-continuous-v1`: the original model on the two-orientations and 3σ tasks, and the continuous-condition pairings (toys, calibration, OT batch size).
  * `pose-3sigma-geo-v1`: the current framework on the 3σ tasks, after the rotation fix (`ablations_summary.md`).
  * `acronym-continuous-v1` and `acronym-discrete-v1`: the ACRONYM study (`acronym_ablations_summary.md`).

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
* `--cfg_scale` / `-CFG`: Classifier-free guidance scale (default `3.0`; `1.0` disables CFG). Scale 3 was the worst setting in every pose experiment; unguided sampling was best ([Findings](#guidance)).
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

These experiments compare the pairings of section 6. The results are in [ablations_continuous.md](ablations_continuous.md#pairing-noise-with-data).

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

### 6. Real Grasp Poses (ACRONYM)

The real-data ablations generate Franka grasp poses on ACRONYM objects [20], conditioned on the object's point cloud (Part 1) or on its ids (Part 2). Results and recommendations, side by side with the synthetic 3σ task, are in [ablations_continuous.md](ablations_continuous.md) (point clouds) and [ablations_discrete.md](ablations_discrete.md) (ids).

**Data.** The grasps are free. The meshes, needed for point clouds, come from ShapeNetSem on Hugging Face (gated: request access, then `hf auth login`). `prepare_acronym.py` indexes the grasp files, picks and splits the objects of a dataset file (`configs/datasets/`), makes watertight meshes with Manifold [22], and writes one cache:
```bash
python -I prepare_acronym.py index
python -I prepare_acronym.py watertight --dataset configs/datasets/acronym_10cat.yaml \
    --obj_dir <ShapeNetSem>/models-OBJ/models --manifold_dir <Manifold>/build
python -I prepare_acronym.py build --dataset configs/datasets/acronym_10cat.yaml --mesh_root data/acronym
python experiments/acronym_metric_check.py --task_config configs/pose_tasks/acronym_10cat_ids.yaml
```

**Training and evaluation.** The task files `configs/pose_tasks/acronym_10cat_{points,ids}.yaml` (`type: acronym`, `utils/acronym.py`) set the conditioning, the prior and the batch layout (objects × grasps per object, with the object as the OT condition). `pose_ablations.sh` detects them and scores every model on each held-out level (held-out grasps, objects and categories) with the pose-recreation metrics of `utils/grasp_metrics.py`:
```bash
export TASK_CONFIG=configs/pose_tasks/acronym_10cat_points.yaml BATCHES_PER_EPOCH=300
./pose_ablations.sh size                                                  # S/M/L backbones
MODEL_FLAGS="--hidden_dim 384 --num_layers 8 --num_heads 6" ./pose_ablations.sh all    # pairing x CFG
python experiments/acronym_figures.py --part points                       # figures and the decision rules
```

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
  * `acronym.py`: ACRONYM grasp poses as a task (`AcronymGraspTask`): batches, priors, and the held-out evaluation sets.
  * `grasp_metrics.py`: Pose-recreation metrics on a jaw-swap-invariant control-point distance (1-NN accuracy, recreation and fidelity error, coverage, energy distance).
* `pose_gen_trainer.py`: SE(3) pose-generation training loop with the built-in multimodal goal distribution, minibatch sequence formatting, loss computation, and checkpointing.
* `pose_gen_inference.py`: ODE sampling (Euler, midpoint or Heun) for SE(3) pose action chunks from the trained vector field, plus 3D trajectory visualization.
* `image_gen_trainer.py`: MNIST training entrypoint. Tokenizes images into 7×7 patch grids and reuses the shared training loop for Euclidean flow matching.
* `image_gen_inference.py`: MNIST sampling entrypoint. Integrates patch-space noise back to images and tiles the noise → denoised trajectory.
* `toy_gen_trainer.py`: Trains a conditional flow on a 2-D toy with one pairing.
* `prepare_acronym.py`: One-time ACRONYM data preparation (index, object selection and splits, watertight meshes, point clouds).
* `pose_ablations.sh`, `toy_ablations.sh`, `mnist_ablations.sh`: Ablation drivers (train every variant and seed, then evaluate).
* `experiments/`: Evaluation and figures. `evaluate_all.py` (pose and MNIST metrics), `steps_sweep.py`, `cfg_sweep.py`, `evaluate_toys.py`, `pairing_diagnostics.py` (pairing statistics and knob calibration, no training), `aggregate_seeds.py`, `combined_figures.py` (the figures in ablations_continuous.md and ablations_discrete.md), the per-study figure scripts `summary_figures.py` and `acronym_figures.py` (which also print the size and baseline rules), and for ACRONYM `acronym_metric_check.py` and `acronym_grasp_figure.py`.
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

[20] Eppner, C., Mousavian, A., & Fox, D. (2021). ACRONYM: A Large-Scale Grasp Dataset Based on Simulation. IEEE International Conference on Robotics and Automation (ICRA 2021). https://arxiv.org/abs/2011.09584

[21] Yu, X., Tang, L., Rao, Y., Huang, T., Zhou, J., & Lu, J. (2022). Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling. CVPR 2022. https://arxiv.org/abs/2111.14819 — and Pang, Y., Wang, W., Tay, F. E. H., Liu, W., Tian, Y., & Yuan, L. (2022). Masked Autoencoders for Point Cloud Self-supervised Learning. ECCV 2022. https://arxiv.org/abs/2203.06604

[22] Huang, J., Su, H., & Guibas, L. (2018). Robust Watertight Manifold Surface Generation Method for ShapeNet Models. arXiv. https://arxiv.org/abs/1802.01698
