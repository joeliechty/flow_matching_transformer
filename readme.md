
# SE(3) Manifold Flow Matching Transformer

This repository serves as a practical tutorial and reference implementation for Continuous-Time Generative Modeling on the SE(3) Manifold. It demonstrates how to train a Flow Matching model using a Transformer backbone to generate 3D poses (position and orientation) by learning and integrating spatial velocities (twists).

Unlike standard diffusion or flow matching models that operate in flat Euclidean space, this codebase is built from the ground up to respect the geometry of 3D rotations, making it ideal for advanced robotics applications like behavior cloning, state estimation, and motion planning.

## Core Concepts and Tutorial Overview

This repository is designed to teach several advanced concepts in generative modeling and manifold mathematics.

### 1. Flow Matching on SE(3)
Standard flow matching learns a vector field that transports a simple base distribution (e.g., a standard Gaussian) to a complex data distribution. In this repository, our "data" consists of 3D poses.
* State Representation: The network state is tracked as poses (represented using quaternions or Ortho6D).
* Network Output: The network predicts Twists (v in R^6), representing linear and angular velocities.
* Integration: ODE integration uses the twist exponential map (`add_twist_to_pose` in `utils/tf_utils.py`) to ensure the generated samples stay strictly on the SE(3) manifold.

### 2. Tokenized Flow Matching (Action Chunks)
To support continuous trajectories or multi-joint systems, this repository natively supports Tokenized Flow Matching (inspired by architectures like pi0). Instead of flattening temporal or spatial dimensions, the model processes inputs as sequences `[batch_size, seq_len, dim]`. 
* Joint Denoising: The joint distribution and kinematic constraints of the action chunk are learned implicitly via the Transformer's self-attention mechanism.
* Independent Integration: ODE integration is applied to each token/slot independently using standard SE(3) algebra.

### 3. Geodesic Optimal Transport (OT)
To make learning efficient, flow matching pairs noise samples with target data samples. Rather than pairing them randomly, this implementation uses Geodesic Optimal Transport (`geodesic_optimal_transport_pairing` in `utils/train_utils.py`). It computes the exact pairwise geodesic distances (the magnitude of the twist required to move between poses) and solves the linear sum assignment problem (Hungarian algorithm) to find the shortest paths on the manifold. For sequence data, OT is applied independently per slot across the batch.

### 4. Transformer Backbone and AdaLN
The core architecture (`models/flow_matching_transformer.py`) relies on a sequence-to-sequence Transformer.
* Timestep Conditioning: The continuous time variable t in [0, 1] is embedded using sinusoidal positional encodings and injected into every layer via Adaptive Layer Normalization (AdaLN). This modulates the scale and shift of the features based on the current integration phase.

### 5. Cross-Attention for Multimodal Conditioning
The `ConditionalFlowMatchingTransformerModel` extends the architecture to support goal-directed generation. Discrete actions or observations (e.g., "top", "left") are embedded and passed as context to a Cross-Attention mechanism, allowing the vector field to split into multimodal trajectories based on the specified condition.

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
* `--conditional` / `-C`: Trains the conditional model variant with action-token cross-attention; omit for the unconditional model.
* `--no_ot` / `-NOOT`: Disables Optimal Transport pairing (useful for seeing how OT improves flow straightness).
* `--no_cfg` / `-NOCFG`: Disables classifier-free guidance (no unconditional dropout during training).
* `--n_steps`: Number of interpolation steps per trajectory during training.
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
* `--no_ot` / `-NOOT`, `--no_cfg` / `-NOCFG`, `--n_steps`, `--save_path`: Same semantics as the pose trainer.
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

## Code Structure

* `models/`: Neural network architectures.
  * `support_models.py`: Core components (Multi-Head Attention, Cross-Attention, AdaLN, FeedForward).
  * `flow_matching_transformer.py`: The unconditional base model natively supporting temporal sequences.
  * `conditional_flow_matching_transformer.py`: The action-conditioned sequence model.
* `utils/`: Mathematical and operational utilities.
  * `tf_utils.py`: High-performance, batched SE(3) operations. Handles conversions between quaternions, rotation matrices, Ortho6D, and computing or applying twists via exponential maps.
  * `train_utils.py`: Data generation, geodesic interpolation, and slot-wise Optimal Transport logic.
  * `visualization_utils.py`: 3D plotting utilities for visualizing SE(3) pose trajectories over time.
  * `logging_utils.py`: Utilities for routing output streams, formatting console logs, and tracking metrics.
* `pose_gen_trainer.py`: SE(3) pose-generation training loop with the built-in multimodal goal distribution, minibatch sequence formatting, loss computation, and checkpointing.
* `pose_gen_inference.py`: ODE solver (Euler integration) for sampling SE(3) pose action chunks from the trained vector field, plus 3D trajectory visualization.
* `image_gen_trainer.py`: MNIST training entrypoint. Tokenizes images into 7×7 patch grids and reuses the shared training loop for Euclidean flow matching.
* `image_gen_inference.py`: MNIST sampling entrypoint. Integrates patch-space noise back to images and tiles the noise → denoised trajectory.
* `pyproject.toml`: Project metadata and build configuration, allowing the repository to be installed as a standard Python package.
