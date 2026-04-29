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

### 2. Geodesic Optimal Transport (OT)
To make learning efficient, flow matching pairs noise samples with target data samples. Rather than pairing them randomly, this implementation uses Geodesic Optimal Transport (`geodesic_optimal_transport_pairing` in `utils/train_utils.py`). It computes the exact pairwise geodesic distances (the magnitude of the twist required to move between poses) and solves the linear sum assignment problem (Hungarian algorithm) to find the shortest paths on the manifold.

### 3. Transformer Backbone and AdaLN
The core architecture (`models/flow_matching_transformer.py`) relies on a sequence-to-sequence Transformer.
* Timestep Conditioning: The continuous time variable t in [0, 1] is embedded using sinusoidal positional encodings and injected into every layer via Adaptive Layer Normalization (AdaLN). This modulates the scale and shift of the features based on the current integration phase.

### 4. Cross-Attention for Multimodal Conditioning
The `ConditionalFlowMatchingTransformerModel` extends the architecture to support goal-directed generation. Discrete actions or observations (e.g., "top", "left") are embedded and passed as context to a Cross-Attention mechanism, allowing the vector field to split into multimodal trajectories based on the specified condition.

## Installation

The repository uses Conda to manage dependencies and isolate the environment.

For Linux (CUDA):
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

### 1. Training the Model

The `train_FmT.py` script contains a built-in toy dataset representing start poses at the origin and multimodal goal poses (rotated corners).

Train an Unconditional Model (with Optimal Transport):
```bash
python train_FmT.py --num_epochs 50 --batch_size 128
```

Train a Conditional Model (Action-Directed):
This trains the model to associate specific target modes with specific conditioning tokens.
```bash
python train_FmT.py --conditional --num_epochs 50
```

Useful Flags:
* `--no_ot`: Disables Optimal Transport pairing (useful for seeing how OT improves flow straightness).
* `--n_steps`: Number of interpolation steps per trajectory during training.
* `--save_path`: Directory to save `.pt` checkpoints and `.yaml` config files.

### 2. Running Inference and Visualization

The `inference.py` script loads a trained checkpoint, integrates the learned ODE vector field, and visualizes the resulting SE(3) trajectories using Matplotlib.

Unconditional Inference:
```bash
python inference.py --checkpoint_epoch 50 --num_samples 10 --return_trajectory
```

Conditional Inference (Guiding the Flow):
If you trained a conditional model, you can force the flow toward specific modes by combining action tokens.
```bash
# Force the flow to the top-right mode
python inference.py --conditional --actions top right --num_samples 5 --return_trajectory
```

When you run inference with the `--return_trajectory` flag, the script will automatically generate a 3D plot showing the positional paths and coordinate frame axes (RGB = XYZ) evolving over time.

## Code Structure

* `models/`: Neural network architectures.
  * `support_models.py`: Core components (Multi-Head Attention, Cross-Attention, AdaLN, FeedForward).
  * `flow_matching_transformer.py`: The unconditional base model.
  * `conditional_flow_matching_transformer.py`: The action-conditioned model.
* `utils/`: Mathematical and operational utilities.
  * `tf_utils.py`: High-performance, batched SE(3) operations. Handles conversions between quaternions, rotation matrices, Ortho6D, and computing/applying twists via exponential maps.
  * `train_utils.py`: Data generation, geodesic interpolation, and Optimal Transport logic.
  * `visualization_utils.py`: 3D plotting utilities for visualizing SE(3) pose trajectories over time.
* `train_FmT.py`: The main training loop, loss computation, and checkpointing logic.
* `inference.py`: ODE solver (Euler integration) for sampling from the trained vector field.
