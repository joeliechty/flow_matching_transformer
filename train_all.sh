#!/usr/bin/env bash
# Train all 12 ablation variants for the SE(3) pose and MNIST tasks.
# Identical hyperparameters across variants — only OT/CFG/conditional knobs differ.
# Single seed (42).
# (or the project root) for the experiment design.
#
# This script lives in the project parent directory, OUTSIDE the git repo, and
# drives the checkout at ./flow_matching_transformer. Point FMT_REPO_ROOT at a
# different checkout to override.

set -euo pipefail

SEED=42
POSE_EPOCHS=100
IMAGE_EPOCHS=400
BATCH=128

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${FMT_REPO_ROOT:-$SCRIPT_DIR/flow_matching_transformer}"

if [[ ! -f "$REPO_ROOT/pose_gen_trainer.py" ]]; then
  echo "ERROR: no flow_matching_transformer checkout at $REPO_ROOT" >&2
  echo "       set FMT_REPO_ROOT=/path/to/flow_matching_transformer and retry" >&2
  exit 1
fi

cd "$REPO_ROOT"
echo "Repo root: $REPO_ROOT"

echo "=== Training pose models (6 variants, ${POSE_EPOCHS} epochs each) ==="
python pose_gen_trainer.py --conditional             --num_epochs ${POSE_EPOCHS} --batch_size ${BATCH} --seed ${SEED}
python pose_gen_trainer.py --conditional --no_cfg    --num_epochs ${POSE_EPOCHS} --batch_size ${BATCH} --seed ${SEED}
python pose_gen_trainer.py --conditional --no_ot     --num_epochs ${POSE_EPOCHS} --batch_size ${BATCH} --seed ${SEED}
python pose_gen_trainer.py --conditional --no_ot --no_cfg --num_epochs ${POSE_EPOCHS} --batch_size ${BATCH} --seed ${SEED}
python pose_gen_trainer.py                           --num_epochs ${POSE_EPOCHS} --batch_size ${BATCH} --seed ${SEED}
python pose_gen_trainer.py            --no_ot        --num_epochs ${POSE_EPOCHS} --batch_size ${BATCH} --seed ${SEED}

echo "=== Training MNIST models (6 variants, ${IMAGE_EPOCHS} epochs each) ==="
python image_gen_trainer.py --conditional             --num_epochs ${IMAGE_EPOCHS} --batch_size ${BATCH} --seed ${SEED}
python image_gen_trainer.py --conditional --no_cfg    --num_epochs ${IMAGE_EPOCHS} --batch_size ${BATCH} --seed ${SEED}
python image_gen_trainer.py --conditional --no_ot     --num_epochs ${IMAGE_EPOCHS} --batch_size ${BATCH} --seed ${SEED}
python image_gen_trainer.py --conditional --no_ot --no_cfg --num_epochs ${IMAGE_EPOCHS} --batch_size ${BATCH} --seed ${SEED}
python image_gen_trainer.py                           --num_epochs ${IMAGE_EPOCHS} --batch_size ${BATCH} --seed ${SEED}
python image_gen_trainer.py            --no_ot        --num_epochs ${IMAGE_EPOCHS} --batch_size ${BATCH} --seed ${SEED}

echo "=== Training MNIST classifier (eval oracle) ==="
python -m utils.mnist_classifier

echo "All training complete. Run python experiments/evaluate_all.py next."
