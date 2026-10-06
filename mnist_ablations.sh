#!/usr/bin/env bash
# MNIST ablations: train the evaluation classifier, then the 6 variants (4 conditional
# OT x CFG + 2 unconditional OT / no-OT) for every seed, then evaluate them. Identical
# hyperparameters across variants — only the OT/CFG/conditional knobs differ.
#
#   ./mnist_ablations.sh [train|eval|all]       (default: all)
#
# Layout:
#   eval_assets/mnist_cnn.pt                        classifier oracle (shared by all seeds)
#   checkpoints/mnist/seed_<N>/                     checkpoints (every 10 epochs), configs, logs
#   experiments/results/mnist/epoch_<E>/seed_<N>/   metrics, CFG + sampling-steps sweeps, grids
#   experiments/results/mnist/epoch_<E>/            *_summary.csv + plots, mean ± std over seeds
#
# Env overrides: SEEDS="1 2 3 4 5" JOBS=1 EPOCHS=400 PYTHON=python CLASSIFIER=...
#                EVAL_EPOCH=<EPOCHS>  evaluate the checkpoints saved at this epoch
#                CKPT_ROOT, RESULTS_ROOT (relative to the repo root), FMT_REPO_ROOT
# Cost: one 400-epoch run is ~26 min on an RTX 4090, so the default 5 seeds x 6
# variants is ~13 h at JOBS=1 (parallel speedup for MNIST hasn't been measured).
# Runs whose final-epoch checkpoint already exists are skipped, so re-running resumes.

set -euo pipefail

STAGE="${1:-all}"
SEEDS="${SEEDS:-1 2 3 4 5}"
JOBS="${JOBS:-1}"
EPOCHS="${EPOCHS:-400}"
EVAL_EPOCH="${EVAL_EPOCH:-$EPOCHS}"
BATCH=128
PYTHON="${PYTHON:-python}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${FMT_REPO_ROOT:-$SCRIPT_DIR}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints/mnist}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/results/mnist/epoch_$EVAL_EPOCH}"
CLASSIFIER="${CLASSIFIER:-eval_assets/mnist_cnn.pt}"

case "$STAGE" in
  train|eval|all) ;;
  *) echo "usage: $0 [train|eval|all]" >&2; exit 2 ;;
esac

if (( EVAL_EPOCH % 10 != 0 )); then
  echo "ERROR: EVAL_EPOCH=$EVAL_EPOCH — checkpoints are only saved every 10 epochs" >&2
  exit 2
fi

if [[ ! -f "$REPO_ROOT/image_gen_trainer.py" ]]; then
  echo "ERROR: no flow_matching_transformer checkout at $REPO_ROOT" >&2
  echo "       set FMT_REPO_ROOT=/path/to/flow_matching_transformer and retry" >&2
  exit 1
fi
cd "$REPO_ROOT"

if ! "$PYTHON" -c 'import torch' 2>/dev/null; then
  echo "ERROR: '$PYTHON' can't import torch — run 'conda activate FmT' or set PYTHON=" >&2
  exit 1
fi

# checkpoint name | trainer flags
VARIANTS=(
  "cond_image_flow_matching_model_OT_CFG|--conditional"
  "cond_image_flow_matching_model_OT_NOCFG|--conditional --no_cfg"
  "cond_image_flow_matching_model_NOOT_CFG|--conditional --no_ot"
  "cond_image_flow_matching_model_NOOT_NOCFG|--conditional --no_ot --no_cfg"
  "image_flow_matching_model_OT_CFG|"
  "image_flow_matching_model_NOOT_CFG|--no_ot"
)

trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

train_one() {  # <seed> <checkpoint name> [trainer flags...]
  local seed=$1 name=$2; shift 2
  local dir="$CKPT_ROOT/seed_$seed" start=$SECONDS
  if "$PYTHON" image_gen_trainer.py "$@" --num_epochs "$EPOCHS" --batch_size "$BATCH" \
       --seed "$seed" --save_path "$dir/" > "$dir/${name}_console.txt" 2>&1; then
    echo "  done    seed $seed  $name  ($(( SECONDS - start ))s)"
  else
    echo "  FAILED  seed $seed  $name — see $dir/${name}_console.txt" >&2
    return 1
  fi
}

train_stage() {
  if [[ -f "$CLASSIFIER" ]]; then
    echo "=== MNIST classifier already at $CLASSIFIER ==="
  else
    echo "=== Training MNIST classifier (eval oracle) -> $CLASSIFIER ==="
    "$PYTHON" -m utils.mnist_classifier --save_path "$CLASSIFIER"
  fi

  echo "=== Training MNIST ablations: seeds [$SEEDS], $EPOCHS epochs, $JOBS at a time ==="
  local running=0 failed=0 seed entry name flags
  for seed in $SEEDS; do
    mkdir -p "$CKPT_ROOT/seed_$seed"
    for entry in "${VARIANTS[@]}"; do
      name="${entry%%|*}"; flags="${entry#*|}"
      if [[ -f "$CKPT_ROOT/seed_$seed/${name}_epoch_${EPOCHS}.pt" ]]; then
        echo "  skip    seed $seed  $name (already trained)"
        continue
      fi
      if (( running >= JOBS )); then
        wait -n || failed=1
        running=$(( running - 1 ))
      fi
      # shellcheck disable=SC2086  # flags are intentionally word-split
      train_one "$seed" "$name" $flags &
      running=$(( running + 1 ))
    done
  done
  while (( running > 0 )); do
    wait -n || failed=1
    running=$(( running - 1 ))
  done
  if (( failed )); then
    echo "ERROR: some training runs failed (see above)" >&2
    exit 1
  fi
}

eval_stage() {
  if [[ ! -f "$CLASSIFIER" ]]; then
    echo "ERROR: no MNIST classifier at $CLASSIFIER — run '$0 train' first" >&2
    exit 1
  fi
  echo "=== Evaluating MNIST ablations: seeds [$SEEDS], epoch-$EVAL_EPOCH checkpoints ==="
  local seed ckpt out
  for seed in $SEEDS; do
    ckpt="$CKPT_ROOT/seed_$seed"; out="$RESULTS_ROOT/seed_$seed"
    if [[ ! -d "$ckpt" ]]; then
      echo "ERROR: no checkpoints at $ckpt — run '$0 train' first" >&2
      exit 1
    fi
    echo "--- seed $seed ---"
    "$PYTHON" experiments/evaluate_all.py --checkpoint_dir "$ckpt" --epoch "$EVAL_EPOCH" --classifier_path "$CLASSIFIER" --output "$out/metrics.csv"
    "$PYTHON" experiments/cfg_sweep.py    --checkpoint_dir "$ckpt" --epoch "$EVAL_EPOCH" --classifier_path "$CLASSIFIER" --output "$out/cfg_sweep.csv"
    "$PYTHON" experiments/steps_sweep.py  --checkpoint_dir "$ckpt" --epoch "$EVAL_EPOCH" --classifier_path "$CLASSIFIER" --output "$out/steps_sweep.csv"
    "$PYTHON" experiments/make_grids.py   --checkpoint_dir "$ckpt" --epoch "$EVAL_EPOCH" --results_dir "$out" --skip_pose
  done
  "$PYTHON" experiments/aggregate_seeds.py --results_dir "$RESULTS_ROOT"
}

if [[ "$STAGE" == train || "$STAGE" == all ]]; then train_stage; fi
if [[ "$STAGE" == eval  || "$STAGE" == all ]]; then eval_stage; fi
echo "Done. Summaries in $RESULTS_ROOT/"
