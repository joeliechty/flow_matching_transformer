#!/usr/bin/env bash
# MNIST ablations: train the evaluation classifier, then the 6 variants (4 conditional
# OT x CFG + 2 unconditional OT / no-OT) for every seed, then evaluate them. Identical
# hyperparameters across variants — only the OT/CFG/conditional knobs differ.
#
#   ./mnist_ablations.sh [train|eval|all|ablate|solver|guidance]       (default: all)
#
# ablate, solver, guidance: as in pose_ablations.sh (single-axis ablations of ABLATE_BASE under
# ablate/<arm>/; solvers at equal network evaluations; guidance intervals). See docs/mnist_todo.md.
#
# Layout (runs from before the current model and training framework are in checkpoints/mnist/
# and checkpoints/pre_ablation/; this code can't load them, see tag pose-continuous-v1):
#   eval_assets/mnist_cnn.pt                             classifier oracle (shared by all seeds)
#   checkpoints/sota/mnist/seed_<N>/                     checkpoints (every 10 epochs), configs, logs
#   experiments/results/sota/mnist/epoch_<E>/seed_<N>/   metrics, CFG + sampling-steps sweeps, grids
#   experiments/results/sota/mnist/epoch_<E>/            *_summary.csv + plots, mean ± std over seeds
#
# Env overrides: SEEDS="1 2 3 4 5" JOBS=1 EPOCHS=400 PYTHON=python CLASSIFIER=...
#                EVAL_EPOCH=<EPOCHS>  evaluate the checkpoints saved at this epoch
#                CKPT_ROOT, RESULTS_BASE (the per-task results folder), RESULTS_ROOT (relative to
#                the repo root), FMT_REPO_ROOT
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
CKPT_ROOT="${CKPT_ROOT:-checkpoints/sota/mnist}"
RESULTS_BASE="${RESULTS_BASE:-experiments/results/sota/mnist}"
RESULTS_ROOT="${RESULTS_ROOT:-$RESULTS_BASE/epoch_$EVAL_EPOCH}"
CLASSIFIER="${CLASSIFIER:-eval_assets/mnist_cnn.pt}"

case "$STAGE" in
  train|eval|all|ablate|solver|guidance) ;;
  *) echo "usage: $0 [train|eval|all|ablate|solver|guidance]" >&2; exit 2 ;;
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

# Single-axis ablation arms: "<arm>|<trainer flags>", each changing one contested choice
ABLATION_ARMS=(
  "t_logit_normal|--t_dist logit_normal"
  "t_beta|--t_dist beta"
  "time_x1000|--time_emb sinusoidal_x1000"
  "time_fourier|--time_emb fourier"
  "cond_cross_attn|--cond_mode cross_attn"
  "cond_joint|--cond_mode joint"
  "cfg|"
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

train_pool() {  # <root> <"setting|checkpoint name|trainer flags">...
  # Every setting x seed from one job pool, into <root>/<setting>/seed_<N>/.
  local root=$1; shift
  local entry setting rest name flags seed running=0 failed=0
  for entry in "$@"; do
    setting="${entry%%|*}"; rest="${entry#*|}"; name="${rest%%|*}"; flags="${rest#*|}"
    for seed in $SEEDS; do
      mkdir -p "$root/$setting/seed_$seed"
      if [[ -f "$root/$setting/seed_$seed/${name}_epoch_${EPOCHS}.pt" ]]; then
        echo "  skip    $setting seed $seed (already trained)"
        continue
      fi
      if (( running >= JOBS )); then
        wait -n || failed=1
        running=$(( running - 1 ))
      fi
      # shellcheck disable=SC2086  # flags are intentionally word-split
      ( CKPT_ROOT="$root/$setting"; train_one "$seed" "$name" $flags ) &
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

ablate_stage() {
  local base="${ABLATE_BASE:?set ABLATE_BASE=\"<checkpoint name>|<trainer flags>\" (the baseline variant)}"
  local base_name="${base%%|*}" base_flags="${base#*|}" root="$CKPT_ROOT/ablate"
  local wanted=" ${ABLATE_ARMS:-t_logit_normal t_beta time_x1000 time_fourier cond_cross_attn cond_joint} "
  local entries=() arms=() entry arm
  for entry in "${ABLATION_ARMS[@]}"; do
    arm="${entry%%|*}"
    [[ "$wanted" == *" $arm "* ]] || continue
    arms+=("$arm")
    if [[ "$arm" == cfg ]]; then  # the baseline, trained with CFG
      entries+=("cfg|${base_name/_NOCFG/_CFG}|${base_flags/--no_cfg/}")
    else
      entries+=("$arm|$base_name|$base_flags ${entry#*|}")
    fi
  done
  echo "=== MNIST ablations: base $base_name, arms [${arms[*]}], seeds [$SEEDS], $JOBS at a time ==="
  train_pool "$root" "${entries[@]}"
  for arm in "${arms[@]}"; do
    CKPT_ROOT="$root/$arm"
    RESULTS_ROOT="$RESULTS_BASE/ablate/$arm/epoch_$EVAL_EPOCH"
    echo "=== Ablation eval: $arm ==="
    eval_stage
  done
}

solver_stage() {
  echo "=== MNIST solvers at equal network evaluations: seeds [$SEEDS], epoch $EVAL_EPOCH ==="
  local seed
  for seed in $SEEDS; do
    "$PYTHON" experiments/steps_sweep.py --checkpoint_dir "$CKPT_ROOT/seed_$seed" --epoch "$EVAL_EPOCH" \
      --classifier_path "$CLASSIFIER" --cfg_scale 1.0 --method euler midpoint heun \
      --steps 2 4 6 10 18 40 100 --output "$RESULTS_ROOT/seed_$seed/solver_sweep.csv"
  done
  "$PYTHON" experiments/aggregate_seeds.py --results_dir "$RESULTS_ROOT"
}

guidance_stage() {
  echo "=== MNIST guidance intervals: CFG models in $CKPT_ROOT, seeds [$SEEDS], epoch $EVAL_EPOCH ==="
  local seed
  for seed in $SEEDS; do
    "$PYTHON" experiments/cfg_sweep.py --checkpoint_dir "$CKPT_ROOT/seed_$seed" --epoch "$EVAL_EPOCH" \
      --classifier_path "$CLASSIFIER" --scales 1 1.5 3 --intervals full 0-0.5 0.25-0.75 0.5-1 \
      --num_steps 9 100 --output "$RESULTS_ROOT/seed_$seed/guidance_sweep.csv"
  done
  "$PYTHON" experiments/aggregate_seeds.py --results_dir "$RESULTS_ROOT"
}

if [[ "$STAGE" == ablate ]]; then ablate_stage; exit 0; fi
if [[ "$STAGE" == solver ]]; then solver_stage; exit 0; fi
if [[ "$STAGE" == guidance ]]; then guidance_stage; exit 0; fi
if [[ "$STAGE" == train || "$STAGE" == all ]]; then train_stage; fi
if [[ "$STAGE" == eval  || "$STAGE" == all ]]; then eval_stage; fi
echo "Done. Summaries in $RESULTS_ROOT/"
