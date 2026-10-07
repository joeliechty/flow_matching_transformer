#!/usr/bin/env bash
# SE(3) pose ablations: train the 6 variants (4 conditional OT x CFG + 2 unconditional
# OT / no-OT) for every seed, then evaluate them. Identical hyperparameters across
# variants — only the OT/CFG/conditional knobs differ.
#
#   ./pose_ablations.sh [train|eval|all|curves|diagnostics|sensitivity]   (default: all = train + eval)
#
# curves: main metrics (100 and 3 sampling steps) at every saved epoch, for training plots.
# diagnostics: pairing statistics at several OT batch sizes, no training (pairing_diagnostics.py).
# sensitivity: retrain one pairing (SENS_PAIRING=c2ot|cluster) with its knob and the OT batch
#              size varied, 3 seeds each, under <dir>/sensitivity/<setting>/.
#
# The variant set follows the task (override with VARIANT_SET):
#   discrete    clean tokens: 4 conditional (OT x CFG) + 2 unconditional variants
#   jitter      noisy tokens (token_sigma > 0): conditional, no CFG, one per pairing
#               (independent, ot = per-condition oracle, global, c2ot, c2ot_fixed, cluster)
#   continuous  `type: continuous` tasks: the jitter set without the oracle
# The jitter and continuous sets pair over OT_BATCH_MULT (default 4) network batches at once.
#
# The task (goal modes and conditioning tokens) comes from TASK_CONFIG, a file in
# configs/pose_tasks/. Each task gets its own folders, <dir> below: the original
# four_corners task keeps the historical "pose"; any other task <name> uses "pose_<name>".
#
# Layout:
#   checkpoints/<dir>/seed_<N>/                     checkpoints (every 10 epochs), configs, logs
#   experiments/results/<dir>/epoch_<E>/seed_<N>/   metrics, CFG + sampling-steps sweeps, grids
#   experiments/results/<dir>/epoch_<E>/            *_summary.csv + plots, mean ± std over seeds
#   experiments/results/<dir>/training_curves/seed_<N>/metrics_epoch<E>_steps<S>.csv   (curves)
#
# Env overrides: SEEDS="1 2 3 4 5" JOBS=6 EPOCHS=100 PYTHON=python VARIANT_SET OT_BATCH_MULT
#                TASK_CONFIG=configs/pose_tasks/four_corners.yaml
#                EVAL_EPOCH=<EPOCHS>  evaluate the checkpoints saved at this epoch
#                CKPT_ROOT, RESULTS_ROOT (relative to the repo root), FMT_REPO_ROOT
# JOBS=6 measured best on an RTX 4090 (~2.7x sequential throughput; 12 barely helps).
# Runs whose final-epoch checkpoint already exists are skipped, so re-running resumes.

set -euo pipefail

STAGE="${1:-all}"
SEEDS="${SEEDS:-1 2 3 4 5}"
JOBS="${JOBS:-6}"
EPOCHS="${EPOCHS:-100}"
EVAL_EPOCH="${EVAL_EPOCH:-$EPOCHS}"
BATCH=128
PYTHON="${PYTHON:-python}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${FMT_REPO_ROOT:-$SCRIPT_DIR}"
TASK_CONFIG="${TASK_CONFIG:-configs/pose_tasks/four_corners.yaml}"
TASK="$(basename "$TASK_CONFIG" .yaml)"
if [[ "$TASK" == four_corners ]]; then TASK_DIR=pose; else TASK_DIR="pose_$TASK"; fi
CKPT_ROOT="${CKPT_ROOT:-checkpoints/$TASK_DIR}"
RESULTS_ROOT="${RESULTS_ROOT:-experiments/results/$TASK_DIR/epoch_$EVAL_EPOCH}"

case "$STAGE" in
  train|eval|all|curves|diagnostics|sensitivity) ;;
  *) echo "usage: $0 [train|eval|all|curves|diagnostics|sensitivity]" >&2; exit 2 ;;
esac

if (( EVAL_EPOCH % 10 != 0 )); then
  echo "ERROR: EVAL_EPOCH=$EVAL_EPOCH — checkpoints are only saved every 10 epochs" >&2
  exit 2
fi

if [[ ! -f "$REPO_ROOT/pose_gen_trainer.py" ]]; then
  echo "ERROR: no flow_matching_transformer checkout at $REPO_ROOT" >&2
  echo "       set FMT_REPO_ROOT=/path/to/flow_matching_transformer and retry" >&2
  exit 1
fi
cd "$REPO_ROOT"

if [[ ! -f "$TASK_CONFIG" ]]; then
  echo "ERROR: no task file at $TASK_CONFIG" >&2
  exit 1
fi

if ! "$PYTHON" -c 'import torch' 2>/dev/null; then
  echo "ERROR: '$PYTHON' can't import torch — run 'conda activate FmT' or set PYTHON=" >&2
  exit 1
fi

if [[ -z "${VARIANT_SET:-}" ]]; then
  if grep -Eq '^type: *continuous' "$TASK_CONFIG"; then VARIANT_SET=continuous
  elif grep -Eq '^token_sigma: *0*\.?0*[1-9]' "$TASK_CONFIG"; then VARIANT_SET=jitter
  else VARIANT_SET=discrete; fi
fi

# checkpoint name | trainer flags
case "$VARIANT_SET" in
  discrete)
    VARIANTS=(
      "cond_pose_flow_matching_model_OT_CFG|--conditional"
      "cond_pose_flow_matching_model_OT_NOCFG|--conditional --no_cfg"
      "cond_pose_flow_matching_model_NOOT_CFG|--conditional --no_ot"
      "cond_pose_flow_matching_model_NOOT_NOCFG|--conditional --no_ot --no_cfg"
      "pose_flow_matching_model_OT_CFG|"
      "pose_flow_matching_model_NOOT_CFG|--no_ot"
    )
    EXTRA_FLAGS=""; NUM_SAMPLES=256 ;;
  jitter|continuous)
    VARIANTS=("cond_pose_flow_matching_model_NOOT_NOCFG|--conditional --no_cfg --pairing independent")
    if [[ "$VARIANT_SET" == jitter ]]; then
      VARIANTS+=("cond_pose_flow_matching_model_OT_NOCFG|--conditional --no_cfg --pairing ot")
    fi
    VARIANTS+=(
      "cond_pose_flow_matching_model_GOT_NOCFG|--conditional --no_cfg --pairing global"
      "cond_pose_flow_matching_model_C2OT_NOCFG|--conditional --no_cfg --pairing c2ot"
      "cond_pose_flow_matching_model_C2OTFIX_NOCFG|--conditional --no_cfg --pairing c2ot_fixed"
      "cond_pose_flow_matching_model_CLUSTER_NOCFG|--conditional --no_cfg --pairing cluster"
    )
    EXTRA_FLAGS="--ot_batch_mult ${OT_BATCH_MULT:-4}"
    # the continuous task scores 16 test conditions, 64 samples each
    if [[ "$VARIANT_SET" == continuous ]]; then NUM_SAMPLES=1024; else NUM_SAMPLES=256; fi ;;
  *) echo "ERROR: unknown VARIANT_SET=$VARIANT_SET" >&2; exit 2 ;;
esac
echo "Task $TASK: variant set $VARIANT_SET"

trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

train_one() {  # <seed> <checkpoint name> [trainer flags...]
  local seed=$1 name=$2; shift 2
  local dir="$CKPT_ROOT/seed_$seed" start=$SECONDS
  # shellcheck disable=SC2086  # EXTRA_FLAGS is intentionally word-split
  if "$PYTHON" pose_gen_trainer.py "$@" $EXTRA_FLAGS --num_epochs "$EPOCHS" --batch_size "$BATCH" \
       --seed "$seed" --task_config "$TASK_CONFIG" --save_path "$dir/" \
       > "$dir/${name}_console.txt" 2>&1; then
    echo "  done    seed $seed  $name  ($(( SECONDS - start ))s)"
  else
    echo "  FAILED  seed $seed  $name — see $dir/${name}_console.txt" >&2
    return 1
  fi
}

train_stage() {
  echo "=== Training pose ablations ($TASK): seeds [$SEEDS], $EPOCHS epochs, $JOBS at a time ==="
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
  echo "=== Evaluating pose ablations ($TASK): seeds [$SEEDS], epoch-$EVAL_EPOCH checkpoints ==="
  local seed ckpt out
  for seed in $SEEDS; do
    ckpt="$CKPT_ROOT/seed_$seed"; out="$RESULTS_ROOT/seed_$seed"
    if [[ ! -d "$ckpt" ]]; then
      echo "ERROR: no checkpoints at $ckpt — run '$0 train' first" >&2
      exit 1
    fi
    echo "--- seed $seed ---"
    "$PYTHON" experiments/evaluate_all.py --checkpoint_dir "$ckpt" --epoch "$EVAL_EPOCH" \
      --num_samples "$NUM_SAMPLES" --output "$out/metrics.csv"
    "$PYTHON" experiments/steps_sweep.py  --checkpoint_dir "$ckpt" --epoch "$EVAL_EPOCH" \
      --num_samples "$NUM_SAMPLES" --output "$out/steps_sweep.csv"
    # guidance and the 3-D grids only apply to the discrete set (CFG-trained, token conditions)
    if [[ "$VARIANT_SET" == discrete ]]; then
      "$PYTHON" experiments/cfg_sweep.py  --checkpoint_dir "$ckpt" --epoch "$EVAL_EPOCH" --output "$out/cfg_sweep.csv"
      "$PYTHON" experiments/make_grids.py --checkpoint_dir "$ckpt" --epoch "$EVAL_EPOCH" --results_dir "$out" --skip_image
    fi
  done
  "$PYTHON" experiments/aggregate_seeds.py --results_dir "$RESULTS_ROOT"
}

curve_one() {  # <seed> <epoch> <steps>
  local seed=$1 epoch=$2 steps=$3 out="$CURVES_ROOT/seed_$1"
  local csv="$out/metrics_epoch${epoch}_steps${steps}.csv"
  [[ -f "$csv" ]] && return 0
  mkdir -p "$out"
  "$PYTHON" experiments/evaluate_all.py --checkpoint_dir "$CKPT_ROOT/seed_$seed" --epoch "$epoch" \
    --num_steps "$steps" --num_samples "$NUM_SAMPLES" --output "$csv" \
    > "$out/metrics_epoch${epoch}_steps${steps}_console.txt" 2>&1 \
    || { echo "  FAILED  seed $seed epoch $epoch, $steps steps" >&2; return 1; }
}

curves_stage() {
  CURVES_ROOT="experiments/results/$TASK_DIR/training_curves"
  echo "=== Training curves ($TASK): seeds [$SEEDS], every 10 epochs to $EPOCHS, $JOBS at a time ==="
  local running=0 failed=0 seed epoch steps
  for seed in $SEEDS; do
    for (( epoch = 10; epoch <= EPOCHS; epoch += 10 )); do
      for steps in 100 3; do
        if (( running >= JOBS )); then
          wait -n || failed=1
          running=$(( running - 1 ))
        fi
        curve_one "$seed" "$epoch" "$steps" &
        running=$(( running + 1 ))
      done
    done
  done
  while (( running > 0 )); do
    wait -n || failed=1
    running=$(( running - 1 ))
  done
  if (( failed )); then
    echo "ERROR: some curve evaluations failed (see above)" >&2
    exit 1
  fi
  echo "Done. Curves in $CURVES_ROOT/"
}

diagnostics_stage() {
  echo "=== Pairing diagnostics ($TASK): OT batches of 1, 4 and 10 network batches ==="
  "$PYTHON" experiments/pairing_diagnostics.py --task "$TASK_CONFIG" \
    --ot_batch "$BATCH" $(( 4 * BATCH )) $(( 10 * BATCH )) \
    --output "experiments/results/$TASK_DIR/pairing_diagnostics.csv"
}

sensitivity_stage() {
  # One pairing, its knob and the OT batch varied around the defaults (r_tar 0.01, K = OT batch,
  # OT batch = 4 network batches), which the main runs already cover.
  local pairing="${SENS_PAIRING:?set SENS_PAIRING=c2ot or cluster}" suffix settings entry
  case "$pairing" in
    c2ot)    suffix=C2OT
             settings=("r_tar_0.005|--r_tar 0.005" "r_tar_0.03|--r_tar 0.03" "r_tar_0.1|--r_tar 0.1") ;;
    cluster) suffix=CLUSTER
             settings=("K_$(( BATCH ))|--num_clusters $(( BATCH ))" "K_$(( 16 * BATCH ))|--num_clusters $(( 16 * BATCH ))") ;;
    *) echo "ERROR: SENS_PAIRING must be c2ot or cluster" >&2; exit 2 ;;
  esac
  settings+=("ot_batch_x1|--ot_batch_mult 1" "ot_batch_x10|--ot_batch_mult 10")
  VARIANTS=("cond_pose_flow_matching_model_${suffix}_NOCFG|--conditional --no_cfg --pairing $pairing")
  SEEDS="${SENS_SEEDS:-1 2 3}"
  local base_flags="$EXTRA_FLAGS" name
  for entry in "${settings[@]}"; do
    name="${entry%%|*}"
    EXTRA_FLAGS="$base_flags ${entry#*|}"   # later flags win
    CKPT_ROOT="checkpoints/$TASK_DIR/sensitivity/$name"
    RESULTS_ROOT="experiments/results/$TASK_DIR/sensitivity/$name/epoch_$EVAL_EPOCH"
    echo "=== Sensitivity: $pairing, $name ==="
    train_stage
    eval_stage
  done
}

if [[ "$STAGE" == curves ]]; then curves_stage; exit 0; fi
if [[ "$STAGE" == diagnostics ]]; then diagnostics_stage; exit 0; fi
if [[ "$STAGE" == sensitivity ]]; then sensitivity_stage; exit 0; fi
if [[ "$STAGE" == train || "$STAGE" == all ]]; then train_stage; fi
if [[ "$STAGE" == eval  || "$STAGE" == all ]]; then eval_stage; fi
echo "Done. Summaries in $RESULTS_ROOT/"
