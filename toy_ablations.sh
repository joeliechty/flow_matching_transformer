#!/usr/bin/env bash
# 2-D toy check of the continuous-condition pairings (readme section 6) before the pose runs:
# train every pairing on every toy for every seed, then evaluate them.
#
#   ./toy_ablations.sh [train|eval|all|diagnostics]   (default: all = train + eval)
#
# Toys (utils/toy_tasks.py): moons (C²OT's 8gaussians -> moons, conditioned on x) and fork
# (COT Policy's 1-D fork). Hyperparameters follow C²OT's toy setup (see toy_gen_trainer.py).
#
# Layout:
#   checkpoints/toy_<toy>/seed_<N>/                        checkpoints, configs, logs
#   experiments/results/toy_<toy>/seed_<N>/toy_metrics.csv W₂² and branch metrics per sampler
#   experiments/results/toy_<toy>/toy_metrics_summary.csv  mean ± std over seeds
#   experiments/results/toy_<toy>/pairing_diagnostics.csv  (diagnostics stage)
#
# Env overrides: SEEDS="1 2 3" JOBS=6 TOYS="moons fork" ITERATIONS=20000 PYTHON=python

set -euo pipefail

STAGE="${1:-all}"
SEEDS="${SEEDS:-1 2 3}"
JOBS="${JOBS:-6}"
TOYS="${TOYS:-moons fork}"
ITERATIONS="${ITERATIONS:-20000}"
PYTHON="${PYTHON:-python}"
PAIRINGS=(independent global c2ot c2ot_fixed cluster)
export MPLBACKEND="${MPLBACKEND:-Agg}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$STAGE" in
  train|eval|all|diagnostics) ;;
  *) echo "usage: $0 [train|eval|all|diagnostics]" >&2; exit 2 ;;
esac

if ! "$PYTHON" -c 'import torch' 2>/dev/null; then
  echo "ERROR: '$PYTHON' can't import torch — run 'conda activate FmT' or set PYTHON=" >&2
  exit 1
fi

trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

declare -A SUFFIX=([independent]=NOOT [global]=GOT [c2ot]=C2OT [c2ot_fixed]=C2OTFIX [cluster]=CLUSTER)

train_one() {  # <toy> <seed> <pairing>
  local toy=$1 seed=$2 pairing=$3 dir="checkpoints/toy_$1/seed_$2" start=$SECONDS
  if "$PYTHON" toy_gen_trainer.py --toy "$toy" --pairing "$pairing" --seed "$seed" \
       --iterations "$ITERATIONS" --save_path "$dir/" > "$dir/toy_${toy}_${SUFFIX[$pairing]}_console.txt" 2>&1; then
    echo "  done    $toy seed $seed  $pairing  ($(( SECONDS - start ))s)"
  else
    echo "  FAILED  $toy seed $seed  $pairing — see $dir/toy_${toy}_${SUFFIX[$pairing]}_console.txt" >&2
    return 1
  fi
}

train_stage() {
  echo "=== Training toys [$TOYS]: seeds [$SEEDS], $ITERATIONS iterations, $JOBS at a time ==="
  local running=0 failed=0 toy seed pairing
  for toy in $TOYS; do
    for seed in $SEEDS; do
      mkdir -p "checkpoints/toy_$toy/seed_$seed"
      for pairing in "${PAIRINGS[@]}"; do
        if [[ -f "checkpoints/toy_$toy/seed_$seed/toy_${toy}_${SUFFIX[$pairing]}_final.pt" ]]; then
          echo "  skip    $toy seed $seed  $pairing (already trained)"
          continue
        fi
        if (( running >= JOBS )); then
          wait -n || failed=1
          running=$(( running - 1 ))
        fi
        train_one "$toy" "$seed" "$pairing" &
        running=$(( running + 1 ))
      done
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
  echo "=== Evaluating toys [$TOYS]: seeds [$SEEDS] ==="
  local running=0 failed=0 toy seed
  for toy in $TOYS; do
    for seed in $SEEDS; do
      if (( running >= JOBS )); then
        wait -n || failed=1
        running=$(( running - 1 ))
      fi
      out="experiments/results/toy_$toy/seed_$seed"
      mkdir -p "$out"
      "$PYTHON" experiments/evaluate_toys.py --checkpoint_dir "checkpoints/toy_$toy/seed_$seed" \
        --output "$out/toy_metrics.csv" > "$out/console.txt" 2>&1 \
        || { echo "  FAILED  eval $toy seed $seed — see $out/console.txt" >&2; exit 1; } &
      running=$(( running + 1 ))
    done
  done
  while (( running > 0 )); do
    wait -n || failed=1
    running=$(( running - 1 ))
  done
  if (( failed )); then
    echo "ERROR: some evaluations failed (see above)" >&2
    exit 1
  fi
  for toy in $TOYS; do
    echo "--- $toy ---"
    "$PYTHON" experiments/evaluate_toys.py --aggregate "experiments/results/toy_$toy"
  done
}

diagnostics_stage() {
  local toy
  for toy in $TOYS; do
    "$PYTHON" experiments/pairing_diagnostics.py --task "$toy" --ot_batch 256 1024 \
      --output "experiments/results/toy_$toy/pairing_diagnostics.csv"
  done
}

if [[ "$STAGE" == diagnostics ]]; then diagnostics_stage; exit 0; fi
if [[ "$STAGE" == train || "$STAGE" == all ]]; then train_stage; fi
if [[ "$STAGE" == eval  || "$STAGE" == all ]]; then eval_stage; fi
echo "Done. Summaries in experiments/results/toy_<toy>/"
