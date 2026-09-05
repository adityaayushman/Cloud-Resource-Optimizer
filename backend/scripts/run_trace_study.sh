#!/usr/bin/env bash
# Full evaluation pipeline for one recorded trace: train the predictors and
# agents on it, then run the controlled ablation against it.
#
# Fleet caps are per-trace because the traces differ in scale by two orders of
# magnitude. The cap is a safety bound on runaway scale-up, not a tuning knob -
# it is set well above the fleet any policy actually reaches, so it never binds
# and never favours one arm.
#
#   ./scripts/run_trace_study.sh google 40
#   ./scripts/run_trace_study.sh azure  40
#
set -euo pipefail

DATASET="${1:?usage: run_trace_study.sh <dataset> [max_fleet]}"
MAX_FLEET="${2:-40}"
EPISODES="${3:-12}"

cd "$(dirname "$0")/.."

TRACE="data/workload_${DATASET}.csv"
ARTIFACTS="artifacts_${DATASET}"

[ -f "$TRACE" ] || { echo "missing $TRACE - run scripts/fetch_trace.py --dataset $DATASET" >&2; exit 1; }

echo "=== ${DATASET}: training (max_fleet=${MAX_FLEET}, ${EPISODES} RL episodes) ==="
python -u scripts/train.py \
    --data "$TRACE" \
    --artifacts "$ARTIFACTS" \
    --trace "$TRACE" \
    --max-fleet "$MAX_FLEET" \
    --no-tune \
    --rl-episodes "$EPISODES"

echo
echo "=== ${DATASET}: ablation ==="
python -u scripts/evaluate.py \
    --trace "$TRACE" \
    --artifacts "$ARTIFACTS" \
    --max-fleet "$MAX_FLEET"

echo
echo "=== ${DATASET}: horizon study ==="
python -u scripts/horizon_study.py \
    --trace "$TRACE" \
    --seeds 1 2 3 \
    --out "${ARTIFACTS}/horizon_study.json"

echo
echo "${DATASET} complete -> ${ARTIFACTS}/"
