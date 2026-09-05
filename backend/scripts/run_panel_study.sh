#!/usr/bin/env bash
# Run the forecasting study across the whole workload panel.
#
# The five-workload version put r = +0.956 on five points, which is not
# significant at any sensible alpha however clean the ordering looked. This runs
# the identical protocol over seventeen samples so the correlation can carry a
# p-value, and reports Spearman alongside Pearson because the claim is about
# ordering rather than linearity.
#
# Build the panel first:  ./scripts/build_workload_panel.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

args=()
for f in data/workload_bb_fs_*.csv data/workload_bb_rnd_*.csv \
         data/workload_google_*.csv data/workload_azure_*.csv \
         data/workload_alibaba_*.csv data/workload_syn_*.csv; do
  [ -f "$f" ] || continue
  name=$(basename "$f" .csv); name=${name#workload_}
  args+=("${name}=${f}")
done

if [ ${#args[@]} -eq 0 ]; then
  echo "No panel datasets found. Run ./scripts/build_workload_panel.sh first." >&2
  exit 1
fi

echo "Panel: ${#args[@]} workload samples"
printf '  %s\n' "${args[@]}"
echo

python -u scripts/cross_dataset_study.py \
    --datasets "${args[@]}" \
    --out artifacts/panel_study.json \
    "$@"
