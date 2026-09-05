#!/usr/bin/env bash
# Build the panel of workload samples the diff_acf1 claim is measured on.
#
# The five-workload version of that claim rested on a correlation over five
# points, which is not significant at any sensible alpha however clean it looks.
# This widens it to seventeen samples without pretending they are seventeen
# independent datacentres.
#
# Two devices, both honest about what they buy:
#
#   Disjoint entity shards. `--shard K/N` partitions one shuffled entity list, so
#   the shards provably share no VMs. Two random seeds would NOT do this - draws
#   from the same pool overlap, and counting the same VM twice would inflate the
#   sample without adding information.
#
#   Genuinely separate collections. Bitbrains publishes four: fastStorage (1,241
#   VMs) and three Rnd months (500 VMs each, different VMs, July/August/September
#   2013). Those three are the most valuable additions here because they differ in
#   both period and population.
#
# What this does not claim: shards of one collection share a datacentre, a month
# and a diurnal cycle, so they are correlated samples, not independent workloads.
# The analysis reports that.
#
#   ./scripts/build_workload_panel.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

fetch() { echo; echo "### $*"; python -u scripts/fetch_trace.py "$@"; }

# Bitbrains fastStorage - three disjoint thirds of 1,241 VMs
for k in 0 1 2; do
  fetch --dataset bitbrains --collection 1201308 --shard "$k/3" \
        --entities 300 --seed 42 --name "bb_fs_$k"
done

# Bitbrains Rnd - three different months, different VMs
for m in 201307 201308 201309; do
  fetch --dataset bitbrains --collection "$m" --shard 0/1 \
        --entities 300 --seed 42 --name "bb_rnd_${m: -2}"
done

# Google Borg - three disjoint thirds of 1,635 tasks
for k in 0 1 2; do
  fetch --dataset google --shard "$k/3" --entities 300 --seed 42 --name "google_$k"
done

# Azure - three disjoint thirds of 1,195 VMs
for k in 0 1 2; do
  fetch --dataset azure --shard "$k/3" --entities 300 --seed 42 --name "azure_$k"
done

# Alibaba - two disjoint halves of 1,000 machines
for k in 0 1; do
  fetch --dataset alibaba --shard "$k/2" --entities 400 --seed 42 --name "alibaba_$k"
done

# Synthetic - three generator seeds, as a reference point on the same axis
for seed in 7 42 99; do
  echo; echo "### synthetic seed $seed"
  python -u scripts/generate_data.py --days 30 --interval 5 --seed "$seed" \
         --out "data/workload_syn_$seed.csv"
done

echo
echo "Panel built:"
ls -1 data/workload_bb_*.csv data/workload_google_*.csv data/workload_azure_*.csv \
      data/workload_alibaba_*.csv data/workload_syn_*.csv 2>/dev/null | wc -l
