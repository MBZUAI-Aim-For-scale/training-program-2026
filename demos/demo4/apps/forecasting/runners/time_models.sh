#!/usr/bin/env bash
# Time models one after another on the Spark (run inside the model container).
#   bash runners/time_models.sh <init YYYY-MM-DDTHH> <lead_hours> model:venv:members [...]
# Appends each run's load/run seconds to outputs/_timings_log.jsonl.
set -u
cd "$(dirname "$0")/.."
init=$1; lead=$2; shift 2
for spec in "$@"; do
  IFS=: read -r model venv members <<<"$spec"
  tag=$(echo "$init" | tr -d -- '-')
  out="outputs/$model/${tag}_${lead}h_m${members}.nc"
  mkdir -p "outputs/$model" outputs/_jobs
  echo '{}' > "outputs/_jobs/time_$model.json"
  echo "=== $model ($members members) $(date +%H:%M:%S)"
  "$venv/bin/python" runners/run_model.py --model "$model" --init "$init" --lead-hours "$lead" \
    --members "$members" --out "$out" --status "outputs/_jobs/time_$model.json" \
    > "outputs/_jobs/time_$model.log" 2>&1
  tail -c 300 "outputs/_jobs/time_$model.json"; echo
done
echo ALL_DONE
