#!/bin/sh
# Start WORKERS background players of the frozen plan on this host.
#   sh experiments/search-scaling/run.sh OUT WORKERS MIN_SIMS MAX_SIMS
# Each worker resumes after any game already recorded in OUT. Logs go to
# OUT/logs/worker-N.log; `pkill -f experiments.search_scaling` stops them.
set -eu
out=$1 workers=$2 min_sims=$3 max_sims=$4
checkpoint=${CHECKPOINT:-$HOME/.cache/manabot/sha256/4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4}
mkdir -p "$out/logs"
export UV_NO_SYNC=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
keep_awake=""
command -v caffeinate >/dev/null 2>&1 && keep_awake="caffeinate -is"
i=0
while [ "$i" -lt "$workers" ]; do
  nohup $keep_awake uv run python -m experiments.search_scaling play \
    --plan experiments/search-scaling/plan-v1.jsonl --checkpoint "$checkpoint" \
    --out "$out" --worker "$i" --workers "$workers" \
    --min-sims "$min_sims" --max-sims "$max_sims" \
    >"$out/logs/worker-$i.log" 2>&1 &
  i=$((i + 1))
done
echo "started $workers workers for sims $min_sims..$max_sims into $out"
