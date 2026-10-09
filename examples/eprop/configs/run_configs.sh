#!/bin/bash
# Run Snake configs ONE AT A TIME (never in parallel on the GPU).
# usage: configs/run_configs.sh "SEEDS" CONFIG.json [CONFIG.json ...]
#   e.g. configs/run_configs.sh "0 1 2" configs/depth_gpu/d3_*.json
# Each run gets its own folder runs/<config name>_s<seed>/ (config with the seed, log, outputs).
# Set PYTHON to the interpreter with pygenn if it is not `python`.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python}
SEEDS=$1; shift
for cfg in "$@"; do
  name=$(basename "$cfg" .json)
  for seed in $SEEDS; do
    dir="$HERE/runs/${name}_s${seed}"
    mkdir -p "$dir"
    "$PY" -c "import json,sys; c=json.load(open(sys.argv[1])); c.update(seed=int(sys.argv[2]), csv_prefix=sys.argv[3], repetition=int(sys.argv[2])); json.dump(c, open(sys.argv[4],'w'), indent=1)" \
      "$cfg" "$seed" "$name" "$dir/config.json"
    echo "[$(date +%H:%M)] $name seed $seed -> $dir"
    (cd "$dir" && SNAKE_CONFIG="$dir/config.json" "$PY" "$HERE/run_headless.py" > log.txt 2>&1)
    echo "[$(date +%H:%M)] $name seed $seed exit $?"
  done
done
