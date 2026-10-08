#!/usr/bin/env bash
# Regenerate the example notebooks from their builders and execute them in place.
# Run inside the fast-llna environment with the notebook extra (pip install -e ".[notebook]").
# consensus_sweep.ipynb only reads notebooks/results/, written by `python notebooks/consensus_sweep.py`.
#
#   bash notebooks/build/build_all.sh            # all three
#   bash notebooks/build/build_all.sh r9 sweep   # a subset
set -euo pipefail
cd "$(dirname "$0")/.."  # notebooks/
for name in "${@:-r9 r8 sweep}"; do
  for n in $name; do
    nb="consensus_$n.ipynb"
    python "build/build_$n.py" "$nb"
    /usr/bin/time -p jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3000 "$nb"
  done
done
