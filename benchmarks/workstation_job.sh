#!/usr/bin/env bash
# Run on the NVIDIA workstation, then paste workstation.log back:
#
#   conda env create -f environment.yml && conda activate fast-llna
#   pip install -e ".[cuda,dev]"          # cupy-cuda12x; use cupy-cuda13x for a CUDA 13 driver
#   bash benchmarks/workstation_job.sh 2>&1 | tee workstation.log
#
# No `set -e`: every section runs so the log is complete; the exit status reports a failed test run.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "== hardware"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)"
free -g 2>/dev/null | head -2
echo "commit $(git rev-parse --short HEAD 2>/dev/null)"
python - <<'PY' || exit 1
import cupy, numpy, fast_llna as fl
assert "cuda" in fl.available_backends(), f"CUDA backend unavailable: {fl.available_backends()}"
p = cupy.cuda.runtime.getDeviceProperties(0)
print("backends", fl.available_backends(), "| cupy", cupy.__version__, "| numpy", numpy.__version__)
print("gpu", p["name"].decode(), "| SMs", p["multiProcessorCount"], "| L2 MB", p["l2CacheSize"] / 2**20)
PY

echo "== tests"
python -m pytest -q
status=$?

echo "== benchmarks (JSON lines)"
for backend in cuda cpu; do
    python benchmarks/bench.py --scenario all --backend "$backend"
done

echo "== register use / spills / LOP3 per kernel"
# two generated sources: the thesis configuration (word kernel, L=32) and a narrow stochastic one (node
# kernel, L=1, hubs, clamp) on a power-law graph
FAST_LLNA_DUMP=/tmp/llna_thesis.cu python benchmarks/bench.py --scenario thesis --backend cuda > /dev/null
FAST_LLNA_DUMP=/tmp/llna_narrow.cu python - <<'PY'
import sys
sys.path.insert(0, "benchmarks")
import numpy as np, fast_llna as fl
from graphs import chung_lu
g = chung_lu(100_000, 2.5, 8, seed=0)
clamp = (np.random.default_rng(0).random(g.n) < 0.01, True)
fl.simulate(g, fl.majority(), fl.random_states(g.n, 1, seed=0), 2, backend="cuda", clamp=clamp, record="final")
PY
arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
if command -v nvcc > /dev/null; then
    for cfg in thesis narrow; do
        echo "-- $cfg"
        nvcc -cubin -std=c++14 -arch="sm_${arch}" -Xptxas -v -o "/tmp/llna_$cfg.cubin" "/tmp/llna_$cfg.cu"
        for k in llna_step llna_node; do
            echo "$k LOP3 instructions: $(cuobjdump -sass -fun $k "/tmp/llna_$cfg.cubin" | grep -c LOP3)"
        done
    done
else
    echo "nvcc not found: skipping ptxas report (CuPy compiles with NVRTC, nvcc is optional)"
fi
exit $status
