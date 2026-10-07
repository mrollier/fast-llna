#!/usr/bin/env bash
# Run on the NVIDIA workstation from the repository root, then paste workstation.log back:
#
#   conda env create -f environment.yml && conda activate fast-llna
#   pip install -e ".[cuda,dev]"          # cupy-cuda12x; use cupy-cuda13x for a CUDA 13 driver
#   bash benchmarks/workstation_job.sh 2>&1 | tee workstation.log
set -uo pipefail

echo "== hardware"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)"
free -g 2>/dev/null | head -2
python - <<'PY'
import cupy, numpy, fast_llna as fl
p = cupy.cuda.runtime.getDeviceProperties(0)
print("backends", fl.available_backends(), "| cupy", cupy.__version__, "| numpy", numpy.__version__)
print("gpu", p["name"].decode(), "| SMs", p["multiProcessorCount"], "| L2 MB", p["l2CacheSize"] / 2**20)
PY

echo "== tests"
python -m pytest -q -p no:warnings

echo "== benchmarks (JSON lines)"
for backend in cuda cpu; do
    python benchmarks/bench.py --scenario all --backend "$backend"
done

echo "== register use / spills / LOP3 (thesis configuration)"
FAST_LLNA_DUMP=/tmp/llna_thesis.cu python benchmarks/bench.py --scenario thesis --backend cuda > /dev/null
arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
if command -v nvcc > /dev/null; then
    nvcc -cubin -std=c++14 -arch="sm_${arch}" -Xptxas -v -o /tmp/llna_thesis.cubin /tmp/llna_thesis.cu
    echo "LOP3 instructions: $(cuobjdump -sass /tmp/llna_thesis.cubin | grep -c LOP3)"
else
    echo "nvcc not found: skipping ptxas report (CuPy compiles with NVRTC, nvcc is optional)"
fi
