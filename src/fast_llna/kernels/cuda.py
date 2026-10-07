"""NVIDIA GPU backend: llna.h compiled by NVRTC through CuPy, one launch per step.

Block (bx, by) = (min(Wp, 32), 256 / bx): a warp is one node x 32 consecutive replica words (coalesced state
loads, warp-uniform degree) once Wp >= 32. Nodes run along grid.x (limit 2^31 - 1), words along grid.y.
Set FAST_LLNA_DUMP=path to write the generated CUDA source (for nvcc -Xptxas -v / SASS inspection).
"""

from __future__ import annotations

import os
from functools import cache

import numpy as np

from . import tables

PRELUDE = """
typedef unsigned int u32;
typedef long long llna_idx;
typedef u32 WORD;
#define LLNA_FN __device__ __forceinline__
#define LLNA_PTR(T) const T *
#define MULHI(a, b) __umulhi((u32)(a), (u32)(b))
#define ZEROW 0u
#define LOADW(ptr, off) __ldg((ptr) + (off))
#define ANYW(x) ((x) != 0u)
#define RANDW(k0, k1, t, i, st, w, d) llna_rand(k0, k1, t, (u32)(i), (st)[w], d)
"""

WRAPPER = """
extern "C" __global__ void llna_step(const u32 *S, const int *indptr, const int *indices, const int *seg_off,
                                     const int *seg_thr, const int *seg_cell, const u32 *one, const u32 *hasf,
                                     const u32 *frac, const u32 *stream, const u32 *cm, const u32 *cv, u32 *nxt,
                                     llna_idx n, llna_idx Wp, u32 k0, u32 k1, u32 t) {
    llna_idx w = (llna_idx)blockIdx.y * blockDim.x + threadIdx.x;
    llna_idx i = (llna_idx)blockIdx.x * blockDim.y + threadIdx.y;
    if (w >= Wp || i >= n) return;
    nxt[i * Wp + w] = llna_update(S, indptr, indices, seg_off, seg_thr, seg_cell, one, hasf, frac, stream, cm,
                                  cv, Wp, k0, k1, t, (int)i, w, 0, 1);
}
"""

ARRAYS = ("indptr", "indices", "seg_off", "seg_thr", "seg_cell", "one", "hasf", "frac", "stream", "cm", "cv")


def available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # ImportError, or CUDA driver/runtime errors on machines without a GPU
        return False


@cache
def _kernel(source: str):
    import cupy as cp

    return cp.RawModule(code=source, options=("-std=c++14",)).get_function("llna_step")


def run(graph, rules, x0, steps, record, clamp, seed, t0, noise_period, threads=None):
    import cupy as cp

    R, N = x0.n_replicas, graph.n
    Wp = tables.gpu_words(R)
    tab = tables.build(graph, rules, x0, clamp, seed, noise_period, Wp)
    source = tab.source(PRELUDE, WRAPPER)
    if os.environ.get("FAST_LLNA_DUMP"):
        with open(os.environ["FAST_LLNA_DUMP"], "w") as f:
            f.write(source)
    kernel = _kernel(source)
    fixed = [cp.asarray(tab.arrays[k]) for k in ARRAYS]
    bx = min(Wp, 32)
    by = 256 // bx
    grid, block = ((N + by - 1) // by, Wp // bx), (bx, by)

    nb = (R + 7) // 8
    rec = 0 if record == "final" else record
    rows = np.empty((steps // rec + 1 if rec else 1, N, nb), np.uint8)  # v1 rows [N, nb], converted at the end
    rows[0] = tab.arrays["state"].view(np.uint8).reshape(N, 4 * Wp)[:, :nb]
    # keep recorded rows on the GPU when they fit comfortably, else copy each one to the host
    on_device = rec and rows.nbytes < 0.5 * cp.cuda.Device().mem_info[0]
    drows = cp.empty(rows.shape, np.uint8) if on_device else None

    cur = cp.asarray(tab.arrays["state"])
    nxt = cp.empty_like(cur)
    for s in range(steps):
        args = (
            cur,
            *fixed,
            nxt,
            np.int64(N),
            np.int64(Wp),
            np.uint32(tab.k0),
            np.uint32(tab.k1),
            np.uint32((t0 + s) % 2**32),
        )
        kernel(grid, block, args)
        cur, nxt = nxt, cur
        if rec and (s + 1) % rec == 0:
            row = cur.view(np.uint8).reshape(N, 4 * Wp)[:, :nb]
            if on_device:
                drows[(s + 1) // rec] = row
            else:
                rows[(s + 1) // rec] = cp.asnumpy(row)
    if on_device:
        rows[1:] = cp.asnumpy(drows[1:])
    if record == "final":
        rows[0] = cp.asnumpy(cur.view(np.uint8).reshape(N, 4 * Wp)[:, :nb])
    return tables.rows_to_frames(rows, R)
