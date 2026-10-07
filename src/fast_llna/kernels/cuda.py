"""NVIDIA GPU backend: llna.h compiled by NVRTC through CuPy, one launch per step.

R <= 32 (llna_node): one thread per node in blocks of 256; a warp counts each hub (degree > LLNA_HUB)
together and assembles 32/L nodes per output word with __shfl_xor_sync. R > 32 (llna_step): block
(bx, by) = (min(Wp, 32), 256 / bx), a warp is one node x 32 consecutive replica words once Wp >= 32.
Set FAST_LLNA_DUMP=path to write the generated CUDA source (for nvcc -Xptxas -v / SASS inspection).
"""

from __future__ import annotations

import os
from functools import cache

import numpy as np

from ..states import frame_bytes
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
#define LLNA_LANE0(x) (x)
#define RANDW(k0, k1, t, i, st, w, d) llna_rand(k0, k1, t, (u32)(i), (st)[w], d)
#define LLNA_SHFL_XOR(x, m) __shfl_xor_sync(0xffffffffu, (x), (m))
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
extern "C" __global__ void llna_node(const u32 *S, const int *indptr, const int *indices, const int *seg_off,
                                     const int *seg_thr, const int *seg_cell, const u32 *one, const u32 *hasf,
                                     const u32 *frac, const u32 *stream, const u32 *cm, const u32 *cv, u32 *nxt,
                                     llna_idx n, llna_idx nw, u32 k0, u32 k1, u32 t) {
    llna_idx i = (llna_idx)blockIdx.x * blockDim.x + threadIdx.x;
    int lane = (int)(threadIdx.x & 31u);
    int hub = i < n && indptr[i + 1] - indptr[i] > LLNA_HUB;
    unsigned hubs = __ballot_sync(0xffffffffu, hub);
    WORD out = 0u;
    while (hubs) { /* the whole warp counts each hub; its own lane keeps the result */
        int h = __ffs(hubs) - 1;
        hubs &= hubs - 1u;
        WORD o = llna_update(S, indptr, indices, seg_off, seg_thr, seg_cell, one, hasf, frac, stream, cm, cv, 1,
                             k0, k1, t, (int)(i - lane + h), 0, lane, 32);
        if (lane == h) out = o;
    }
    if (i < n && !hub)
        out = llna_update(S, indptr, indices, seg_off, seg_thr, seg_cell, one, hasf, frac, stream, cm, cv, 1,
                          k0, k1, t, (int)i, 0, 0, 1);
    WORD v = out << ((i * LLNA_L) & 31);
    for (int sh = 1; sh < 32 / LLNA_L; sh <<= 1) v |= __shfl_xor_sync(0xffffffffu, v, sh);
    llna_idx wi = (i * LLNA_L) >> 5;
    if ((lane & (32 / LLNA_L - 1)) == 0 && wi < nw) nxt[wi] = v;
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

    return cp.RawModule(code=source, options=("-std=c++14",))


def run(graph, rules, x0, steps, record, clamp, seed, t0, noise_period, threads=None):
    import cupy as cp

    R, N = x0.n_replicas, graph.n
    Wp = tables.gpu_words(R)
    node = Wp == 1  # R <= 32: one thread per node, L bits per node
    L = tables.lanes(R) if node else 32
    tab = tables.build(graph, rules, x0, clamp, seed, noise_period, Wp, L)
    tab.defines["LLNA_COOP"] = 1
    source = tab.source(PRELUDE, WRAPPER)
    if os.environ.get("FAST_LLNA_DUMP"):
        with open(os.environ["FAST_LLNA_DUMP"], "w") as f:
            f.write(source)
    kernel = _kernel(source).get_function("llna_node" if node else "llna_step")
    fixed = [cp.asarray(tab.arrays[k]) for k in ARRAYS]
    cur = cp.asarray(tab.arrays["state"])
    nxt = cp.empty_like(cur)
    if node:
        grid, block, p1 = (-(-N // 256),), (256,), cur.size
    else:
        bx = min(Wp, 32)
        by = 256 // bx
        grid, block, p1 = ((N + by - 1) // by, Wp // bx), (bx, by), Wp

    rec = 0 if record == "final" else record
    frames = np.empty((steps // rec + 1 if rec else 1, frame_bytes(N, R)), np.uint8)
    frames[0] = x0.bits
    # keep recorded frames on the GPU when they fit comfortably, else copy each one to the host
    on_device = rec and frames.nbytes < 0.5 * cp.cuda.Device().mem_info[0]
    dframes = cp.empty(frames.shape, np.uint8) if on_device else None

    def frame(a, out):
        tables.to_frame(a.view(np.uint8), N, R, L * Wp, out)

    for s in range(steps):
        args = (
            cur,
            *fixed,
            nxt,
            np.int64(N),
            np.int64(p1),
            np.uint32(tab.k0),
            np.uint32(tab.k1),
            np.uint32((t0 + s) % 2**32),
        )
        kernel(grid, block, args)
        cur, nxt = nxt, cur
        if rec and (s + 1) % rec == 0:
            if on_device:
                frame(cur, dframes[(s + 1) // rec])
            else:
                frame(cp.asnumpy(cur), frames[(s + 1) // rec])
    if on_device:
        frames[1:] = cp.asnumpy(dframes[1:])
    if record == "final":
        frame(cp.asnumpy(cur), frames[0])
    return frames
