"""Apple GPU backend: llna.h as MLX custom Metal kernels, one launch per step.

R <= 32: one thread per node (llna_group in llna.h: a SIMD group counts each hub together and assembles 32/L
nodes per output word with XOR shuffles). R > 32: thread (x, y) updates replica word x of node y.
"""

from __future__ import annotations

import hashlib
from functools import cache

import numpy as np

from ..states import frame_bytes
from . import tables

PRELUDE = """
typedef uint u32;
typedef long llna_idx;
typedef u32 WORD;
#define LLNA_FN inline
#define LLNA_PTR(T) device const T *
#define MULHI(a, b) metal::mulhi((u32)(a), (u32)(b))
#define ZEROW 0u
#define LOADW(ptr, off) ((ptr)[off])
#define ANYW(x) ((x) != 0u)
#define LLNA_LANE0(x) (x)
#define RANDW(k0, k1, t, i, st, w, d) llna_rand(k0, k1, t, (u32)(i), (st)[w], d)
#define LLNA_SHFL_XOR(x, m) simd_shuffle_xor((x), (ushort)(m))
#define LLNA_BALLOT(b) ((u32)(simd_vote::vote_t)simd_ballot((bool)(b)))
#define LLNA_CTZ(x) ((int)ctz(x))
"""

# params: [n, Wp or word count, k0, k1, t, hub threshold]
BODY = """
    uint w = thread_position_in_grid.x, i = thread_position_in_grid.y;
    uint n = params[0], Wp = params[1];
    if (w >= Wp || i >= n) return;
    nxt[(llna_idx)i * Wp + w] = llna_update(S, indptr, indices, seg_off, seg_thr, seg_cell, one, hasf, frac,
                                            stream, cm, cv, (llna_idx)Wp, params[2], params[3], params[4],
                                            (int)i, (llna_idx)w, 0, 1);
"""

NODE_BODY = """
    uint i = thread_position_in_grid.x, lane = thread_index_in_simdgroup;
    WORD v = llna_group(S, indptr, indices, seg_off, seg_thr, seg_cell, one, hasf, frac, stream, cm, cv,
                        params[2], params[3], params[4], (int)params[0], (int)params[5], (int)i, (int)lane);
    ulong wi = ((ulong)i * LLNA_L) >> 5;
    if ((lane & (32u / LLNA_L - 1u)) == 0u && wi < params[1]) nxt[wi] = v;
"""

INPUTS = ("S", *tables.ARRAYS, "params")
EVAL_EVERY = 32


def available() -> bool:
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return mx.metal.is_available()


@cache
def _kernel(header: str, body: str):
    import mlx.core as mx

    name = "llna_" + hashlib.sha256((header + body).encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(
        name=name, input_names=list(INPUTS), output_names=["nxt"], source=body, header=header
    )


def _pad(a):
    """MLX passes inputs of fewer than 8 elements in the constant address space; keep every buffer a device
    pointer by padding to 16 elements."""
    return np.pad(a, (0, max(0, 16 - a.size)))


def run(graph, rules, x0, steps, record, clamp, seed, t0, noise_period, threads=None):
    import mlx.core as mx

    R, N = x0.n_replicas, graph.n
    Wp = tables.gpu_words(R)
    node = Wp == 1  # R <= 32: one thread per node, L bits per node
    tab = tables.build(graph, rules, x0, clamp, seed, noise_period, Wp)
    tab.defines["LLNA_COOP"] = 1
    kernel = _kernel(tab.source(PRELUDE, ""), NODE_BODY if node else BODY)
    fixed = [mx.array(_pad(tab.arrays[k])) for k in tables.ARRAYS]
    cur = mx.array(_pad(tab.arrays["state"]))
    if node:
        grid, group, p1 = (-(-N // 256) * 256, 1, 1), (256, 1, 1), tab.arrays["state"].size
    else:
        bx = min(Wp, 32)
        grid, group, p1 = (Wp, N, 1), (bx, max(1, min(256 // bx, N)), 1), Wp
    hub = tables.hub_threshold(graph)

    rec = 0 if record == "final" else record
    frames = np.empty((steps // rec + 1 if rec else 1, frame_bytes(N, R)), np.uint8)
    frames[0] = x0.bits
    pending = []  # (frame index, lazy mx array)

    def frame(a, f):
        tables.to_frame(np.asarray(a).view(np.uint8), N, R, tab.Lk, frames[f])

    def flush():
        mx.eval(cur, *(a for _, a in pending))
        for f, a in pending:
            frame(a, f)
        pending.clear()

    for s in range(steps):
        params = mx.array(np.array([N, p1, tab.k0, tab.k1, t0 + s, hub], np.uint32))
        (cur,) = kernel(
            inputs=[cur, *fixed, params],
            grid=grid,
            threadgroup=group,
            output_shapes=[cur.shape],
            output_dtypes=[mx.uint32],
        )
        if rec and (s + 1) % rec == 0:
            pending.append(((s + 1) // rec, cur))
        if (s + 1) % EVAL_EVERY == 0:
            flush()
    flush()
    if record == "final":
        frame(cur, 0)
    return frames
