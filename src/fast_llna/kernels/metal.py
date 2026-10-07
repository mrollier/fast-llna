"""Apple GPU backend: llna.h as an MLX custom Metal kernel, one launch per step.

Thread (x, y) of the grid updates word x of node y, so a SIMD group covers one node and consecutive words.
"""

from __future__ import annotations

import hashlib
from functools import cache

import numpy as np

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
#define RANDW(k0, k1, t, i, st, w, d) llna_rand(k0, k1, t, (u32)(i), (st)[w], d)
"""

BODY = """
    uint w = thread_position_in_grid.x, i = thread_position_in_grid.y;
    uint n = params[0], Wp = params[1];
    if (w >= Wp || i >= n) return;
    nxt[(llna_idx)i * Wp + w] = llna_update(S, indptr, indices, seg_off, seg_thr, seg_cell, one, hasf, frac,
                                            stream, cm, cv, (llna_idx)Wp, params[2], params[3], params[4],
                                            (int)i, (llna_idx)w);
"""

INPUTS = (
    "S",
    "indptr",
    "indices",
    "seg_off",
    "seg_thr",
    "seg_cell",
    "one",
    "hasf",
    "frac",
    "stream",
    "cm",
    "cv",
    "params",
)
EVAL_EVERY = 32


def available() -> bool:
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return mx.metal.is_available()


@cache
def _kernel(header: str):
    import mlx.core as mx

    name = "llna_" + hashlib.sha256((header + BODY).encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(
        name=name, input_names=list(INPUTS), output_names=["nxt"], source=BODY, header=header
    )


def _pad(a):
    """MLX passes inputs of fewer than 8 elements in the constant address space; keep every buffer a device
    pointer by padding to 16 elements."""
    return np.pad(a, (0, max(0, 16 - a.size)))


def run(graph, rules, x0, steps, record, clamp, seed, t0, stream, threads=None):
    import mlx.core as mx

    R, N = x0.n_replicas, graph.n
    Wp = tables.gpu_words(R)
    tab = tables.build(graph, rules, x0, clamp, seed, stream, Wp)
    kernel = _kernel(tab.source(PRELUDE, ""))
    fixed = [mx.array(_pad(tab.arrays[k])) for k in INPUTS[1:-1]]
    bx = min(Wp, 32)
    grid, group = (Wp, N, 1), (bx, max(1, min(256 // bx, N)), 1)

    nb = (R + 7) // 8
    rec = 0 if record == "final" else record
    frames = np.empty((steps // rec + 1 if rec else 1, N, nb), np.uint8)
    frames[0] = x0.bits
    pending = []  # (frame index, lazy mx array)

    def flush():
        mx.eval(cur, *(a for _, a in pending))
        for f, a in pending:
            frames[f] = np.asarray(a)[: N * Wp].view(np.uint8).reshape(N, 4 * Wp)[:, :nb]
        pending.clear()

    cur = mx.array(_pad(tab.arrays["state"].ravel()))
    size = cur.size
    for s in range(steps):
        params = mx.array(np.array([N, Wp, tab.k0, tab.k1, (t0 + s) % 2**32], np.uint32))
        (cur,) = kernel(
            inputs=[cur, *fixed, params],
            grid=grid,
            threadgroup=group,
            output_shapes=[(size,)],
            output_dtypes=[mx.uint32],
        )
        if rec and (s + 1) % rec == 0:
            pending.append(((s + 1) // rec, cur))
        if (s + 1) % EVAL_EVERY == 0:
            flush()
    flush()
    if record == "final":
        frames[0] = np.asarray(cur)[: N * Wp].view(np.uint8).reshape(N, 4 * Wp)[:, :nb]
    return frames
