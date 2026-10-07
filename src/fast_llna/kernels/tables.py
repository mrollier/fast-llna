"""Host-side preparation shared by the compiled backends: everything the kernel reads, as flat numpy arrays.

Word layout: uint32 ``[N][Wp]``; replica r is bit r % 32 of word r // 32. Padding replicas have all-zero
masks and stay 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..rng import quantize, split_seed

HEADER = (Path(__file__).parent / "llna.h").read_text()
PMAX_BUCKETS = (4, 8, 12, 16, 24, 32)


def gpu_words(R: int) -> int:
    """Padded words per node on GPUs: a power of two up to one warp, then whole warps."""
    W = (R + 31) // 32
    return 1 << (W - 1).bit_length() if W <= 32 else -(-W // 32) * 32


def _pack_words(x: np.ndarray, Wp: int) -> np.ndarray:
    """bool [..., R] -> uint32 [..., Wp] (replica r -> bit r % 32 of word r // 32)."""
    b = np.packbits(x, axis=-1, bitorder="little")
    out = np.zeros(x.shape[:-1] + (4 * Wp,), np.uint8)
    out[..., : b.shape[-1]] = b
    return out.view("<u4")


@dataclass
class Tables:
    defines: dict
    arrays: dict = field(default_factory=dict)
    k0: np.uint32 = np.uint32(0)
    k1: np.uint32 = np.uint32(0)

    def source(self, prelude: str, wrapper: str) -> str:
        defs = "".join(f"#define {k} {v}\n" for k, v in sorted(self.defines.items()))
        return defs + prelude + HEADER + wrapper


def build(graph, rules, x0, clamp, seed, stream, Wp) -> Tables:
    R, N = x0.n_replicas, graph.n
    deg = graph.degree
    kmax = int(deg.max())
    part = rules.partition

    # segments of q in which (cell for dead, cell for alive) is constant, per degree present
    seg_off = np.zeros(kmax + 2, np.int32)
    thr, cells = [], []
    present = np.zeros(kmax + 1, bool)
    present[deg] = True
    for k in range(kmax + 1):
        seg_off[k] = len(thr)
        if not present[k]:
            continue
        q = np.arange(k + 1)
        cB, cS = part.cell(q, np.full_like(q, k), 0), part.cell(q, np.full_like(q, k), 1)
        start = np.r_[0, np.flatnonzero((cB[1:] != cB[:-1]) | (cS[1:] != cS[:-1])) + 1]
        thr += q[start].tolist()
        cells += np.c_[cB[start], cS[start]].ravel().tolist()
    seg_off[kmax + 1] = len(thr)
    nseg = int(np.diff(seg_off).max())

    # per-replica masks: ONE (p == 1), HASF (0 < p < 1), FRAC digit planes of p (MSB first)
    P, depth = quantize(rules.p)
    P = np.broadcast_to(P, (R, 2, part.ncell)) if len(rules) == 1 else P
    one = P == np.uint64(2**32)
    hasf = (P > 0) & ~one
    digits = np.arange(depth, dtype=np.uint64)
    frac = ((P[..., None] >> (np.uint64(31) - digits)) & np.uint64(1)).astype(bool) & hasf[..., None]

    def lanes(m):  # [R, ...] -> [..., Wp] words
        return _pack_words(np.moveaxis(m, 0, -1), Wp)

    pmax = next(b for b in PMAX_BUCKETS if b >= kmax.bit_length())
    t = Tables(
        defines={
            "PMAX": pmax,
            "SEGMAX": 1 << (nseg - 1).bit_length() if nseg > 1 else 1,
            "NCELL": part.ncell,
            "LLNA_D": depth,
            "LLNA_CLAMP": int(clamp is not None),
        }
    )
    t.k0, t.k1 = split_seed(seed)
    a = t.arrays
    a["indptr"], a["indices"] = graph.indptr, graph.indices
    a["seg_off"], a["seg_thr"] = seg_off, np.asarray(thr or [0], np.int32)
    a["seg_cell"] = np.asarray(cells or [0, 0], np.int32)
    a["one"], a["hasf"] = lanes(one).ravel(), lanes(hasf).ravel()
    a["frac"] = lanes(frac).ravel() if depth else np.zeros(1, np.uint32)
    st = np.zeros(Wp, np.uint32)
    st[: len(stream)] = stream
    a["stream"] = st
    if clamp is not None:
        a["cm"] = _pack_words(clamp[0].T, Wp).ravel()
        a["cv"] = _pack_words((clamp[0] & clamp[1]).T, Wp).ravel()
    else:
        a["cm"] = a["cv"] = np.zeros(1, np.uint32)
    state = np.zeros((N, 4 * Wp), np.uint8)  # canonical bytes are already the word layout
    state[:, : x0.bits.shape[1]] = x0.bits
    a["state"] = state.view("<u4")  # [N, Wp]
    return t
