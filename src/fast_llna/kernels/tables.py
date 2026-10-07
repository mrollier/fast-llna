"""Host-side preparation shared by the compiled backends: everything the kernel reads, as flat numpy arrays.

Word layout: uint32 ``[N][Wp]``; replica r is bit r % 32 of word r // 32. Padding replicas have all-zero
masks and stay 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..rng import noise_streams, quantize, split_seed
from ..states import bits_per_node, frame_bytes, pack, unpack

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


def _words(b: np.ndarray, n_words: int) -> np.ndarray:
    """uint8 bytes -> uint32 words, zero-padded to n_words."""
    return np.pad(b, (0, 4 * n_words - b.size)).view("<u4")


def to_kernel(x0, Lk: int) -> np.ndarray:
    """States -> kernel state words with Lk bits per node (node-major), ceil(N * Lk / 32) words."""
    N, Ls = x0.n, bits_per_node(x0.n_replicas)
    if Ls == Lk:
        b = x0.bits
    elif Ls % 8 == 0 and Lk % 8 == 0:  # whole bytes per node: pad each node row
        b = np.pad(x0.bits.reshape(N, Ls // 8), ((0, 0), (0, (Lk - Ls) // 8))).ravel()
    else:
        b = pack(x0.to_bool(), Lk)
    return _words(b, -(-N * Lk // 32))


def to_frame(state, n: int, R: int, Lk: int):
    """Kernel state bytes (Lk bits per node, trailing padding allowed) -> one frame in the States layout.
    The first two cases also work on cupy arrays."""
    Ls = bits_per_node(R)
    if Ls == Lk:
        return state[: frame_bytes(n, R)]
    if Ls % 8 == 0 and Lk % 8 == 0:
        return state[: n * Lk // 8].reshape(n, Lk // 8)[:, : Ls // 8].reshape(-1)
    return pack(unpack(state[: -(-n * Lk // 8)], n, R, Lk), Ls)


def rows_to_frames(rows: np.ndarray, R: int) -> np.ndarray:
    """v1 rows [F, N, ceil(R / 8)] -> frames [F, frame bytes]: a free reshape for R >= 5."""
    F, n, nb = rows.shape
    if bits_per_node(R) == 8 * nb:
        return rows.reshape(F, n * nb)
    x = np.unpackbits(rows, axis=-1, count=R, bitorder="little").swapaxes(-1, -2).astype(bool)
    return pack(x, bits_per_node(R))


def lanes(R: int) -> int:
    """Bits per node in the kernel state for R <= 32 replicas."""
    return min(32, 1 << (R - 1).bit_length())


@dataclass
class Tables:
    defines: dict
    arrays: dict = field(default_factory=dict)
    k0: np.uint32 = np.uint32(0)
    k1: np.uint32 = np.uint32(0)

    def source(self, prelude: str, wrapper: str) -> str:
        defs = "".join(f"#define {k} {v}\n" for k, v in sorted(self.defines.items()))
        return defs + prelude + HEADER + wrapper


def build(graph, rules, x0, clamp, seed, noise_period, Wp, L=32) -> Tables:
    R, N = x0.n_replicas, graph.n
    Lk = L * Wp  # kernel bits per node
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
    period = noise_period if noise_period is not None and noise_period < 32 else 32
    t = Tables(
        defines={
            "PMAX": pmax,
            "SEGMAX": 1 << (nseg - 1).bit_length() if nseg > 1 else 1,
            "NCELL": part.ncell,
            "LLNA_D": depth,
            "LLNA_CLAMP": int(clamp is not None),
            "LLNA_NP": period,
            "LLNA_REPL": f"{sum(1 << s for s in range(0, 32, period)):#x}u",
            "LLNA_L": L,
            "LLNA_LMASK": f"{(1 << L) - 1:#x}u",
            "LLNA_COOP": 0,
            "LLNA_HUB": max(32, 4 * int(np.ceil(deg.mean()))),
        }
    )
    t.k0, t.k1 = split_seed(seed)
    a = t.arrays
    a["indptr"], a["indices"] = graph.indptr, graph.indices
    a["seg_off"], a["seg_thr"] = seg_off, np.asarray(thr or [0], np.int32)
    a["seg_cell"] = np.asarray(cells or [0, 0], np.int32)
    a["one"], a["hasf"] = lanes(one).ravel(), lanes(hasf).ravel()
    a["frac"] = lanes(frac).ravel() if depth else np.zeros(1, np.uint32)
    a["stream"] = noise_streams(Wp, noise_period)
    if clamp is not None:
        n_words = -(-N * Lk // 32)
        a["cm"] = _words(pack(clamp[0], Lk), n_words)
        a["cv"] = _words(pack(clamp[0] & clamp[1], Lk), n_words)
    else:
        a["cm"] = a["cv"] = np.zeros(1, np.uint32)
    a["state"] = to_kernel(x0, Lk)  # flat, ceil(N * Lk / 32) words
    return t
