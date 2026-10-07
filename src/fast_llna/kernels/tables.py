"""Host-side preparation shared by the compiled backends: everything the kernel reads, as flat numpy arrays.

Kernel state layout: node-major fields of Lk = L * Wp bits, where L = lanes(R) and Wp is the host's padded
word count per node; replica r of node i is bit i * Lk + r. For L < 32 (R <= 16) this is the narrow layout
of several nodes per word; for L == 32 it is the word layout u32 [N][Wp]. Padding replicas have all-zero
masks and stay 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..rng import ONE, noise_streams, quantize, split_seed
from ..states import bits_per_node, pack

HEADER = (Path(__file__).parent / "llna.h").read_text()
PMAX_BUCKETS = (4, 8, 12, 16, 24, 32)
ARRAYS = ("indptr", "indices", "seg_off", "seg_thr", "seg_cell", "one", "hasf", "frac", "stream", "cm", "cv")


def gpu_words(R: int) -> int:
    """Padded words per node on GPUs: a power of two up to one warp, then whole warps."""
    W = (R + 31) // 32
    return 1 << (W - 1).bit_length() if W <= 32 else -(-W // 32) * 32


def lanes(R: int) -> int:
    """Bits per node per word in the kernel state: pow2ceil(R) up to 32."""
    return min(32, 1 << (R - 1).bit_length())


def hub_threshold(graph) -> int:
    """GPU node kernels count nodes of degree above this cooperatively (a whole SIMD group per node)."""
    return max(32, 4 * int(np.ceil(graph.degree.mean())))


def _words(b: np.ndarray, n_words: int) -> np.ndarray:
    """uint8 bytes -> uint32 words, zero-padded to n_words."""
    return np.pad(b, (0, 4 * n_words - b.size)).view("<u4")


def to_kernel(x0, Lk: int) -> np.ndarray:
    """States -> kernel state words with Lk bits per node (node-major), ceil(N * Lk / 32) words."""
    N, Ls = x0.n, bits_per_node(x0.n_replicas)
    if Ls == Lk:
        b = x0.bits
    else:  # whole bytes per node: pad each node row
        assert Ls % 8 == 0 and Lk % 8 == 0, (Ls, Lk)
        b = np.pad(x0.bits.reshape(N, Ls // 8), ((0, 0), (0, (Lk - Ls) // 8))).ravel()
    return _words(b, -(-N * Lk // 32))


def to_frame(state, n: int, R: int, Lk: int, out) -> None:
    """Write kernel state bytes (Lk bits per node, trailing padding allowed) into ``out``, one contiguous frame
    in the States layout, without temporaries. Works on numpy and cupy arrays."""
    Ls = bits_per_node(R)
    if Ls == Lk:
        out[...] = state[: out.size]
    else:
        assert Ls % 8 == 0 and Lk % 8 == 0, (Ls, Lk)
        out.reshape(n, Ls // 8)[...] = state[: n * Lk // 8].reshape(n, Lk // 8)[:, : Ls // 8]


@dataclass
class Tables:
    defines: dict
    Lk: int  # kernel bits per node
    arrays: dict = field(default_factory=dict)
    k0: np.uint32 = np.uint32(0)
    k1: np.uint32 = np.uint32(0)

    def source(self, prelude: str, wrapper: str) -> str:
        defs = "".join(f"#define {k} {v}\n" for k, v in sorted(self.defines.items()))
        return defs + prelude + HEADER + wrapper


def build(graph, rules, x0, clamp, seed, noise_period, Wp) -> Tables:
    R, N = x0.n_replicas, graph.n
    L = lanes(R)
    Lk = L * Wp
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
    one = P == ONE
    hasf = (P > 0) & ~one
    digits = np.arange(depth, dtype=np.uint64)
    frac = ((P[..., None] >> (np.uint64(31) - digits)) & np.uint64(1)).astype(bool) & hasf[..., None]

    def words(m):  # [R, ...] -> flat words, Wp per trailing index, replica r in bit r of word r // 32
        return pack(m.reshape(R, -1), 32 * Wp).view("<u4")

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
        },
        Lk=Lk,
    )
    t.k0, t.k1 = split_seed(seed)
    a = t.arrays
    a["indptr"], a["indices"] = graph.indptr, graph.indices
    a["seg_off"], a["seg_thr"] = seg_off, np.asarray(thr or [0], np.int32)
    a["seg_cell"] = np.asarray(cells or [0, 0], np.int32)
    a["one"], a["hasf"] = words(one), words(hasf)
    a["frac"] = words(frac) if depth else np.zeros(1, np.uint32)
    a["stream"] = noise_streams(Wp, noise_period)
    if clamp is not None:
        n_words = -(-N * Lk // 32)
        a["cm"] = _words(pack(clamp[0], Lk), n_words)
        a["cv"] = _words(pack(clamp[0] & clamp[1], Lk), n_words)
    else:
        a["cm"] = a["cv"] = np.zeros(1, np.uint32)
    a["state"] = to_kernel(x0, Lk)  # flat, ceil(N * Lk / 32) words
    return t
