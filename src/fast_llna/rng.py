"""Counter-based randomness shared bit-for-bit by every backend.

Contract (the C kernel implements the same):

    word_d(t, i, w) = Philox4x32-10(ctr=(t, i, stream[w], purpose << 16 | d >> 2), key=(seed_lo, seed_hi))[d & 3]

t is the timestep being left (t -> t+1), i the node, w the 32-replica word, purpose 0 the rule outputs (1 and
2 are reserved for alpha-asynchronous updating and output noise). Without shared noise, replica r reads bit
r % 32 of word_d(t, i, r // 32), d = 0, 1, ...; these bits are the
binary digits of a uniform U = sum_d bit_d 2^-(d+1), most significant first. A node becomes alive with
probability p iff U < p, where p is rounded to a multiple of 2^-32. Because p = P / 2^depth is decided by
the first ``depth`` digits, a replica's outcome does not depend on the other rules in the batch.

Shared noise: with ``noise_period`` p, replica r reads exactly the digits of replica r' = r mod p, i.e. bit
r' % 32 of word_d(t, i, r' // 32). p must divide 32 or be a multiple of 32. Kernels implement multiples of 32
with stream[w] = w mod (p / 32), and divisors of 32 with stream[w] = 0 plus repeating the low p bits of each
digit word across the word.
"""

from __future__ import annotations

import warnings

import numpy as np

_M0, _M1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
_W0, _W1 = np.uint32(0x9E3779B9), np.uint32(0xBB67AE85)
_LO = np.uint64(0xFFFFFFFF)
ONE = np.uint64(2**32)  # the quantized probability 1
DEPTHS = (0, 1, 2, 4, 8, 16, 32)


def philox4x32(c0, c1, c2, c3, k0, k1):
    """Philox4x32-10 on uint32 arrays (broadcast); returns four uint32 arrays."""
    c0, c1, c2, c3 = (np.asarray(c, np.uint32) for c in (c0, c1, c2, c3))
    k0, k1 = np.asarray(k0, np.uint32), np.asarray(k1, np.uint32)
    with np.errstate(over="ignore"):
        for rnd in range(10):
            if rnd:
                k0, k1 = k0 + _W0, k1 + _W1
            p0, p1 = _M0 * c0.astype(np.uint64), _M1 * c2.astype(np.uint64)
            hi0, lo0 = (p0 >> np.uint64(32)).astype(np.uint32), (p0 & _LO).astype(np.uint32)
            hi1, lo1 = (p1 >> np.uint64(32)).astype(np.uint32), (p1 & _LO).astype(np.uint32)
            c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return c0, c1, c2, c3


def split_seed(seed: int) -> tuple[np.uint32, np.uint32]:
    seed = int(seed) % 2**64
    return np.uint32(seed & 0xFFFFFFFF), np.uint32(seed >> 32)


def noise_streams(n_words: int, noise_period: int | None) -> np.ndarray:
    """stream[w] of each 32-replica word w (see the module contract); periods below 32 use stream 0."""
    w = np.arange(n_words, dtype=np.uint32)
    return w if noise_period is None else w % np.uint32(max(1, noise_period // 32))


def rule_words(seed: int, t, i, stream, d: int, purpose: int = 0):
    """Random word d of (t, i, stream) for the given purpose, per the module contract."""
    k0, k1 = split_seed(seed)
    out = philox4x32(t, i, stream, np.uint32(purpose << 16 | d >> 2), k0, k1)
    return out[d & 3]


def quantize(p) -> tuple[np.ndarray, int]:
    """Round probabilities to multiples of 2^-32: returns P = round(p * 2^32) (uint64; P = 2^32 means p = 1)
    and the number of random digits needed by the fractional ones, bucketed to one of DEPTHS."""
    p = np.asarray(p, np.float64)
    P = np.rint(p * 2.0**32).astype(np.uint64)
    if np.any((p > 0) & (P == 0)) or np.any((p < 1) & (P == ONE)):
        warnings.warn("a probability rounds to 0 or 1 at 2^-32 resolution", stacklevel=2)
    frac = P[(P > 0) & (P < ONE)]
    if frac.size == 0:
        return P, 0
    v = int(np.bitwise_or.reduce(frac))  # its trailing zeros are the fewest of any fractional P
    depth = 32 - ((v & -v).bit_length() - 1)
    return P, next(b for b in DEPTHS if b >= depth)
