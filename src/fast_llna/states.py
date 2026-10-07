"""Bit-packed configurations of R replicas and trajectories of them.

Layout: a configuration is a flat little-endian bit array, uint8 ``bits[..., ceil(N * L / 8)]``, in which
replica r of node i is bit i * L + r. L = bits_per_node(R) is the next power of two (1, 2, 4, 8) for R <= 8,
and whole bytes, 8 * ceil(R / 8), beyond. Padding bits are 0. For R >= 8 these are the bytes of the rows
[N, ceil(R / 8)].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rules import Rules

_CHUNK = 1 << 26  # unpacked bytes per reducer chunk


def bits_per_node(R: int) -> int:
    return 1 << (R - 1).bit_length() if R <= 8 else 8 * -(-R // 8)


def frame_bytes(n: int, R: int) -> int:
    return -(-n * bits_per_node(R) // 8)


def pack(x, L: int) -> np.ndarray:
    """bool [..., R, N] -> uint8 [..., ceil(N * L / 8)] with replica r of node i at bit i * L + r (R <= L)."""
    x = np.swapaxes(np.asarray(x, bool), -1, -2)  # [..., N, R]
    *lead, n, R = x.shape
    if L == 8 * -(-R // 8):  # whole bytes per node: packbits pads each node row itself
        return np.packbits(x, axis=-1, bitorder="little").reshape(*lead, -1)
    x = np.concatenate([x, np.zeros((*lead, n, L - R), bool)], axis=-1)
    return np.packbits(x.reshape(*lead, n * L), axis=-1, bitorder="little")


def unpack(bits, n: int, R: int, L: int) -> np.ndarray:
    """Inverse of :func:`pack`: uint8 [..., ceil(n * L / 8)] -> bool [..., R, n]."""
    x = np.unpackbits(bits, axis=-1, count=n * L, bitorder="little")
    return np.swapaxes(x.reshape(*bits.shape[:-1], n, L)[..., :R], -1, -2).view(bool)


@dataclass(frozen=True, eq=False)
class States:
    bits: np.ndarray
    n_replicas: int
    n: int

    def __post_init__(self):
        if self.n_replicas < 1 or self.n < 1:
            raise ValueError(f"need at least one replica and one node, got R={self.n_replicas}, N={self.n}")
        want = frame_bytes(self.n, self.n_replicas)
        if np.ndim(self.bits) == 0 or np.shape(self.bits)[-1] != want:
            raise ValueError(
                f"bits of shape {np.shape(self.bits)} do not end in the {want} bytes that {self.n} nodes x "
                f"{self.n_replicas} replicas need"
            )

    @classmethod
    def from_bool(cls, x) -> States:
        """From a boolean array ``[..., R, N]``."""
        x = np.asarray(x, dtype=bool)
        if x.ndim < 2:
            raise ValueError(f"expected a boolean array [R, N] (or [..., R, N]), got shape {x.shape}")
        return cls(pack(x, bits_per_node(x.shape[-2])), x.shape[-2], x.shape[-1])

    def to_bool(self) -> np.ndarray:
        """Boolean array ``[..., R, N]``."""
        return unpack(self.bits, self.n, self.n_replicas, bits_per_node(self.n_replicas))


def random_states(n: int, replicas: int, density: float = 0.5, seed=None) -> States:
    """Independent configurations with exactly ``round(density * n)`` living nodes each."""
    if not 0 <= density <= 1:
        raise ValueError(f"density must lie in [0, 1], got {density}")
    rng = np.random.default_rng(seed)
    alive = np.arange(n) < round(density * n)
    return States.from_bool(rng.permuted(np.broadcast_to(alive, (replicas, n)), axis=1))


def defect_twins(states: States, flips: int = 1, seed=None) -> tuple[States, np.ndarray]:
    """Append to R replicas a twin each with ``flips`` distinct random nodes toggled. Twin of replica r is
    replica r + R; returns the 2R states and the [R, 2] index pairs. For stochastic rules pass
    ``noise_period=R`` to ``simulate`` so twins share their random draws (R must divide 32 or be a
    multiple of 32)."""
    rng = np.random.default_rng(seed)
    x = states.to_bool()
    R, n = x.shape
    if not 0 <= flips <= n:
        raise ValueError(f"flips must lie in [0, {n}], got {flips}")
    toggled = np.stack([rng.choice(n, flips, replace=False) for _ in range(R)])  # [R, flips]
    twins = x.copy()
    twins[np.arange(R)[:, None], toggled] ^= True
    return States.from_bool(np.concatenate([x, twins])), np.c_[np.arange(R), np.arange(R, 2 * R)]


def product(rules: Rules, states: States) -> tuple[Rules, States]:
    """All rule x configuration combinations; replica k * C + c runs rule k from configuration c."""
    K, C = len(rules), states.n_replicas
    return rules.take(np.repeat(np.arange(K), C)), States.from_bool(np.tile(states.to_bool(), (K, 1)))


@dataclass(frozen=True, eq=False)
class Trajectory:
    """Recorded configurations ``states.bits[T', frame bytes]`` at timesteps ``times``."""

    states: States
    times: np.ndarray

    def _chunks(self):
        """The trajectory unpacked to bool [t, R, N], a few frames at a time."""
        s = self.states
        L = bits_per_node(s.n_replicas)
        step = max(1, _CHUNK // max(1, s.n * L))
        for t in range(0, s.bits.shape[0], step):
            yield unpack(s.bits[t : t + step], s.n, s.n_replicas, L)

    def density(self) -> np.ndarray:
        """Fraction of living nodes, ``[T', R]``."""
        return np.concatenate([x.mean(axis=-1) for x in self._chunks()])

    def hamming(self, pairs) -> np.ndarray:
        """Normalised Hamming distance between replicas ``pairs[:, 0]`` and ``pairs[:, 1]``, ``[T', P]``."""
        a, b = np.atleast_2d(pairs).T
        return np.concatenate([(x[:, a] ^ x[:, b]).mean(axis=-1) for x in self._chunks()])

    def final(self) -> States:
        return States(self.states.bits[-1], self.states.n_replicas, self.states.n)
