"""Bit-packed configurations of R replicas and trajectories of them.

Canonical layout: uint8 ``bits[..., N, ceil(R/8)]`` where replica r of node i is bit r % 8 (little bit order)
of byte r // 8 of row i, with padding bits 0. On little-endian machines this is byte-for-byte the uint32 word
layout ``[N][W]`` used by the kernels (replica r = bit r % 32 of word r // 32).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rules import Rules

_CHUNK = 1 << 26  # unpacked bytes per reducer chunk


@dataclass(frozen=True, eq=False)
class States:
    bits: np.ndarray
    n_replicas: int

    @classmethod
    def from_bool(cls, x) -> States:
        """From a boolean array ``[..., R, N]``."""
        x = np.asarray(x, dtype=bool)
        return cls(np.packbits(np.swapaxes(x, -1, -2), axis=-1, bitorder="little"), x.shape[-2])

    def to_bool(self) -> np.ndarray:
        """Boolean array ``[..., R, N]``."""
        x = np.unpackbits(self.bits, axis=-1, count=self.n_replicas, bitorder="little")
        return np.swapaxes(x, -1, -2).astype(bool)

    @property
    def n(self) -> int:
        return self.bits.shape[-2]


def random_states(n: int, replicas: int, density: float = 0.5, seed=None) -> States:
    """Independent configurations with exactly ``round(density * n)`` living nodes each."""
    rng = np.random.default_rng(seed)
    alive = round(density * n)
    ranks = rng.random((replicas, n)).argsort(axis=1).argsort(axis=1)
    return States.from_bool(ranks < alive)


def defect_twins(states: States, flips: int = 1, seed=None) -> tuple[States, np.ndarray]:
    """Append to R replicas a twin each with ``flips`` distinct random nodes toggled. Twin of replica r is
    replica r + R; returns the 2R states and the [R, 2] index pairs. For stochastic rules pass
    ``noise_period=R`` to ``simulate`` so twins share their random draws (R must divide 32 or be a
    multiple of 32)."""
    rng = np.random.default_rng(seed)
    x = states.to_bool()
    R, n = x.shape
    toggled = rng.random((R, n)).argsort(axis=1)[:, :flips]
    twins = x.copy()
    np.logical_xor.at(twins, (np.arange(R)[:, None], toggled), True)
    return States.from_bool(np.concatenate([x, twins])), np.c_[np.arange(R), np.arange(R, 2 * R)]


def product(rules: Rules, states: States) -> tuple[Rules, States]:
    """All rule x configuration combinations; replica k * L + l runs rule k from configuration l."""
    K, L = len(rules), states.n_replicas
    return rules.take(np.repeat(np.arange(K), L)), States.from_bool(np.tile(states.to_bool(), (K, 1)))


@dataclass(frozen=True, eq=False)
class Trajectory:
    """Recorded configurations ``states.bits[T', N, ceil(R/8)]`` at timesteps ``times``."""

    states: States
    times: np.ndarray

    def _chunks(self):
        bits = self.states.bits
        step = max(1, _CHUNK // max(1, bits.shape[-2] * self.states.n_replicas))
        for t in range(0, bits.shape[0], step):
            yield np.unpackbits(bits[t : t + step], axis=-1, count=self.states.n_replicas, bitorder="little")

    def density(self) -> np.ndarray:
        """Fraction of living nodes, ``[T', R]``."""
        return np.concatenate([x.mean(axis=1) for x in self._chunks()])

    def hamming(self, pairs) -> np.ndarray:
        """Normalised Hamming distance between replicas ``pairs[:, 0]`` and ``pairs[:, 1]``, ``[T', P]``."""
        a, b = np.asarray(pairs).T
        return np.concatenate([(x[..., a] ^ x[..., b]).mean(axis=1) for x in self._chunks()])

    def final(self) -> States:
        return States(self.states.bits[-1], self.states.n_replicas)
