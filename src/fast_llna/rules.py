"""Local update rules of Life-like network automata, in exact integer arithmetic.

A node with state s, in-degree k and q living in-neighbours has neighbourhood density rho = q / k. A
:class:`Partition` assigns rho to one of ``ncell`` cells; a :class:`Rules` table gives, per cell and per
own state, the probability that the node is alive at the next timestep (0 or 1 for deterministic rules).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Partition:
    """A partition of the density interval [0, 1] into ``ncell`` cells.

    kind
        ``"symmetric"``: the complementation-symmetric partition of the thesis (ch05, app03). Odd r has one
        closed central interval. Even r uses two partitions R+ and R- that differ only at rho = 1/2;
        ``even="+-"`` lets dead nodes (born set B) use R+ and living nodes (survive set S) use R-,
        ``even="-+"`` swaps them.
        ``"uniform"``: the legacy left-closed partition [j/r, (j+1)/r[ with rho = 1 in the last cell.
        ``"majority"``: three cells rho < 1/2, rho = 1/2, rho > 1/2.
    """

    kind: str
    r: int
    even: str = "+-"

    def __post_init__(self):
        if self.kind not in ("symmetric", "uniform", "majority") or self.r < 1:
            raise ValueError(f"invalid partition {self}")
        if self.kind == "majority" and self.r != 3:
            raise ValueError("the majority partition has exactly 3 cells")
        if self.even not in ("+-", "-+"):
            raise ValueError("even must be '+-' or '-+'")
        if self.kind != "symmetric" or self.r % 2:  # the convention only matters for even symmetric r
            object.__setattr__(self, "even", "+-")

    @property
    def ncell(self) -> int:
        return self.r

    def cell(self, q, k, s) -> np.ndarray:
        """Cell index of density q/k for a node in state s (exact; q, k integer arrays, k >= 1)."""
        q, k = np.asarray(q, dtype=np.int64), np.asarray(k, dtype=np.int64)
        r = self.r
        if self.kind == "uniform":
            return np.minimum(q * r // k, r - 1)
        if self.kind == "majority":
            return np.sign(2 * q - k) + 1
        if r % 2:
            mid = np.full_like(q, (r - 1) // 2)
        else:  # R+ puts rho = 1/2 in cell r/2 - 1, R- in cell r/2
            plus = (np.asarray(s) == 0) == (self.even == "+-")
            mid = np.where(plus, r // 2 - 1, r // 2)
        return np.where(2 * q < k, q * r // k, np.where(2 * q > k, r - 1 - (k - q) * r // k, mid))


def symmetric(r: int, even: str = "+-") -> Partition:
    return Partition("symmetric", r, even)


def uniform(r: int) -> Partition:
    return Partition("uniform", r)


MAJORITY = Partition("majority", 3)


@dataclass(frozen=True, eq=False)
class Rules:
    """K rules on one partition: ``p[i, s, j]`` is the probability that a node in state s whose density lies
    in cell j is alive after the update, under rule i."""

    partition: Partition
    p: np.ndarray

    def __post_init__(self):
        p = np.array(self.p, dtype=np.float64)  # a private copy: the check below must stay true
        if p.ndim != 3 or p.shape[1:] != (2, self.partition.ncell):
            raise ValueError(f"p must have shape [K, 2, {self.partition.ncell}], got {p.shape}")
        if not np.all((p >= 0) & (p <= 1)):  # also rejects NaN
            raise ValueError("probabilities must lie in [0, 1]")
        p.setflags(write=False)
        object.__setattr__(self, "p", p)

    def __len__(self) -> int:
        return len(self.p)

    def take(self, idx) -> Rules:
        return Rules(self.partition, self.p[np.atleast_1d(idx)])


def _codes(r: int, beta, sigma) -> tuple[np.ndarray, np.ndarray]:
    """beta and sigma broadcast against each other, as int64 arrays checked against [0, 2^r)."""
    beta, sigma = np.broadcast_arrays(np.asarray(beta, np.int64), np.asarray(sigma, np.int64))
    if np.any((beta < 0) | (beta >= 2**r) | (sigma < 0) | (sigma >= 2**r)):
        raise ValueError(f"beta and sigma must lie in [0, {2**r})")
    return beta, sigma


def life_like(r: int, beta, sigma, partition: Partition | None = None) -> Rules:
    """Rules phi^r_{beta, sigma}: born if bit j of beta is set, survive if bit j of sigma is set (cell j).

    beta and sigma broadcast against each other; the result has one rule per element.
    """
    partition = symmetric(r) if partition is None else partition
    if partition.ncell != r:
        raise ValueError(f"partition has {partition.ncell} cells, rule code has resolution {r}")
    beta, sigma = (c.ravel() for c in _codes(r, beta, sigma))
    bits = np.arange(r)
    p = np.stack([(beta[:, None] >> bits) & 1, (sigma[:, None] >> bits) & 1], axis=1)
    return Rules(partition, p)


def majority(p_tie: float = 0.5) -> Rules:
    """Watts' majority rule on the in-neighbourhood: alive if rho > 1/2, dead if rho < 1/2, and alive with
    probability ``p_tie`` at rho = 1/2. The node's own state is ignored."""
    return Rules(MAJORITY, np.array([[[0.0, p_tie, 1.0]] * 2]))


def _reverse_bits(v, r):
    out = np.zeros_like(v)
    for i in range(r):
        out |= ((v >> i) & 1) << (r - 1 - i)
    return out


def equivalent(r: int, beta, sigma):
    """The rule equivalent to phi^r_{beta, sigma} under state complementation (app03): B -> mirror(S)^C,
    S -> mirror(B)^C. Valid for the symmetric partition (odd r; even r within one convention). Scalars in,
    ints out; arrays broadcast like :func:`life_like`."""
    beta, sigma = _codes(r, beta, sigma)
    mask = 2**r - 1
    b, s = _reverse_bits(sigma, r) ^ mask, _reverse_bits(beta, r) ^ mask
    return (int(b), int(s)) if b.ndim == 0 else (b, s)


def nonequivalent(r: int) -> np.ndarray:
    """One representative (beta, sigma) per complementation class: the lexicographic minimum, ascending."""
    beta, sigma = np.meshgrid(np.arange(2**r), np.arange(2**r), indexing="ij")
    b2, s2 = equivalent(r, beta, sigma)
    keep = (beta < b2) | ((beta == b2) & (sigma <= s2))
    return np.argwhere(keep)
