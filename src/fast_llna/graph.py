"""Graphs as CSR in-neighbour lists: ``A[i, j] = 1`` means node i reads the state of node j."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class Graph:
    """A simple (no self-loops, no multi-edges) possibly directed graph in which every node has at least one
    in-neighbour. Row i of the adjacency matrix lists the in-neighbours of node i; undirected graphs are
    symmetric matrices."""

    def __init__(self, adjacency):
        A = sp.csr_array(adjacency)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"adjacency must be square, got shape {A.shape}")
        A.sum_duplicates()
        A.eliminate_zeros()
        if np.any(A.data != 1):
            raise ValueError("adjacency entries must be 0/1")
        if A.diagonal().any():
            raise ValueError(f"self-loops are not allowed (nodes {np.flatnonzero(A.diagonal())[:10]})")
        A.sort_indices()
        if A.nnz >= 2**31:
            raise ValueError("more than 2**31 - 1 edges are not supported")
        self.indptr = A.indptr.astype(np.int32)
        self.indices = A.indices.astype(np.int32)
        isolated = np.flatnonzero(self.degree == 0)
        if isolated.size:
            raise ValueError(f"nodes with in-degree 0 have no defined density (nodes {isolated[:10]})")

    @property
    def n(self) -> int:
        return len(self.indptr) - 1

    @property
    def degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    def adjacency(self) -> sp.csr_array:
        data = np.ones(len(self.indices), dtype=np.int8)
        return sp.csr_array((data, self.indices, self.indptr), shape=(self.n, self.n))


def ring(n: int, radius: int = 1) -> Graph:
    """Ring lattice: node i reads nodes i +- 1, ..., i +- radius (periodic)."""
    i = np.repeat(np.arange(n), 2 * radius)
    offsets = np.tile(np.r_[np.arange(-radius, 0), np.arange(1, radius + 1)], n)
    return Graph(sp.coo_array((np.ones(len(i)), (i, (i + offsets) % n)), shape=(n, n)))


def moore_torus(rows: int, cols: int) -> Graph:
    """Periodic 2-D grid with Moore neighbourhood; node (r, c) has index r * cols + c."""
    r, c = np.divmod(np.arange(rows * cols), cols)
    shifts = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    i = np.repeat(np.arange(rows * cols), len(shifts))
    j = np.concatenate([((r + dr) % rows) * cols + (c + dc) % cols for dr, dc in shifts], axis=0)
    j = j.reshape(len(shifts), -1).T.ravel()
    return Graph(sp.coo_array((np.ones(len(i)), (i, j)), shape=(rows * cols,) * 2))


def union(*graphs: Graph) -> tuple[Graph, np.ndarray]:
    """Disjoint union; node v of ``graphs[g]`` becomes node ``offsets[g] + v``."""
    offsets = np.cumsum([0] + [g.n for g in graphs])
    return Graph(sp.block_diag([g.adjacency() for g in graphs], format="csr")), offsets
