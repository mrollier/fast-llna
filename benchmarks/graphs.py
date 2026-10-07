"""Graph generators for benchmarks (numpy/scipy ports of the llna thesis generators)."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

import fast_llna as fl


def _undirected(n, i, j):
    A = sp.coo_array((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    return ((A + A.T) > 0).astype(np.int8)


def rewired_torus(L, p, seed=0):
    """Moore torus L x L with every edge's endpoint moved with probability p to a random non-neighbour
    (Watts-Strogatz style, as llna.networks.watts_strogatz_rewire); retried until connected."""
    rng = np.random.default_rng(seed)
    base = sp.triu(fl.moore_torus(L, L).adjacency(), format="coo")
    n = L * L
    for _ in range(100):
        nbrs = [set() for _ in range(n)]
        for a, b in zip(base.row, base.col, strict=True):
            nbrs[a].add(b)
            nbrs[b].add(a)
        for a, b in zip(base.row, base.col, strict=True):
            if rng.random() < p:
                nbrs[a].discard(b)
                nbrs[b].discard(a)
                while (c := int(rng.integers(n))) == a or c in nbrs[a]:
                    pass
                nbrs[a].add(c)
                nbrs[c].add(a)
        i = np.repeat(np.arange(n), [len(s) for s in nbrs])
        A = _undirected(n, i, np.fromiter((c for s in nbrs for c in s), int))
        if connected_components(A, directed=False)[0] == 1 and A.sum(axis=1).min() > 0:
            return fl.Graph(A)
    raise RuntimeError("could not build a connected rewired torus")


def erdos_renyi(n, mean_degree, seed=0):
    """G(n, m) with m = n * mean_degree / 2 undirected edges; isolated nodes get one random edge."""
    rng = np.random.default_rng(seed)
    m = int(n * mean_degree / 2)
    i, j = rng.integers(0, n, size=(2, int(m * 1.05) + 16))
    keep = i != j
    i, j = i[keep][:m], j[keep][:m]
    deg = np.bincount(np.r_[i, j], minlength=n)
    lonely = np.flatnonzero(deg == 0)
    partner = (lonely + 1 + rng.integers(0, n - 1, size=len(lonely))) % n
    return fl.Graph(_undirected(n, np.r_[i, lonely], np.r_[j, partner]))


def barabasi_albert(n, m, seed=0):
    """Preferential attachment: each new node links to m existing nodes chosen proportionally to degree."""
    rng = np.random.default_rng(seed)
    targets = list(range(m))
    repeated, src, dst = [], [], []
    for v in range(m, n):
        chosen = set(targets)
        src += [v] * len(chosen)
        dst += list(chosen)
        repeated += list(chosen) + [v] * len(chosen)
        targets = []
        while len(targets) < m:
            c = repeated[rng.integers(len(repeated))]
            if c not in targets:
                targets.append(c)
    return fl.Graph(_undirected(n, np.array(src), np.array(dst)))
