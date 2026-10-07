import numpy as np
import scipy.sparse as sp

import fast_llna as fl


def random_digraph(n, kmin, kmax, seed):
    """Directed graph where node i reads kmin..kmax distinct random other nodes."""
    return degree_graph(np.random.default_rng(seed).integers(kmin, kmax + 1, size=n), seed)


def star_hub(n, seed, hub=0):
    """Node `hub` reads every other node (in-degree n - 1); the others read it and 0-3 random nodes."""
    rng = np.random.default_rng(seed)
    A = sp.lil_array((n, n), dtype=np.int8)
    A[0, 1:] = 1
    for i in range(1, n):
        A[i, 0] = 1
        A[i, rng.choice(np.delete(np.arange(1, n), i - 1), size=rng.integers(0, 4), replace=False)] = 1
    p = np.arange(n)
    p[[0, hub]] = p[[hub, 0]]
    return fl.Graph(sp.csr_array(A)[p][:, p])


def degree_graph(degrees, seed):
    """Directed graph in which node i reads degrees[i] distinct random other nodes."""
    rng = np.random.default_rng(seed)
    n = len(degrees)
    rows = np.repeat(np.arange(n), degrees)
    cols = np.concatenate(
        [rng.choice(np.delete(np.arange(n), i), size=d, replace=False) for i, d in enumerate(degrees)]
    )
    return fl.Graph(sp.coo_array((np.ones(len(rows)), (rows, cols)), shape=(n, n)))
