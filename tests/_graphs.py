import numpy as np
import scipy.sparse as sp

import fast_llna as fl


def random_digraph(n, kmin, kmax, seed):
    """Directed graph where node i reads kmin..kmax distinct random other nodes."""
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for i in range(n):
        k = rng.integers(kmin, kmax + 1)
        rows += [i] * k
        cols += list(rng.choice(np.delete(np.arange(n), i), size=k, replace=False))
    return fl.Graph(sp.coo_array((np.ones(len(rows)), (rows, cols)), shape=(n, n)))


def star_hub(n, seed):
    """Node 0 reads every other node (in-degree n - 1); the others read node 0 and 0-3 random nodes."""
    rng = np.random.default_rng(seed)
    A = sp.lil_array((n, n), dtype=np.int8)
    A[0, 1:] = 1
    for i in range(1, n):
        A[i, 0] = 1
        A[i, rng.choice(np.delete(np.arange(1, n), i - 1), size=rng.integers(0, 4), replace=False)] = 1
    return fl.Graph(A)
