import numpy as np
import pytest
import scipy.sparse as sp

import fast_llna as fl


def test_rows_are_in_neighbours():
    # A[i, j] = 1 means node i reads node j
    A = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0]])
    g = fl.Graph(A)
    assert g.n == 3
    assert [list(g.indices[g.indptr[i] : g.indptr[i + 1]]) for i in range(3)] == [[1, 2], [2], [0]]
    assert np.array_equal(g.degree, [2, 1, 1])


def test_sparse_and_dense_inputs_agree():
    A = sp.random(30, 30, density=0.2, random_state=1, format="coo")
    A = ((A + A.T) > 0).astype(np.int8).tolil()
    A.setdiag(0)
    A[np.arange(30), (np.arange(30) + 1) % 30] = 1
    g1, g2 = fl.Graph(A), fl.Graph(A.toarray())
    assert np.array_equal(g1.indptr, g2.indptr) and np.array_equal(g1.indices, g2.indices)


@pytest.mark.parametrize(
    "A, msg",
    [
        (np.array([[1, 1], [1, 0]]), "self-loop"),
        (np.array([[0, 2], [1, 0]]), "0/1"),
        (np.array([[0, 1], [0, 0]]), "in-degree 0"),
        (np.zeros((2, 3)), "square"),
    ],
)
def test_invalid_adjacency_is_rejected(A, msg):
    with pytest.raises(ValueError, match=msg):
        fl.Graph(A)


def test_ring_degrees_and_neighbours():
    g = fl.ring(10, radius=2)
    assert np.all(g.degree == 4)
    assert sorted(g.indices[g.indptr[0] : g.indptr[1]]) == [1, 2, 8, 9]


def test_moore_torus_is_8_regular_and_symmetric():
    g = fl.moore_torus(5, 7)
    assert g.n == 35 and np.all(g.degree == 8)
    A = g.adjacency()
    assert (A != A.T).nnz == 0
    # node (0, 0) neighbours (r, c) with r in {-1, 0, 1}, c in {-1, 0, 1} minus itself
    expected = sorted(((dr % 5) * 7 + dc % 7) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))
    assert sorted(g.indices[: g.indptr[1]]) == expected


def test_union_is_block_diagonal_with_offsets():
    g, offsets = fl.union(fl.ring(4), fl.moore_torus(3, 3))
    assert np.array_equal(offsets, [0, 4, 13])
    A = g.adjacency().toarray()
    assert np.array_equal(A[:4, :4], fl.ring(4).adjacency().toarray())
    assert np.array_equal(A[4:, 4:], fl.moore_torus(3, 3).adjacency().toarray())
    assert A[:4, 4:].sum() == 0 and A[4:, :4].sum() == 0
