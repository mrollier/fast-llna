import numpy as np
import pytest

import fast_llna as fl
from fast_llna.states import bits_per_node, frame_bytes


@pytest.mark.parametrize("R, L", [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8), (9, 16), (17, 24), (33, 40)])
def test_bits_per_node(R, L):
    assert bits_per_node(R) == L


@pytest.mark.parametrize("R", [1, 2, 3, 5, 7, 8, 9, 17, 31, 33])
def test_flat_node_major_layout(R):
    n, L = 13, bits_per_node(R)
    x = np.random.default_rng(R).integers(0, 2, size=(3, R, n)).astype(bool)
    s = fl.States.from_bool(x)
    assert s.bits.shape == (3, frame_bytes(n, R)) == (3, -(-n * L // 8))
    assert s.n_replicas == R and s.n == n
    assert np.array_equal(s.to_bool(), x)
    bit = np.unpackbits(s.bits, axis=-1, bitorder="little").astype(bool)  # [3, 8 * frame bytes]
    t, i, r = np.meshgrid(np.arange(3), np.arange(n), np.arange(R), indexing="ij")
    assert np.array_equal(bit[t, i * L + r], x[t, r, i])  # replica r of node i is bit i * L + r
    unused = np.ones(bit.shape[-1], bool)
    unused[(i * L + r)[0].ravel()] = False
    assert not bit[:, unused].any()  # padding lanes and trailing bits are 0


def test_random_states_have_exact_density():
    s = fl.random_states(50, 40, density=0.3, seed=1).to_bool()
    assert s.shape == (40, 50)
    assert np.all(s.sum(axis=1) == 15)
    assert len({row.tobytes() for row in s}) == 40


def test_random_states_are_seeded():
    a, b = fl.random_states(20, 5, seed=3), fl.random_states(20, 5, seed=3)
    assert np.array_equal(a.bits, b.bits)


def test_defect_twins_flip_exactly_k_nodes():
    s = fl.random_states(30, 10, seed=0)
    twins, pairs = fl.defect_twins(s, flips=3, seed=1)
    x = twins.to_bool()
    assert twins.n_replicas == 20 and np.array_equal(pairs, np.c_[np.arange(10), np.arange(10, 20)])
    assert np.array_equal(x[:10], s.to_bool())
    assert np.all((x[:10] ^ x[10:]).sum(axis=1) == 3)


def test_product_is_rule_major():
    rules = fl.life_like(5, [1, 2], 3)
    s = fl.random_states(12, 3, seed=0)
    pr, ps = fl.product(rules, s)
    assert np.array_equal(pr.p, rules.p[[0, 0, 0, 1, 1, 1]])
    assert np.array_equal(ps.to_bool(), np.tile(s.to_bool(), (2, 1)))


@pytest.mark.parametrize("R", [1, 3, 11])
def test_trajectory_reducers_match_naive(R):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=(4, R, 25)).astype(bool)  # [T, R, N]
    traj = fl.Trajectory(fl.States.from_bool(x), np.arange(4))
    assert np.allclose(traj.density(), x.mean(axis=2))
    pairs = np.array([[0, R - 1], [R // 2, 0], [R - 1, R - 1]])
    assert np.allclose(traj.hamming(pairs), (x[:, pairs[:, 0]] ^ x[:, pairs[:, 1]]).mean(axis=2))
    assert np.array_equal(traj.final().to_bool(), x[-1])


def test_states_rejects_bits_that_do_not_match_n_and_R():
    bits = fl.States.from_bool(np.ones((8, 4), bool)).bits  # 4 nodes x 8 replicas: 4 bytes
    with pytest.raises(ValueError, match="bytes"):
        fl.States(bits.reshape(4, 1), 8, 4)  # v1 row shape [N, nb]
    with pytest.raises(ValueError, match="bytes"):
        fl.States(bits, 8, 3)


def test_from_bool_needs_replica_and_node_axes():
    with pytest.raises(ValueError, match=r"\[R, N\]"):
        fl.States.from_bool(np.ones(5, bool))


def test_states_need_at_least_one_replica_and_node():
    with pytest.raises(ValueError):
        fl.States(np.zeros(1, np.uint8), 0, 4)


def test_random_states_and_twins_validate_their_counts():
    with pytest.raises(ValueError, match="density"):
        fl.random_states(10, 2, density=1.5)
    with pytest.raises(ValueError, match="flips"):
        fl.defect_twins(fl.random_states(10, 2, seed=0), flips=11)


def test_hamming_accepts_a_single_pair():
    x = np.random.default_rng(1).integers(0, 2, size=(3, 4, 9)).astype(bool)
    traj = fl.Trajectory(fl.States.from_bool(x), np.arange(3))
    got = traj.hamming([0, 3])
    assert got.shape == (3, 1) and np.allclose(got[:, 0], (x[:, 0] ^ x[:, 3]).mean(axis=1))
