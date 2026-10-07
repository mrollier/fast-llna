import numpy as np
import pytest

import fast_llna as fl


@pytest.mark.parametrize("R", [1, 7, 8, 9, 31, 33])
def test_pack_roundtrip_and_canonical_layout(R):
    x = np.random.default_rng(R).integers(0, 2, size=(3, R, 13)).astype(bool)
    s = fl.States.from_bool(x)
    assert s.bits.shape == (3, 13, (R + 7) // 8) and s.n_replicas == R
    assert np.array_equal(s.to_bool(), x)
    # canonical: bit r % 8 of byte r // 8 of node row i is replica r (little bit order)
    t, i, r = 2, 5, R - 1
    assert bool((s.bits[t, i, r // 8] >> (r % 8)) & 1) == x[t, r, i]


def test_padding_bits_are_zero():
    s = fl.States.from_bool(np.ones((9, 4), bool))
    assert np.all(s.bits[:, 1] == 1)


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


def test_trajectory_reducers_match_naive():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=(4, 11, 25)).astype(bool)  # [T, R, N]
    traj = fl.Trajectory(fl.States.from_bool(x), np.arange(4))
    assert np.allclose(traj.density(), x.mean(axis=2))
    pairs = np.array([[0, 5], [3, 10], [7, 7]])
    assert np.allclose(traj.hamming(pairs), (x[:, pairs[:, 0]] ^ x[:, pairs[:, 1]]).mean(axis=2))
    assert np.array_equal(traj.final().to_bool(), x[-1])
