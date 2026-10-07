import numpy as np
import pytest
import scipy.sparse as sp
from _graphs import random_digraph

import fast_llna as fl


def sim(*args, **kw):
    return fl.simulate(*args, backend="reference", **kw)


def test_game_of_life_glider_moves_diagonally():
    x = np.zeros((8, 8), bool)
    x[[0, 1, 2, 2, 2], [1, 2, 0, 1, 2]] = True
    traj = sim(fl.moore_torus(8, 8), fl.life_like(9, 8, 12), x.reshape(1, -1), 4)
    assert np.array_equal(traj.final().to_bool()[0].reshape(8, 8), np.roll(x, (1, 1), axis=(0, 1)))


def test_eca_150_on_a_ring_is_parity():
    x = np.random.default_rng(0).integers(0, 2, size=(5, 40)).astype(bool)
    traj = sim(fl.ring(40), fl.life_like(3, 2, 5), x, 7).states.to_bool()
    for t in range(7):
        assert np.array_equal(
            traj[t + 1], np.roll(traj[t], 1, axis=1) ^ traj[t] ^ np.roll(traj[t], -1, axis=1)
        )


def _complement_check(graph, partition, r, seed, n_rules=40):
    rng = np.random.default_rng(seed)
    beta, sigma = rng.integers(0, 2**r, size=(2, n_rules))
    x = rng.integers(0, 2, size=(n_rules, graph.n)).astype(bool)
    a = sim(graph, fl.life_like(r, beta, sigma, partition), x, 12).states.to_bool()
    b = sim(graph, fl.life_like(r, *fl.equivalent(r, beta, sigma), partition), ~x, 12).states.to_bool()
    return np.array_equal(a, ~b)


@pytest.mark.parametrize("r", [5, 9])
def test_complementation_symmetry_odd_r(r):
    assert _complement_check(random_digraph(60, 1, 20, seed=r), fl.symmetric(r), r, seed=r)


@pytest.mark.parametrize("r", [4, 6])
@pytest.mark.parametrize("even", ["+-", "-+"])
def test_complementation_symmetry_even_r(r, even):
    assert _complement_check(random_digraph(60, 1, 20, seed=r), fl.symmetric(r, even), r, seed=r)


def test_uniform_partition_breaks_complementation():
    # negative control: shows the symmetry test can fail (degrees multiple of 5 hit closed boundaries)
    g = fl.ring(60, radius=5)
    assert not _complement_check(g, fl.uniform(5), 5, seed=1, n_rules=200)


def test_directed_edges_point_from_source_to_reader():
    n = 6
    g = fl.Graph(sp.coo_array((np.ones(n), (np.arange(n), (np.arange(n) - 1) % n)), shape=(n, n)))
    copy = fl.life_like(3, 4, 4)  # k = 1: alive iff the single in-neighbour is alive
    x = np.zeros((1, n), bool)
    x[0, 0] = True
    assert np.flatnonzero(sim(g, copy, x, 1).final().to_bool()[0]).tolist() == [1]


def test_mixed_rules_per_replica_equal_separate_runs():
    g = random_digraph(50, 2, 9, seed=3)
    rules = fl.life_like(5, [3, 17, 30], [12, 0, 31])
    x = fl.random_states(50, 3, seed=4)
    together = sim(g, rules, x, 9).states.to_bool()
    xb = x.to_bool()
    for k in range(3):
        alone = sim(g, rules.take([k]), xb[k : k + 1], 9).states.to_bool()
        assert np.array_equal(together[:, k], alone[:, 0])


def test_majority_is_deterministic_without_ties():
    g = random_digraph(40, 3, 3, seed=5)  # every in-degree odd
    x = fl.random_states(40, 4, seed=6)
    a = sim(g, fl.majority(), x, 5, seed=1).states.to_bool()
    b = sim(g, fl.majority(), x, 5, seed=2).states.to_bool()
    assert np.array_equal(a, b)
    q = x.to_bool().astype(int) @ g.adjacency().T.toarray()
    assert np.array_equal(a[1], 2 * q > 3)


def test_majority_tie_is_a_fair_coin():
    x = np.tile(np.array([1, 1, 0, 0], bool), 25)[None].repeat(
        320, axis=0
    )  # every node sees one of two alive
    nxt = sim(fl.ring(100), fl.majority(), x, 1, seed=11).final().to_bool()
    n = nxt.size
    assert abs(nxt.mean() - 0.5) < 5 * 0.5 / np.sqrt(n)
    assert len({row.tobytes() for row in nxt}) == len(nxt)


def test_clamped_nodes_hold_including_initial_state():
    g = random_digraph(30, 2, 6, seed=7)
    R = 5
    mask = np.zeros((R, 30), bool)
    mask[np.arange(R), np.arange(R)] = True
    value = np.ones((R, 30), bool)
    traj = sim(g, fl.life_like(5, 0, 0), fl.random_states(30, R, density=0.0, seed=0), 6, clamp=(mask, value))
    x = traj.states.to_bool()
    assert np.all(x[:, mask])
    assert x[1:][:, ~mask].sum() == 0  # rule (0, 0) kills everything that is not clamped


def test_continuation_with_t0_matches_one_run():
    g = random_digraph(40, 2, 8, seed=8)
    x = fl.random_states(40, 33, seed=9)
    whole = sim(g, fl.majority(), x, 10, seed=5).final().bits
    half = sim(g, fl.majority(), x, 6, seed=5).final()
    rest = sim(g, fl.majority(), half, 4, seed=5, t0=6)
    assert np.array_equal(rest.final().bits, whole) and rest.times.tolist() == [6, 7, 8, 9, 10]


def test_record_modes():
    g = random_digraph(20, 1, 5, seed=10)
    x = fl.random_states(20, 9, seed=0)
    full = sim(g, fl.life_like(5, 6, 28), x, 9)
    every3 = sim(g, fl.life_like(5, 6, 28), x, 9, record=3)
    final = sim(g, fl.life_like(5, 6, 28), x, 9, record="final")
    assert np.array_equal(every3.states.bits, full.states.bits[::3]) and every3.times.tolist() == [0, 3, 6, 9]
    assert np.array_equal(final.states.bits, full.states.bits[-1:]) and final.times.tolist() == [9]
    with pytest.raises(ValueError, match="multiple"):
        sim(g, fl.life_like(5, 6, 28), x, 10, record=3)


def test_size_guard_reports_projected_size():
    with pytest.raises(ValueError, match="bytes"):
        sim(fl.ring(100), fl.life_like(3, 2, 5), fl.random_states(100, 64), 1000, max_bytes=1e4)


def test_twins_share_noise_with_noise_period():
    g = fl.ring(64)
    x, pairs = fl.defect_twins(fl.random_states(64, 32, seed=1), flips=0)
    shared = sim(g, fl.majority(), x, 8, seed=3, noise_period=32).hamming(pairs)
    indep = sim(g, fl.majority(), x, 8, seed=3).hamming(pairs)
    assert np.all(shared == 0) and indep.max() > 0


def test_rule_count_must_match_replicas():
    with pytest.raises(ValueError, match="rules"):
        sim(fl.ring(10), fl.life_like(3, [1, 2], 5), fl.random_states(10, 3), 1)
