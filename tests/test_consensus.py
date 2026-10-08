"""fl.consensus against a brute-force walk over directly simulated frames."""

import importlib
import weakref

import numpy as np
import pytest
import scipy.sparse as sp
from _graphs import degree_graph, random_digraph

import fast_llna as fl

AVAILABLE = fl.available_backends()
BACKENDS = [
    pytest.param(
        b,
        marks=[getattr(pytest.mark, b), pytest.mark.skipif(b not in AVAILABLE, reason=f"{b} not available")],
    )
    for b in ("cpu", "metal", "cuda")
]
FIELDS = ("t_consensus", "state", "t_cycle", "period")


def cycle(n):
    """Directed cycle: node i reads node i + 1."""
    i = np.arange(n)
    return fl.Graph(sp.coo_array((np.ones(n), (i, (i + 1) % n)), shape=(n, n)))


GRAPH, OFFSETS = fl.union(random_digraph(5, 1, 3, 1), random_digraph(7, 1, 4, 2), random_digraph(9, 2, 5, 3))


def mixed(R, seed):
    """Random r=5 rules, every third one quiescent self-equivalent."""
    rng = np.random.default_rng(seed)
    beta, sigma = rng.integers(0, 32, size=(2, R))
    se = fl.self_equivalent(5)[rng.integers(0, 16, size=R) * 2]
    beta[::3], sigma[::3] = se[::3].T
    return fl.life_like(5, beta, sigma)


def init(R, n, seed):
    return np.random.default_rng(seed).random((R, n)) < 0.5


def oracle(graph, offsets, rules, x0, t_max, det, seed=0):
    """Walk every (segment, lane) of a direct reference run: first consensus; for deterministic lanes without
    one, the first repeat (mu, lambda) and when the power-of-two schedule finds it."""
    x = fl.simulate(graph, rules, x0, t_max, backend="reference", seed=seed).states.to_bool()
    S, R = len(offsets) - 1, x.shape[1]
    t_cons, state = np.full((S, R), np.inf), np.full((S, R), -1, np.int8)
    t_cyc, period = np.full((S, R), np.inf), np.zeros((S, R), np.int64)
    for s, lane in np.ndindex(S, R):
        seen = {}
        for t, frame in enumerate(x[:, lane, offsets[s] : offsets[s + 1]]):
            if frame.all() or not frame.any():
                t_cons[s, lane], state[s, lane] = t, frame[0]
                break
            key = frame.tobytes()
            if det[lane] and key in seen:
                mu, lam = seen[key], t - seen[key]
                t_ref = 0 if mu == 0 else 1 << (mu - 1).bit_length()  # smallest of 0, 1, 2, 4, ... >= mu
                while lam > max(t_ref, 1):
                    t_ref = max(1, 2 * t_ref)
                if t_ref + lam <= t_max:
                    t_cyc[s, lane], period[s, lane] = t_ref + lam, lam
                break
            seen[key] = t
    return t_cons, state, t_cyc, period


def check(got, want):
    for name, w in zip(FIELDS, want, strict=True):
        np.testing.assert_array_equal(getattr(got, name), w, err_msg=name)


@pytest.mark.parametrize("R", [1, 3, 8, 9, 31, 64, 65, 200])
def test_matches_oracle(R):
    rules, x0 = mixed(R, R), init(R, GRAPH.n, R)
    got = fl.consensus(GRAPH, rules, x0, 100, offsets=OFFSETS, chunk=8, backend="reference")
    check(got, oracle(GRAPH, OFFSETS, rules, x0, 100, np.ones(R, bool)))


def test_chunk_independent():
    rules, x0 = mixed(40, 7), init(40, GRAPH.n, 7)
    want = oracle(GRAPH, OFFSETS, rules, x0, 150, np.ones(40, bool))
    for chunk in (1, 2, 8, 64):
        check(fl.consensus(GRAPH, rules, x0, 150, offsets=OFFSETS, chunk=chunk, backend="reference"), want)
    with pytest.raises(ValueError, match="power of two"):
        fl.consensus(GRAPH, rules, x0, 150, offsets=OFFSETS, chunk=3)


@pytest.mark.parametrize("t_max", [0, 37])
def test_short_runs(t_max):
    rules, x0 = mixed(20, 8), init(20, GRAPH.n, 8)
    x0[:5, OFFSETS[1] : OFFSETS[2]] = np.arange(5)[:, None] % 2  # uniform in segment 1 from the start
    got = fl.consensus(GRAPH, rules, x0, t_max, offsets=OFFSETS, chunk=8, backend="reference")
    check(got, oracle(GRAPH, OFFSETS, rules, x0, t_max, np.ones(20, bool)))
    assert np.all(got.t_consensus[1, :5] == 0) and np.all(got.state[1, :5] == np.arange(5) % 2)
    assert np.any(np.isinf(got.t_consensus) & np.isinf(got.t_cycle))  # some still open at t_max


def test_shared_rule_and_broadcast_init():
    x0 = init(1, GRAPH.n, 9)[0]
    rule = fl.life_like(5, 4, 27)
    got = fl.consensus(GRAPH, rule, x0, 60, offsets=OFFSETS, backend="reference")
    check(got, oracle(GRAPH, OFFSETS, rule, x0[None], 60, [True]))
    states = fl.consensus(
        GRAPH, rule, fl.States.from_bool(x0[None]), 60, offsets=OFFSETS, backend="reference"
    )
    check(states, oracle(GRAPH, OFFSETS, rule, x0[None], 60, [True]))


def test_non_uniform_fixed_point():
    x0 = np.arange(GRAPH.n) % 2 == 0
    got = fl.consensus(GRAPH, fl.life_like(5, 0, 31), x0, 50, offsets=OFFSETS, backend="reference")
    assert np.all(got.t_cycle == 1) and np.all(got.period == 1) and np.all(np.isinf(got.t_consensus))


def test_period_power_of_two():
    """One living node walks around a directed 8-cycle: mu = 0, lambda = 8, found at t_ref + lambda = 8 + 8
    (a 5-cycle: at 8 + 5)."""
    graph, offsets = fl.union(cycle(8), cycle(5))
    x0 = np.zeros(13, bool)
    x0[[2, 9]] = True
    got = fl.consensus(
        graph, fl.life_like(2, 2, 2, fl.symmetric(2)), x0, 100, offsets=offsets, backend="reference"
    )
    assert got.t_cycle.ravel().tolist() == [16, 13] and got.period.ravel().tolist() == [8, 5]
    assert np.all(np.isinf(got.t_consensus)) and np.all(got.state == -1)


def test_quiescent_lanes():
    """Only quiescent lanes: consensus is absorbing, so only chunk ends are tested until one appears."""
    se = fl.self_equivalent(5)[::2]
    rules, x0 = fl.product(fl.life_like(5, *se.T), fl.States.from_bool(init(3, GRAPH.n, 4)))
    got = fl.consensus(GRAPH, rules, x0, 100, offsets=OFFSETS, chunk=8, backend="reference")
    check(got, oracle(GRAPH, OFFSETS, rules, x0, 100, np.ones(48, bool)))


@pytest.mark.parametrize(
    "beta, sigma, cell, seg, t", [(6, 11, (0, 0), 1, 13), (6, 21, (1, 4), 2, 12)], ids=["born", "dies"]
)
def test_transient_consensus_of_non_quiescent_lane(beta, sigma, cell, seg, t):
    """Lane 0 is not quiet on one degree (a dead node of degree 1 among dead neighbours is born, or a living
    node of degree 2 among living ones dies), so a uniform segment need not stay uniform: segment seg agrees
    at t only, inside the chunk 8..16. Every frame must be scanned while such a lane runs."""
    ks = np.unique(GRAPH.degree)
    se = fl.self_equivalent(5)[::2]  # quiescent
    rules = fl.life_like(5, np.r_[beta, se[:, 0]], np.r_[sigma, se[:, 1]]).exact(ks)
    p = rules.p.copy()
    p[0, cell[0], cell[1]] = 1 - cell[0]  # exact cells: (k=1, q=0) is 0, (k=2, q=2) is 4
    rules = fl.Rules(rules.partition, p)
    x0 = np.random.default_rng(5).random(GRAPH.n) < 0.5
    got = fl.consensus(GRAPH, rules, x0, 64, offsets=OFFSETS, chunk=8, backend="reference")
    check(got, oracle(GRAPH, OFFSETS, rules, np.tile(x0, (17, 1)), 64, np.ones(17, bool)))
    assert got.t_consensus[seg, 0] == t


def test_stochastic_lanes():
    """Deterministic lanes first, stochastic ones last: these must keep their positions (random draws)."""
    graph, offsets = fl.union(fl.ring(5), fl.ring(7), fl.ring(9))  # even degrees: majority ties are random
    p = np.array([fl.majority(0.0).p[0], fl.majority(1.0).p[0]] * 4 + [fl.majority(0.5).p[0]] * 4)
    rules, x0 = fl.Rules(fl.MAJORITY, p), init(12, graph.n, 12)
    got = fl.consensus(graph, rules, x0, 200, offsets=offsets, seed=5, chunk=16, backend="reference")
    check(got, oracle(graph, offsets, rules, x0, 200, np.arange(12) < 8, seed=5))
    assert np.all(np.isinf(got.t_cycle[:, 8:])) and np.all(got.period[:, 8:] == 0)
    assert np.isfinite(got.t_consensus[:, 8:]).any()


def test_majority_on_odd_degrees_is_deterministic():
    graph, offsets = fl.union(degree_graph([3] * 7, 1), degree_graph([1, 3, 5, 3, 1, 3, 3, 5], 2))
    x0 = init(16, graph.n, 13)
    got = fl.consensus(graph, fl.majority(0.5), x0, 100, offsets=offsets, seed=1, backend="reference")
    check(got, oracle(graph, offsets, fl.majority(0.5), x0, 100, np.ones(16, bool)))
    assert np.any(got.period > 0)


def test_compaction(monkeypatch):
    """58 lanes die at t = 1, 7 copy a living node around a 13-cycle (period 13, found at 16 + 13)."""
    calls = []
    simulate = fl.simulate

    def recording(graph, rules, x, *args, **kw):
        calls.append(x.n_replicas)
        return simulate(graph, rules, x, *args, **kw)

    monkeypatch.setattr(importlib.import_module("fast_llna.consensus"), "simulate", recording)
    graph, offsets = fl.union(cycle(13), fl.ring(7))
    long = np.zeros(65, bool)
    long[[0, 9, 20, 33, 47, 60, 64]] = True
    rules = fl.life_like(2, np.where(long, 2, 0), np.where(long, 2, 0), fl.symmetric(2))
    x0 = init(65, graph.n, 14)
    x0[long, :13] = np.arange(13) == 4
    got = fl.consensus(graph, rules, x0, 100, offsets=offsets, chunk=4, backend="reference")
    check(got, oracle(graph, offsets, rules, x0, 100, np.ones(65, bool)))
    assert got.t_cycle[0, long].tolist() == [29] * 7
    assert calls[0] == 128 and calls[-1] == 8


def test_one_chunk_alive_at_a_time(monkeypatch):
    """The previous chunk's trajectory is freed before the next simulate call: peak memory is one chunk."""
    refs, alive, simulate = [], [], fl.simulate

    def recording(*args, **kw):
        alive.append(sum(r() is not None for r in refs))
        traj = simulate(*args, **kw)
        refs.append(weakref.ref(traj.states.bits))
        return traj

    monkeypatch.setattr(importlib.import_module("fast_llna.consensus"), "simulate", recording)
    fl.consensus(GRAPH, mixed(9, 9), init(9, GRAPH.n, 9), 100, offsets=OFFSETS, chunk=8, backend="reference")
    assert len(alive) > 3 and alive == [0] * len(alive)


@pytest.mark.parametrize(
    "offsets", [[1, 5, 12, 21], [0, 5, 12, 20], [0, 7, 5, 21], [0, 5, 5, 12, 21], [0.0, 21.0], [[0, 21]]]
)
def test_bad_offsets(offsets):
    with pytest.raises(ValueError, match="offsets"):
        fl.consensus(GRAPH, mixed(2, 1), init(2, GRAPH.n, 1), 10, offsets=offsets)


def test_cross_segment_edge():
    with pytest.raises(ValueError, match="crosses"):
        fl.consensus(GRAPH, mixed(2, 1), init(2, GRAPH.n, 1), 10, offsets=[0, 6, 21])


def test_bad_arguments():
    rules, x0 = mixed(2, 1), init(2, GRAPH.n, 1)
    with pytest.raises(ValueError, match="t_max"):
        fl.consensus(GRAPH, rules, x0, -1, offsets=OFFSETS)
    with pytest.raises(ValueError, match="init"):
        fl.consensus(GRAPH, rules, x0[:, :-1], 10, offsets=OFFSETS)
    with pytest.raises(ValueError, match="rules"):
        fl.consensus(GRAPH, mixed(3, 1), x0, 10, offsets=OFFSETS)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("R", [3, 40, 65])
def test_backend_matches_reference(backend, R):
    p = mixed(R, R).p.copy()
    if R < 65:  # a stochastic lane (no compaction)
        p[-1, :, 2] = 0.5
    rules = fl.Rules(fl.symmetric(5), p)
    x0 = init(R, GRAPH.n, R)
    kw = {"offsets": OFFSETS, "seed": 3, "chunk": 8}
    want = fl.consensus(GRAPH, rules, x0, 120, backend="reference", **kw)
    check(fl.consensus(GRAPH, rules, x0, 120, backend=backend, **kw), [getattr(want, f) for f in FIELDS])
