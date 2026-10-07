"""Every compiled backend must reproduce the reference backend bit for bit."""

import numpy as np
import pytest
from _graphs import degree_graph, random_digraph, star_hub

import fast_llna as fl

BACKENDS = [
    pytest.param(b, marks=getattr(pytest.mark, b, ()))
    for b in ("cpu", "metal", "cuda")
    if b in fl.available_backends()
]


def mixed(r, R, seed, partition=None):
    beta, sigma = np.random.default_rng(seed).integers(0, 2**r, size=(2, R))
    return fl.life_like(r, beta, sigma, partition)


def stochastic(partition, R, seed):
    rng = np.random.default_rng(seed)
    choices = np.array([0, 1, 0.5, 0.25, 0.75, 0.3, 1e-3, 0.999])
    return fl.Rules(partition, rng.choice(choices, size=(R, 2, partition.ncell)))


def twins(R, seed):
    return fl.defect_twins(fl.random_states(60, R, seed=seed), flips=2, seed=seed)[0]


def clamp(R, n, seed):
    rng = np.random.default_rng(seed)
    return rng.random((R, n)) < 0.1, rng.random((R, n)) < 0.5


# mean degree 3.95 -> LLNA_HUB = 32: degree 32 takes the normal path, 33, 34, 64 and 140 the cooperative one;
# the 140-degree hub is node 149 of 150, in the last (partial) SIMD group
HUBS = degree_graph([2] * 7 + [32] + [2] * 32 + [33, 34] + [2] * 57 + [64] + [2] * 49 + [140], seed=21)


def test_hub_threshold_scales_with_mean_degree():
    from fast_llna.kernels import tables

    rule = fl.life_like(5, 6, 28)
    assert tables.build(HUBS, rule, fl.random_states(150, 1), None, 0, None, 1, 1).defines["LLNA_HUB"] == 32
    dense = fl.ring(300, 20)  # degree 40
    assert tables.build(dense, rule, fl.random_states(300, 1), None, 0, None, 1, 1).defines["LLNA_HUB"] == 160


# (id, graph, R, rules, steps, simulate kwargs); init defaults to random states with R replicas
CASES = [
    ("eca150-R1", fl.ring(40), 1, lambda R: fl.life_like(3, 2, 5), 12, {}),
    ("moore-mixed-R31", fl.moore_torus(6, 7), 31, lambda R: mixed(9, R, 1), 10, {}),
    (
        "even4-R32-record3",
        random_digraph(60, 1, 20, 2),
        32,
        lambda R: mixed(4, R, 2, fl.symmetric(4)),
        9,
        {"record": 3},
    ),
    (
        "uniform5-R33-clamp",
        random_digraph(60, 1, 20, 3),
        33,
        lambda R: mixed(5, R, 3, fl.uniform(5)),
        8,
        {"clamp": clamp(33, 60, 3)},
    ),
    ("star1024-R64-final", star_hub(1024, 4), 64, lambda R: fl.life_like(9, 72, 12), 6, {"record": "final"}),
    (
        "union-majority-R65",
        fl.union(fl.ring(30), fl.moore_torus(5, 6))[0],
        65,
        lambda R: fl.majority(),
        10,
        {"seed": 2**63 + 12345},
    ),
    (
        "twins-shared-noise-R256",
        random_digraph(60, 2, 9, 5),
        256,
        lambda R: fl.majority(),
        10,
        {"noise_period": 128, "init": twins(128, 5)},
    ),
    (
        "stochastic-depth32-R1056",
        fl.moore_torus(6, 7),
        1056,
        lambda R: stochastic(fl.symmetric(5), R, 6),
        6,
        {"seed": 7, "t0": 1000},
    ),
    (
        "even6-swapped-R100",
        random_digraph(60, 1, 25, 7),
        100,
        lambda R: mixed(6, R, 7, fl.symmetric(6, "-+")),
        8,
        {},
    ),
    (
        "stochastic-clamp-R257",
        random_digraph(60, 1, 12, 8),
        257,
        lambda R: stochastic(fl.MAJORITY, R, 8),
        7,
        {"clamp": clamp(257, 60, 8), "record": 7},
    ),
    (
        "majority-R64-noise16",
        fl.ring(50, 2),
        64,
        lambda R: fl.majority(),
        6,
        {"noise_period": 16, "seed": 9},
    ),
    ("ring5-R1", fl.ring(5), 1, lambda R: fl.life_like(3, 2, 5), 7, {}),
    (
        "twins-R2-noise1",
        random_digraph(60, 2, 9, 11),
        2,
        lambda R: fl.majority(),
        9,
        {"noise_period": 1, "init": twins(1, 11)},
    ),
    (
        "even4-R3-clamp",
        random_digraph(77, 1, 20, 12),
        3,
        lambda R: mixed(4, R, 12, fl.symmetric(4)),
        8,
        {"clamp": clamp(3, 77, 12), "record": 2},
    ),
    ("stochastic-R5", random_digraph(61, 1, 12, 13), 5, lambda R: stochastic(fl.symmetric(5), R, 13), 7, {"seed": 5}),
    (
        "uniform5-R9-record",
        random_digraph(45, 1, 16, 14),
        9,
        lambda R: mixed(5, R, 14, fl.uniform(5)),
        6,
        {"record": 3},
    ),
    ("R20-strided-final", random_digraph(50, 1, 10, 15), 20, lambda R: mixed(9, R, 15), 5, {"record": "final"}),
    (
        "hubs-R1-stochastic-clamp",
        HUBS,
        1,
        lambda R: stochastic(fl.symmetric(5), R, 21),
        8,
        {"clamp": clamp(1, 150, 21), "seed": 21},
    ),
    ("hubs-R3-mixed", HUBS, 3, lambda R: mixed(9, R, 22), 8, {"record": 2}),
    ("hubs-R32-majority", HUBS, 32, lambda R: stochastic(fl.MAJORITY, R, 23), 8, {"seed": 23}),
    ("star-last-R4", star_hub(200, 24, hub=199), 4, lambda R: fl.life_like(9, 72, 12), 6, {}),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_backend_matches_reference(backend, case):
    _, graph, R, rules, steps, kw = case
    kw = dict(kw)
    init = kw.pop("init", None) or fl.random_states(graph.n, R, seed=R)
    want = fl.simulate(graph, rules(R), init, steps, backend="reference", **kw)
    got = fl.simulate(graph, rules(R), init, steps, backend=backend, **kw)
    assert np.array_equal(got.times, want.times)
    assert got.states.bits.shape == want.states.bits.shape
    assert np.array_equal(got.states.bits, want.states.bits)


@pytest.mark.skipif("cpu" not in fl.available_backends(), reason="no C compiler")
@pytest.mark.parametrize("mode", ["words", "nodes"])
@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("R", [3, 300])
def test_cpu_modes_and_thread_counts(mode, threads, R, monkeypatch):
    monkeypatch.setenv("FAST_LLNA_CPU_MODE", mode)
    graph = random_digraph(700, 1, 15, 9)
    rules = stochastic(fl.symmetric(5), R, 9)
    init = fl.random_states(700, R, seed=9)
    want = fl.simulate(graph, rules, init, 9, backend="reference", seed=3, record=3)
    got = fl.simulate(graph, rules, init, 9, backend="cpu", seed=3, record=3, threads=threads)
    assert np.array_equal(got.states.bits, want.states.bits)


@pytest.mark.parametrize("backend", [*BACKENDS, "reference"])
def test_init_must_be_one_configuration_per_replica(backend):
    g, rule = fl.ring(40), fl.life_like(3, 2, 5)
    traj = fl.simulate(g, rule, fl.random_states(40, 2, seed=0), 3, backend="reference")
    with pytest.raises(ValueError, match="final"):
        fl.simulate(g, rule, traj.states, 2, backend=backend)  # [T, ...] frames instead of traj.final()
