"""Every compiled backend must reproduce the reference backend bit for bit."""

import numpy as np
import pytest
from _graphs import random_digraph, star_hub

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
def test_cpu_modes_and_thread_counts(mode, threads, monkeypatch):
    monkeypatch.setenv("FAST_LLNA_CPU_MODE", mode)
    graph, R = random_digraph(70, 1, 15, 9), 300
    rules = stochastic(fl.symmetric(5), R, 9)
    init = fl.random_states(70, R, seed=9)
    want = fl.simulate(graph, rules, init, 9, backend="reference", seed=3, record=3)
    got = fl.simulate(graph, rules, init, 9, backend="cpu", seed=3, record=3, threads=threads)
    assert np.array_equal(got.states.bits, want.states.bits)
