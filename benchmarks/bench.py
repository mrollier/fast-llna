"""Throughput benchmarks; one JSON line per measurement (node-updates/s = N * R * T / seconds).

python benchmarks/bench.py --scenario thesis --backend cpu
python benchmarks/bench.py --scenario all --backend cuda > results.jsonl
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from graphs import barabasi_albert, chung_lu, erdos_renyi, rewired_torus  # noqa: E402

import fast_llna as fl  # noqa: E402


def measure(name, graph, rules, init, steps, backend, repeats=3, **kw):
    t = time.perf_counter()
    fl.simulate(graph, rules, init, steps, backend=backend, **kw)
    first = time.perf_counter() - t
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fl.simulate(graph, rules, init, steps, backend=backend, **kw)
        times.append(time.perf_counter() - t)
    steady = float(np.median(times))
    R = init.n_replicas
    row = {
        "scenario": name,
        "backend": backend,
        "N": graph.n,
        "R": R,
        "T": steps,
        "mean_degree": float(graph.degree.mean()),
        "max_degree": int(graph.degree.max()),
        "record": kw.get("record", 1),
        "first_s": round(first, 4),
        "steady_s": round(steady, 4),
        "updates_per_s": graph.n * R * steps / steady,
        "machine": platform.platform(),
    }
    print(json.dumps(row), flush=True)
    return row


def thesis(backend):
    """528 non-equivalent r=5 rules x 60 inits on a rewired Moore torus (N=900, p=0.2), T=100.
    The llna torch engine needs 35 s for this on the M4 CPU."""
    g = rewired_torus(30, 0.2, seed=0)
    rules, init = fl.product(fl.life_like(5, *fl.nonequivalent(5).T), fl.random_states(900, 60, seed=0))
    for record in (1, "final"):
        measure("thesis", g, rules, init, 100, backend, record=record)


def scale_R(backend):
    g = rewired_torus(30, 0.2, seed=1)
    for R in (32, 1024, 32768, 1 << 20):
        rules = fl.life_like(5, *np.random.default_rng(R).integers(0, 32, size=(2, R)))
        measure("scale_R", g, rules, fl.random_states(900, R, seed=R), 100, backend, record="final")


def scale_N(backend):
    for N in (100, 1000, 10_000, 100_000):
        g = erdos_renyi(N, 8, seed=N)
        measure(
            "scale_N",
            g,
            fl.life_like(5, 6, 28),
            fl.random_states(N, 1024, seed=N),
            100,
            backend,
            record="final",
        )


def huge_N(backend):
    for N in (100_000, 1_000_000, 10_000_000):
        g = erdos_renyi(N, 8, seed=N)
        for R in (1, 32):
            measure(
                "huge_N",
                g,
                fl.life_like(5, 6, 28),
                fl.random_states(N, R, seed=R),
                100,
                backend,
                repeats=1,
                record="final",
            )


def _random(n, R, seed):
    return fl.States.from_bool(np.random.default_rng(seed).random((R, n)) < 0.5)  # fast for huge N


def small_R(backend):
    """Few replicas, N up to 1e7: the narrow layout (L = pow2ceil(R) bits per node) and cooperative hubs."""
    for N in (1_000, 100_000, 1_000_000, 10_000_000):
        g = erdos_renyi(N, 8, seed=N)
        steps, repeats = (20, 1) if N >= 1_000_000 else (100, 3)
        for R in (1, 2, 4, 8, 16, 32):
            measure("small_R", g, fl.life_like(5, 6, 28), _random(N, R, R), steps, backend, repeats, record="final")
    for name, g in (("power_law", chung_lu(10_000_000, 2.5, 8, seed=0)), ("moore", fl.moore_torus(3163, 3163))):
        measure(f"small_R_{name}", g, fl.life_like(5, 6, 28), _random(g.n, 1, 1), 20, backend, 1, record="final")


def hubs(backend):
    for name, g in (("ER", erdos_renyi(10_000, 8, seed=2)), ("BA", barabasi_albert(10_000, 4, seed=2))):
        measure(
            f"hubs_{name}",
            g,
            fl.life_like(9, 72, 12),
            fl.random_states(g.n, 1024, seed=2),
            100,
            backend,
            record="final",
        )


def overheads(backend):
    g = rewired_torus(30, 0.2, seed=3)
    init = fl.random_states(900, 4096, seed=3)
    measure("deterministic", g, fl.life_like(9, 72, 12), init, 100, backend, record="final")
    measure("majority", g, fl.majority(), init, 100, backend, record="final")
    clamp = (np.random.default_rng(3).random((4096, 900)) < 0.01, True)
    measure("clamp", g, fl.life_like(9, 72, 12), init, 100, backend, record="final", clamp=clamp)


def reducers(backend):
    g = rewired_torus(30, 0.2, seed=0)
    rules, init = fl.product(fl.life_like(5, *fl.nonequivalent(5).T), fl.random_states(900, 60, seed=0))
    traj = fl.simulate(g, rules, init, 100, backend=backend)
    for what in ("density", "hamming"):
        t = time.perf_counter()
        traj.density() if what == "density" else traj.hamming(
            np.c_[np.arange(0, 31680, 2), np.arange(1, 31680, 2)]
        )
        print(json.dumps({"scenario": f"reducer_{what}", "seconds": round(time.perf_counter() - t, 4)}))


SCENARIOS = {f.__name__: f for f in (thesis, scale_R, scale_N, huge_N, small_R, hubs, overheads, reducers)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="thesis", choices=[*SCENARIOS, "all"])
    ap.add_argument("--backend", default="auto")
    a = ap.parse_args()
    backend = fl.available_backends()[0] if a.backend == "auto" else a.backend
    for name, f in SCENARIOS.items():
        if a.scenario in (name, "all"):
            f(backend)
