"""Front end: validate inputs, pick a backend, run, wrap the result."""

from __future__ import annotations

import importlib
import os

import numpy as np

from .graph import Graph
from .rules import Rules
from .states import States, Trajectory

BACKENDS = ("cuda", "metal", "cpu", "reference")  # auto picks the first available


def _module(name):
    return importlib.import_module(
        f".{'reference' if name == 'reference' else 'kernels.' + name}", __package__
    )


def available_backends() -> list[str]:
    out = []
    for name in BACKENDS:
        try:
            if _module(name).available():
                out.append(name)
        except ImportError:
            pass
    return out


def simulate(
    graph: Graph,
    rules: Rules,
    init,
    steps: int,
    *,
    record=1,
    clamp=None,
    seed: int = 0,
    t0: int = 0,
    noise_period: int | None = None,
    backend: str = "auto",
    threads: int | None = None,
    max_bytes: float = 4e9,
) -> Trajectory:
    """Evolve R replicas for ``steps`` synchronous updates and return the recorded trajectory.

    graph
        :class:`Graph`; node i reads its in-neighbours (row i of the adjacency matrix).
    rules
        :class:`Rules` with one rule (shared by all replicas) or one rule per replica.
    init
        :class:`States` or boolean array ``[R, N]``.
    record
        ``n`` records timesteps t0, t0 + n, ..., t0 + steps (``steps`` must be a multiple of n);
        ``"final"`` records only t0 + steps.
    clamp
        ``(mask, value)``, boolean arrays ``[N]`` or ``[R, N]``: where mask is set, the node is forced to
        value at every timestep, including the initial one.
    seed, t0
        Randomness depends only on (seed, timestep, node, replica word), so a run of a + b steps equals a run
        of a steps continued with ``t0=a``.
    noise_period
        Replicas r and r + noise_period draw the same random numbers (e.g. defect twins); a multiple of 32.
    backend
        "auto" or one of :data:`BACKENDS`; the environment variable FAST_LLNA_BACKEND overrides "auto".
    max_bytes
        Refuse to allocate a recorded trajectory larger than this.
    """
    x0 = init if isinstance(init, States) else States.from_bool(init)
    R, N = x0.n_replicas, graph.n
    if x0.n != N:
        raise ValueError(f"initial states have {x0.n} nodes, graph has {N}")
    if len(rules) not in (1, R):
        raise ValueError(f"need 1 or {R} rules (one per replica), got {len(rules)}")
    if steps < 0:
        raise ValueError("steps must be >= 0")
    if record == "final":
        times = np.array([t0 + steps])
    elif isinstance(record, int) and record >= 1:
        if steps % record:
            raise ValueError(f"steps ({steps}) must be a multiple of record ({record})")
        times = t0 + np.arange(0, steps + 1, record)
    else:
        raise ValueError("record must be a positive int or 'final'")
    nbytes = len(times) * N * ((R + 7) // 8)
    if nbytes > max_bytes:
        raise ValueError(
            f"recorded trajectory would need {nbytes:.3g} bytes (> max_bytes={max_bytes:.3g}); "
            "record fewer timesteps (record=n or 'final') or raise max_bytes"
        )
    if clamp is not None:
        mask, value = (np.broadcast_to(np.asarray(a, bool), (R, N)) for a in clamp)
        clamp = (mask, value)
        x0 = States.from_bool(np.where(mask, value, x0.to_bool()))
    W = (R + 31) // 32
    stream = np.arange(W, dtype=np.uint32)
    if noise_period is not None:
        if noise_period <= 0 or noise_period % 32:
            raise ValueError("noise_period must be a positive multiple of 32")
        stream %= np.uint32(noise_period // 32)
    if backend == "auto":
        backend = os.environ.get("FAST_LLNA_BACKEND") or available_backends()[0]
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")
    bits = _module(backend).run(graph, rules, x0, steps, record, clamp, seed, t0, stream, threads)
    return Trajectory(States(bits, R), times)
