"""Reference backend: a direct numpy transcription of the model, independent of the kernel code.

Slow but obviously correct; every other backend must reproduce its output bit for bit.
"""

from __future__ import annotations

import numpy as np

from .rng import quantize, rule_words
from .states import States

ONE = np.uint64(2**32)


def available() -> bool:
    return True


def _digits(seed, t, n, stream, R, depth):
    """Uniform random integers in [0, 2^depth) per (replica, node): the first ``depth`` binary digits of U."""
    nodes = np.arange(n, dtype=np.uint32)[None, :]
    lane = (np.arange(R) % 32).astype(np.uint32)[:, None]
    u = np.zeros((R, n), np.uint64)
    for d in range(depth):
        words = rule_words(seed, np.uint32(t), nodes, stream[:, None], d)  # [W, N]
        u = (u << np.uint64(1)) | ((words[np.arange(R) // 32] >> lane) & 1).astype(np.uint64)
    return u


def run(graph, rules, x0: States, steps, record, clamp, seed, t0, stream, threads=None):
    A = graph.adjacency().astype(np.int64)
    k = graph.degree.astype(np.int64)
    x = x0.to_bool()
    R = x.shape[0]
    replicas = np.arange(R)[:, None]
    P, depth = quantize(rules.p)
    P = P[np.zeros(R, int) if len(rules) == 1 else np.arange(R)]  # [R, 2, ncell]
    frames = [x] if record and record != "final" else []
    for step in range(steps):
        q = (A @ x.T.astype(np.int64)).T
        cB, cS = rules.partition.cell(q, k, 0), rules.partition.cell(q, k, 1)
        p = np.where(x, P[replicas, 1, cS], P[replicas, 0, cB])
        alive = p == ONE
        frac = (p > 0) & (p < ONE)
        if depth and frac.any():
            u = _digits(seed, t0 + step, graph.n, stream, R, depth)
            alive |= frac & (u < (p >> np.uint64(32 - depth)))
        x = alive if clamp is None else np.where(clamp[0], clamp[1], alive)
        if record == "final" and step == steps - 1 or record != "final" and (step + 1) % record == 0:
            frames.append(x)
    if not frames:  # record="final" with steps == 0
        frames = [x]
    return States.from_bool(np.stack(frames)).bits
