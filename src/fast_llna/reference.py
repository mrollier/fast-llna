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


def _digits(seed, t, n, R, depth, noise_period):
    """Uniform random integers in [0, 2^depth) per (replica, node): the first ``depth`` binary digits of U.
    Replica r uses the digits of replica r mod noise_period (rng module contract)."""
    nodes = np.arange(n, dtype=np.uint32)[None, :]
    src = np.arange(R) if noise_period is None else np.arange(R) % noise_period
    streams, which = np.unique(src // 32, return_inverse=True)
    lane = (src % 32).astype(np.uint32)[:, None]
    u = np.zeros((R, n), np.uint64)
    for d in range(depth):
        words = rule_words(seed, np.uint32(t), nodes, streams.astype(np.uint32)[:, None], d)  # [S, N]
        u = (u << np.uint64(1)) | ((words[which] >> lane) & 1).astype(np.uint64)
    return u


def run(graph, rules, x0: States, steps, record, clamp, seed, t0, noise_period, threads=None):
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
            u = _digits(seed, t0 + step, graph.n, R, depth, noise_period)
            alive |= frac & (u < (p >> np.uint64(32 - depth)))
        x = alive if clamp is None else np.where(clamp[0], clamp[1], alive)
        if record == "final" and step == steps - 1 or record != "final" and (step + 1) % record == 0:
            frames.append(x)
    if not frames:  # record="final" with steps == 0
        frames = [x]
    return States.from_bool(np.stack(frames)).bits
