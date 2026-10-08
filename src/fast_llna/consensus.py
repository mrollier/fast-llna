"""Consensus times and exact cycle detection per (network, lane) of a union of networks.

The run advances in chunks of :func:`simulate` (continuation with ``t0`` is exact); every recorded frame is
reduced per segment on its packed node rows, without unpacking the lanes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import Graph
from .rules import Rules
from .simulate import simulate
from .states import States


@dataclass(frozen=True, eq=False)
class Consensus:
    """Per (segment, lane) ``[S, R]``, lanes in input order.

    t_consensus
        First timestep at which all nodes of the segment agree; inf if none by ``t_max``.
    state
        The agreed state (0 or 1) at that timestep; -1 if none.
    t_cycle, period
        Deterministic lanes without consensus only: the timestep at which the segment's configuration
        repeated, and the exact minimal period of that cycle; inf and 0 otherwise. A cycle proves the segment
        never reaches consensus.
    """

    t_consensus: np.ndarray
    state: np.ndarray
    t_cycle: np.ndarray
    period: np.ndarray


def _padded(R: int) -> int:
    """Lanes after padding: a power of two up to 32, then whole 64-lane words; node rows are then u8..u64."""
    return 1 << (R - 1).bit_length() if R <= 32 else -(-R // 64) * 64


def _pack(x: np.ndarray, Rp: int) -> np.ndarray:
    """bool [R, N] -> packed bits of Rp lanes, the last Rp - R all 0."""
    return States.from_bool(np.concatenate([x, np.zeros((Rp - len(x), x.shape[1]), bool)])).bits


def _frozen(p: np.ndarray, Rp: int) -> np.ndarray:
    """Rule tables [R, 2, ncell] padded to Rp with the frozen rule (every node keeps its state)."""
    pad = np.zeros((Rp - len(p), *p.shape[1:]))
    pad[:, 1] = 1
    return np.concatenate([p, pad])


def _reduce(op, frames: np.ndarray, starts: np.ndarray, n: int, L: int) -> np.ndarray:
    """``op``-reduce packed frames over the nodes of each segment -> bool [F, S, L]."""
    if L < 8:  # 1, 2 or 4 lanes
        x = np.unpackbits(frames, axis=-1, count=n * L, bitorder="little").reshape(len(frames), n, L)
        return op.reduceat(x, starts, axis=1).view(bool)
    # node rows as the widest uint that fits. ponytail: reduceat runs a scalar loop over rows of several u64
    # words (~0.1 ms per 3 MB frame on an M4); a fold over equal-size segments would vectorise it
    dtype = {8: np.uint8, 16: np.uint16, 32: np.uint32}.get(L, np.uint64)
    red = op.reduceat(frames.view(dtype).reshape(len(frames), n, -1), starts, axis=1)
    return np.unpackbits(red.view(np.uint8), axis=-1, count=L, bitorder="little").view(bool)


def _changed(frames: np.ndarray, ref: np.ndarray, starts: np.ndarray, n: int, L: int) -> np.ndarray:
    """Whether each packed frame differs from ``ref`` in a segment, per lane -> bool [F, S, L]. Frame by
    frame: the XOR is reduced while in cache, and the frames are left intact."""
    tmp = np.empty_like(ref)
    diff = (_reduce(np.bitwise_or, np.bitwise_xor(f, ref, out=tmp)[None], starts, n, L) for f in frames)
    return np.concatenate(list(diff))


def consensus(
    graph: Graph,
    rules: Rules,
    init,
    t_max: int,
    *,
    offsets=None,
    seed: int = 0,
    chunk: int = 32,
    backend: str = "auto",
    threads: int | None = None,
) -> Consensus:
    """Consensus time of every segment (network) under every lane, and for deterministic lanes without
    consensus the exact cycle that proves there is none.

    graph, offsets
        Typically ``fl.union(*networks)``: its graph and offsets (S + 1 ints; no edge may cross segments).
        ``offsets=None`` is one segment of all nodes.
    rules
        One rule (shared) or one rule per lane.
    init
        :class:`States` or bool ``[R, N]``, or bool ``[N]`` for every lane (R = len(rules)).
    t_max
        Last timestep examined (frames 0..t_max).
    seed, backend, threads
        As in :func:`simulate`.
    chunk
        Steps per :func:`simulate` call once t >= chunk; a power of two. Results do not depend on it.

    A lane is deterministic if its rule is 0 or 1 on every (degree, living neighbours) the graph has; only
    those get cycles, stochastic lanes settle only by consensus. Cycles are found Brent-style: the
    configuration at t_ref = 0, 1, 2, 4, ... is compared with the frames t_ref + 1 .. max(2 t_ref, 1). A cycle
    with transient mu and minimal period lambda is found at t_cycle = t_ref + lambda, t_ref the smallest of
    0, 1, 2, 4, ... with t_ref >= mu and lambda <= max(t_ref, 1), whatever the chunk or the lanes. It is found
    only if that t_cycle <= t_max: at t_max = 1e5 a cycle entered after t = 65536 can go unseen.
    """
    N = graph.n
    off = np.array([0, N]) if offsets is None else np.asarray(offsets)
    if off.ndim != 1 or off.dtype.kind not in "iu" or off.size < 2 or off[0] != 0 or off[-1] != N:
        raise ValueError(f"offsets must be 1-D ints from 0 to N = {N}, got {offsets}")
    if np.any(np.diff(off) <= 0):
        raise ValueError(f"offsets must be strictly increasing, got {offsets}")
    S, starts = off.size - 1, off[:-1]
    seg = np.repeat(np.arange(S), np.diff(off))
    if not np.array_equal(seg[graph.indices], np.repeat(seg, graph.degree)):
        raise ValueError("an edge crosses segments: offsets must bound the networks of the union")
    if not isinstance(chunk, (int, np.integer)) or chunk < 1 or chunk & (chunk - 1):
        raise ValueError(f"chunk must be a positive power of two, got {chunk}")
    if not isinstance(t_max, (int, np.integer)) or t_max < 0:
        raise ValueError(f"t_max must be an int >= 0, got {t_max}")
    chunk, t_max = int(chunk), int(t_max)
    x0 = init.to_bool() if isinstance(init, States) else np.asarray(init, bool)
    if x0.ndim == 1:
        x0 = np.broadcast_to(x0, (len(rules), x0.size))
    if x0.ndim != 2 or x0.shape[1] != N:
        raise ValueError(f"init must be [R, {N}] or [{N}], got shape {x0.shape}")
    R = len(x0)
    if len(rules) not in (1, R):
        raise ValueError(f"need 1 or {R} rules (one per lane), got {len(rules)}")

    # behaviour on the reachable cells: deterministic, and quiet (both uniform states are fixed points)
    ks = np.unique(graph.degree)
    tab = np.broadcast_to(rules.exact(ks).p, (R, 2, int(np.sum(ks + 1))))
    q0 = np.cumsum(ks + 1) - ks - 1  # cell (k, q = 0)
    det = np.all((tab == 0) | (tab == 1), axis=(1, 2))
    quiet = np.all(tab[:, 0, q0] == 0, axis=1) & np.all(tab[:, 1, q0 + ks] == 1, axis=1)

    t_cons, state = np.full((S, R), np.inf), np.full((S, R), -1, np.int8)
    t_cyc, period = np.full((S, R), np.inf), np.zeros((S, R), np.int64)
    # current lanes: lane[j] is the input lane of lane j < len(lane), the Rp - len(lane) after it are padding
    lane, Rp = np.arange(R), _padded(R)
    p = _frozen(np.broadcast_to(rules.p, (R, *rules.p.shape[1:])), Rp)
    last = ref = _pack(x0, Rp)
    t = t_ref = 0

    def unsettled():
        return np.isinf(t_cons) & np.isinf(t_cyc)

    def agree(frames):  # bool [F, S, current lanes]: all nodes equal, all alive
        all1 = _reduce(np.bitwise_and, frames, starts, N, Rp)[..., : len(lane)]
        return all1 | ~_reduce(np.bitwise_or, frames, starts, N, Rp)[..., : len(lane)], all1

    def settle(frames, t1):  # first consensus in frames of timesteps t1, t1 + 1, ...
        eq, all1 = agree(frames)
        s, j = np.nonzero(eq.any(0) & unsettled()[:, lane])
        f = eq.argmax(0)[s, j]
        t_cons[s, lane[j]], state[s, lane[j]] = t1 + f, all1[f, s, j]

    settle(last[None], 0)
    while t < t_max and unsettled().any():
        n = min(chunk, t_max - t, max(t, 1))  # every power of two is a chunk end
        x, cur = States(last, Rp, N), Rules(rules.partition, p)
        traj = simulate(graph, cur, x, n, t0=t, seed=seed, backend=backend, threads=threads)
        frames = traj.states.bits[1:]  # timesteps t + 1 .. t + n
        last = frames[-1].copy()  # not a view: the trajectory is freed
        # quiet lanes keep a consensus: then only a chunk that ends in a new one is scanned
        if not quiet[lane].all() or (agree(last[None])[0][0] & unsettled()[:, lane]).any():
            settle(frames, t + 1)

        hunt = det[lane] & unsettled()[:, lane]
        if hunt.any():  # Brent: first return to the reference configuration
            same = ~_changed(frames, ref, starts, N, Rp)[..., : len(lane)]
            s, j = np.nonzero(same.any(0) & hunt)
            f = same.argmax(0)[s, j] + 1
            t_cyc[s, lane[j]], period[s, lane[j]] = t + f, t + f - t_ref

        t += n
        if t & (t - 1) == 0:
            ref, t_ref = last, t
        live = np.flatnonzero(unsettled()[:, lane].any(0))
        if live.size and det[lane].all() and _padded(live.size) <= Rp // 2:  # drop settled lanes
            Rn = _padded(live.size)
            last, ref = (_pack(States(b, Rp, N).to_bool()[live], Rn) for b in (last, ref))
            p, lane, Rp = _frozen(p[live], Rn), lane[live], Rn
    return Consensus(t_cons, state, t_cyc, period)
