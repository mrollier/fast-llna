"""Consensus sweep: quiescent self-equivalent Life-like rules of resolution r = 5..10 vs Watts' majority (coin
at rho = 1/2) on Watts-Strogatz networks (N = 1000, K = 8), rewiring p = 0 and 20 values from 1e-3 to 1.

    python notebooks/consensus_sweep.py [--smoke] [--stage discovery|select|validation|all] [--p I ...]

Stages (``all`` runs the three in order; existing files are skipped, so a rerun resumes after an interruption;
keep the Mac awake with ``caffeinate -i``):

discovery
    Per p: dedupe all codes into behaviours (equal exact tables on the degrees of the discovery AND
    validation networks), run one representative per behaviour (its first code: lowest r, then "+-", then
    smallest beta) on the S = 100 discovery networks, and majority (seed 2) in its own call.
select
    Per (r, p): the top 3 behaviours that contain a code of resolution r, by runs with consensus by T_MAX
    (at least 1), then by 1e4, then by the median consensus time of the reached runs; each is named by its
    first code of resolution r. Plus the tracked codes. Over 256 codes: top 1 per (r, p), the top 3 per p
    over all r, and the tracked codes. Reruns only if validation_set.npz is missing.
validation
    Per p: the validation set as one merged exact-partition call on V = 500 fresh networks, and majority
    (seed 12) on the same networks and init (paired).

``--smoke``: N = 200, S = 4, V = 8, p in (0, 0.01, 1), T_MAX = 2000, into notebooks/results_smoke/.
``--p I ...``: only these p indices (e.g. ``--stage discovery --p 12`` to time one p).

Files in notebooks/results/ (all arrays int64 / float64 unless noted; scalars are 0-d arrays). A code is
one row (r, conv, beta, sigma) with conv 0 = "+-", 1 = "-+" (odd r: always 0); codes are ordered by r, conv,
beta. Times are timesteps as float64, inf = not reached by T_MAX. ``state`` is int8: the agreed state 0/1,
-1 if none. ``t_cycle``/``period``: deterministic runs without consensus whose configuration provably
repeats (inf / 0 otherwise); a run with neither consensus nor cycle is "open" at T_MAX.

disc_pII.npz (II = p index 00..20)
    p, i                          the rewiring probability P[i] and its index
    N, K, S, V, T_MAX             config (V: validation networks, whose degrees enter ``degrees``)
    net_seeds [S]                 network s is connected_watts_strogatz_graph(N, K, p, seed=net_seeds[s])
    seed_init, seed_rules, seed_majority   network s starts from random_states(N, S, seed_init).to_bool()[s]
    codes [n_codes, 4]            all quiescent self-equivalent codes (1680 at r = 5..10)
    behaviour [n_codes]           behaviour id of each code (ids ordered by representative)
    rep [n_beh]                   code index of each behaviour's representative (ascending)
    degrees [n_deg] int32         degrees of the exact tables (union over discovery and validation networks)
    t_consensus, t_cycle [S, n_beh] float64; state [S, n_beh] int8; period [S, n_beh]
                                  per behaviour; per code: t_consensus[:, behaviour]
    maj_t_consensus [S] float64; maj_state [S] int8
    group_r, group_conv, group_lanes, group_wall [n_groups]   one consensus call per (r, conv); wall in s
    maj_wall                      wall time of the majority call, s
validation_set.npz
    codes [n_valid, 4]            the validation set, in code order
    tracked [n_valid] bool        the code is one of TRACKED
    pick_code, pick_r, pick_i, pick_rank [n_picks]   pick n chose codes[pick_code[n]] as rank pick_rank
                                  (1..3) at p index pick_i among behaviours with a code of resolution pick_r
                                  (pick_r = 0: over all r, named by the representative)
    fallback bool                 the full union exceeded MAX_VALID: only rank-1 and pick_r = 0 picks are kept
    n_union                       size of the full union (picks of every rank plus tracked)
valid_pII.npz
    p, i, N, K, V, T_MAX, net_seeds [V], seed_init, seed_rules, seed_majority
    codes [n_valid, 4]            equal to validation_set.npz codes (lanes in this order)
    degrees [n_deg] int32         degrees of the validation union
    t_consensus, t_cycle [V, n_valid] float64; state [V, n_valid] int8; period [V, n_valid]
    maj_t_consensus [V] float64; maj_state [V] int8
    wall, maj_wall                wall times of the two calls, s
"""

import argparse
import os
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

import fast_llna as fl

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
N, K = 1000, 8  # Watts-Strogatz: N nodes on a ring with K neighbours each, rewired with probability p
P = np.r_[0.0, np.geomspace(1e-3, 1, 20)]
S, V = 100, 500  # discovery and validation networks per p (the same seeds at every p)
VALID_SEED0 = 1000  # validation network v has seed VALID_SEED0 + v
T_MAX, T_EARLY = 100_000, 10_000
SEED_INIT, SEED_INIT_VALID = 1, 11
SEED_RULES = 0  # deterministic rules: irrelevant
SEED_MAJ, SEED_MAJ_VALID = 2, 12
RESOLUTIONS = range(5, 11)
CONV = ("+-", "-+")
TRACKED = [(8, 0, 244, 208), (9, 0, 408, 460), (9, 0, 424, 468), (8, 0, 240, 240), (8, 1, 240, 240)]
TOP, MAX_VALID = 3, 256
RESULTS = ("t_consensus", "state", "t_cycle", "period")


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def all_codes():
    """int64 [n, 4] rows (r, conv, beta, sigma): quiescent self-equivalent codes, ordered by r, conv, beta."""
    out = []
    for r in RESOLUTIONS:
        se = fl.self_equivalent(r)
        se = se[se[:, 0] % 2 == 0]
        for conv in (0, 1) if r % 2 == 0 else (0,):
            out.append(np.column_stack([np.full((len(se), 2), (r, conv)), se]))
    return np.concatenate(out)


def groups(codes):
    """Index arrays of the runs of equal (r, conv) in ``codes`` (rows in code order)."""
    _, start = np.unique(codes[:, :2], axis=0, return_index=True)
    return np.split(np.arange(len(codes)), np.sort(start)[1:])


def rules_of(codes):
    """Rules of codes that share (r, conv), on their own symmetric partition."""
    r, conv = (int(x) for x in codes[0, :2])
    return fl.life_like(r, codes[:, 2], codes[:, 3], partition=fl.symmetric(r, CONV[conv]))


def networks(p, seeds):
    ws = (nx.connected_watts_strogatz_graph(N, K, p, seed=s) for s in seeds)
    return [fl.Graph(nx.to_scipy_sparse_array(g)) for g in ws]


def behaviours(codes, degrees):
    """Behaviour id per code and the representative code of each behaviour (ids ordered by representative)."""
    table = np.concatenate([rules_of(codes[g]).exact(degrees).p for g in groups(codes)])
    flat = table.reshape(len(codes), -1)
    _, first, inverse = np.unique(flat, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rep, behaviour = first[order], np.argsort(order)[inverse.ravel()]
    assert np.array_equal(behaviour[rep], np.arange(len(rep)))
    assert np.array_equal(flat[rep[behaviour]], flat)  # each code has exactly its representative's table
    for r in RESOLUTIONS[1::2]:  # even r: bits r/2 - 1 and r/2 of beta agree -> same rule in both conventions
        plus, minus = (np.flatnonzero((codes[:, 0] == r) & (codes[:, 1] == c)) for c in (0, 1))
        beta = codes[plus, 2]
        assert np.array_equal(beta, codes[minus, 2])
        shared = (beta >> (r // 2 - 1) & 1) == (beta >> (r // 2) & 1)
        assert shared.any() and np.array_equal(behaviour[plus[shared]], behaviour[minus[shared]])
    return behaviour, rep


def outcome(c):
    """'consensus a, cycle b, open c' over all (network, lane) runs of a Consensus."""
    n_con, n_cyc = np.isfinite(c.t_consensus).sum(), np.isfinite(c.t_cycle).sum()
    return f"consensus {n_con}, cycle {n_cyc}, open {c.t_consensus.size - n_con - n_cyc}"


def config(seeds, seed_init, seed_majority):
    return dict(N=N, K=K, T_MAX=T_MAX, net_seeds=np.asarray(seeds), seed_init=seed_init,
                seed_rules=SEED_RULES, seed_majority=seed_majority)


def majority(graph, offsets, x0, seed):
    start = time.perf_counter()
    m = fl.consensus(graph, fl.majority(0.5), x0, T_MAX, offsets=offsets, seed=seed)
    wall = time.perf_counter() - start
    log(f"  majority: {wall:7.1f} s  {outcome(m)}")
    return dict(maj_t_consensus=m.t_consensus[:, 0], maj_state=m.state[:, 0], maj_wall=wall)


def discovery(i):
    p, codes = P[i], all_codes()
    nets = networks(p, range(S))
    valid_degrees = [g.degree for g in networks(p, range(VALID_SEED0, VALID_SEED0 + V))]
    degrees = np.unique(np.concatenate([g.degree for g in nets] + valid_degrees))
    behaviour, rep = behaviours(codes, degrees)
    if p == 0:  # ring lattice, one degree: a quiescent self-equivalent table is set by its K births at q > 0
        assert np.array_equal(degrees, [K]) and len(rep) <= 2**K
    log(f"disc p{i:02d} = {p:.4g}: degrees {degrees.min()}..{degrees.max()}, "
        f"{len(codes)} codes -> {len(rep)} behaviours")

    graph, offsets = fl.union(*nets)
    x0 = fl.random_states(N, S, seed=SEED_INIT).to_bool().ravel()  # network s starts from row s
    dtypes = (float, np.int8, float, np.int64)
    res = {key: np.zeros((S, len(rep)), dtype) for key, dtype in zip(RESULTS, dtypes, strict=True)}
    walls = []
    for g in groups(codes[rep]):  # g: behaviour ids whose representatives share (r, conv)
        r, conv = codes[rep[g[0]], :2]
        start = time.perf_counter()
        c = fl.consensus(graph, rules_of(codes[rep[g]]), x0, T_MAX, offsets=offsets, seed=SEED_RULES)
        walls.append((r, conv, len(g), time.perf_counter() - start))
        for key in RESULTS:
            res[key][:, g] = getattr(c, key)
        log(f"  r={r:2d} {CONV[conv]}: {len(g):4d} lanes {walls[-1][3]:7.1f} s  {outcome(c)}")
    group_r, group_conv, group_lanes, group_wall = (np.array(x) for x in zip(*walls, strict=True))
    return dict(p=p, i=i, S=S, V=V, **config(range(S), SEED_INIT, SEED_MAJ), codes=codes, behaviour=behaviour,
                rep=rep, degrees=degrees, **res, **majority(graph, offsets, x0, SEED_MAJ), group_r=group_r,
                group_conv=group_conv, group_lanes=group_lanes, group_wall=group_wall)


def select(path):
    """Winners per (r, p) over the discovery files, plus the tracked codes: the validation set."""
    codes, picks = all_codes(), []  # pick: (code index, r or 0 = over all r, p index, rank 1..TOP)
    for i in range(len(P)):
        d = np.load(OUT / f"disc_p{i:02d}.npz")
        assert np.array_equal(d["codes"], codes)
        t, beh, rep = d["t_consensus"], d["behaviour"], d["rep"]
        n_max, n_early = np.isfinite(t).sum(0), (t <= T_EARLY).sum(0)
        med, ok = np.full(len(rep), np.inf), n_max > 0
        med[ok] = np.nanmedian(np.where(np.isfinite(t[:, ok]), t[:, ok], np.nan), axis=0)
        ranked = np.lexsort((med, -n_early, -n_max))  # primary key last
        ranked = ranked[n_max[ranked] > 0]
        picks += [(rep[b], 0, i, k) for k, b in enumerate(ranked[:TOP], 1)]
        for r in RESOLUTIONS:
            idx = np.flatnonzero(codes[:, 0] == r)
            has_r, j = np.unique(beh[idx], return_index=True)  # behaviours with a code of r, their first one
            top = ranked[np.isin(ranked, has_r)][:TOP]
            picks += [(idx[j[np.searchsorted(has_r, b)]], r, i, k) for k, b in enumerate(top, 1)]
    picks = np.array(picks, dtype=np.int64).reshape(-1, 4)
    tracked = np.array([np.flatnonzero((codes == c).all(1))[0] for c in TRACKED])
    n_union = len(np.union1d(picks[:, 0], tracked))
    if n_union > MAX_VALID:
        picks = picks[(picks[:, 1] == 0) | (picks[:, 3] == 1)]
    chosen = np.union1d(picks[:, 0], tracked)
    assert len(chosen) <= MAX_VALID
    save(path, codes=codes[chosen], tracked=np.isin(chosen, tracked),
         pick_code=np.searchsorted(chosen, picks[:, 0]), pick_r=picks[:, 1], pick_i=picks[:, 2],
         pick_rank=picks[:, 3], fallback=n_union > MAX_VALID, n_union=n_union)
    per_r = ", ".join(f"r={r}: {np.sum(codes[chosen, 0] == r)}" for r in RESOLUTIONS)
    log(f"select: {len(chosen)} codes ({per_r}); union {n_union}, fallback {n_union > MAX_VALID}")


def validation(i, codes):
    p, nets = P[i], networks(P[i], range(VALID_SEED0, VALID_SEED0 + V))
    graph, offsets = fl.union(*nets)
    x0 = fl.random_states(N, V, seed=SEED_INIT_VALID).to_bool().ravel()
    degrees = np.unique(graph.degree)
    tables = [rules_of(codes[g]).exact(degrees) for g in groups(codes)]
    rules = fl.Rules(tables[0].partition, np.concatenate([t.p for t in tables]))
    log(f"valid p{i:02d} = {p:.4g}: degrees {degrees.min()}..{degrees.max()}, {len(codes)} lanes")
    start = time.perf_counter()
    c = fl.consensus(graph, rules, x0, T_MAX, offsets=offsets, seed=SEED_RULES)
    wall = time.perf_counter() - start
    log(f"  merged: {len(codes):4d} lanes {wall:7.1f} s  {outcome(c)}")
    return dict(p=p, i=i, V=V, **config(range(VALID_SEED0, VALID_SEED0 + V), SEED_INIT_VALID, SEED_MAJ_VALID),
                codes=codes, degrees=degrees, **{key: getattr(c, key) for key in RESULTS},
                **majority(graph, offsets, x0, SEED_MAJ_VALID), wall=wall)


def save(path, **arrays):
    """Atomic: a crash leaves either the old file or the new one, never half of one."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:  # through a handle: np.savez would append .npz to a name
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def stage(prefix, compute, indices):
    """compute(i) and save it for every p index whose file is missing; log progress and an ETA."""
    todo = [i for i in indices if not (OUT / f"{prefix}_p{i:02d}.npz").exists()]
    log(f"{prefix}: {len(indices) - len(todo)} of {len(indices)} files exist, computing {todo}")
    start = time.perf_counter()
    for n, i in enumerate(todo, 1):
        save(OUT / f"{prefix}_p{i:02d}.npz", **compute(i))
        took = time.perf_counter() - start
        log(f"{prefix}: saved p{i:02d} ({n}/{len(todo)}), {took / 60:.1f} min so far, "
            f"ETA {took / n * (len(todo) - n) / 60:.0f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true", help="small end-to-end run into notebooks/results_smoke/")
    ap.add_argument("--stage", choices=("discovery", "select", "validation", "all"), default="all")
    ap.add_argument("--p", type=int, nargs="+", help="only these p indices (default: all)")
    args = ap.parse_args()
    if args.smoke:
        N, S, V, P, T_MAX, OUT = 200, 4, 8, np.array([0.0, 0.01, 1.0]), 2000, HERE / "results_smoke"
    indices = range(len(P)) if args.p is None else args.p
    OUT.mkdir(exist_ok=True)
    log(f"N={N} K={K} S={S} V={V} T_MAX={T_MAX} p={np.round(P, 5).tolist()} -> {OUT} "
        f"(backends {fl.available_backends()})")

    if args.stage in ("discovery", "all"):
        stage("disc", discovery, indices)
    if args.stage in ("select", "all"):
        path = OUT / "validation_set.npz"
        missing = [i for i in range(len(P)) if not (OUT / f"disc_p{i:02d}.npz").exists()]
        if path.exists():
            log(f"select: {path.name} exists, kept")
        elif missing:
            sys.exit(f"select needs every discovery file; missing p indices {missing}")
        else:
            select(path)
    if args.stage in ("validation", "all"):
        path = OUT / "validation_set.npz"
        if not path.exists():
            sys.exit(f"validation needs {path}: run --stage select first")
        codes = np.load(path)["codes"]
        for f in sorted(OUT.glob("valid_p*.npz")):
            if not np.array_equal(np.load(f)["codes"], codes):
                sys.exit(f"{f} was run on another validation set: delete it (or validation_set.npz) to rerun")
        stage("valid", lambda i: validation(i, codes), indices)

    for f in sorted(OUT.glob("*.npz")):
        d = np.load(f)
        log(f"{f.name}: " + " ".join(f"{k}{list(d[k].shape)}" for k in d.files if d[k].ndim))
