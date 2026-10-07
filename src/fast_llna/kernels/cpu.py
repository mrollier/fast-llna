"""CPU backend: llna.h compiled by the system C compiler, called through ctypes from a thread pool.

Two parallel modes (FAST_LLNA_CPU_MODE overrides the choice):
  words  each task owns a slice of replica words and runs all steps without synchronisation; with one word
         per node (R <= 256) that is a single task, chosen for graphs below ~8e5 edges where the per-step
         join of nodes mode costs more than it saves;
  nodes  every step is split into node ranges, joined before the next step (few replicas, huge graphs).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np

from ..states import frame_bytes
from . import tables

PRELUDE = """
typedef unsigned int u32;
typedef long llna_idx;
typedef u32 WORD __attribute__((vector_size(4 * VW), aligned(4)));
#define LLNA_FN static inline
#define LLNA_PTR(T) const T *
#define MULHI(a, b) ((u32)(((unsigned long)(a) * (unsigned long)(b)) >> 32))
#define ZEROW ((WORD){0})
#define LOADW(ptr, off) (*(const WORD *)((ptr) + (off)))
#define ANYW(x) llna_any(x)
#define LLNA_LANE0(x) ((x)[0])
#define RANDW(k0, k1, t, i, st, w, d) llna_randw(k0, k1, t, i, st, w, d)
static inline int llna_any(WORD x);
static inline WORD llna_randw(u32 k0, u32 k1, u32 t, int i, const u32 *st, llna_idx w, int d);
"""

WRAPPER = """
#include <string.h>
static inline int llna_any(WORD x) {
    u32 a = 0;
    for (int j = 0; j < VW; j++) a |= x[j];
    return a != 0;
}
static inline WORD llna_randw(u32 k0, u32 k1, u32 t, int i, const u32 *st, llna_idx w, int d) {
    WORD r;
    for (int j = 0; j < VW; j++) r[j] = llna_rand(k0, k1, t, (u32)i, st[w + j], d);
    return r;
}
/* Run nsteps updates, ping-ponging between buffers a and b (step s reads a if s is even), of the task's
   share of the state: output words [i0, i1) (32 / L nodes each) in the narrow layout, else nodes [i0, i1) x
   words [w0, w1) of the [n][Wp] layout. If rec > 0, after every rec-th step the task's bytes of the new state
   are copied into frame frame0 + (s + 1) / rec - 1 of frames ([F][frame bytes]; a frame of the word layout
   is the rows [n][nb]). */
void llna_cpu(const int *indptr, const int *indices, const int *seg_off, const int *seg_thr,
              const int *seg_cell, const u32 *one, const u32 *hasf, const u32 *frac, const u32 *stream,
              const u32 *cm, const u32 *cv, u32 *a, u32 *b, unsigned char *frames, llna_idx n, llna_idx nb,
              llna_idx Wp, u32 k0, u32 k1, u32 t0, llna_idx i0, llna_idx i1, llna_idx w0, llna_idx w1,
              int nsteps, int rec, llna_idx frame0) {
    for (int s = 0; s < nsteps; s++) {
        const u32 *cur = (s & 1) ? b : a;
        u32 *nxt = (s & 1) ? a : b;
        u32 t = t0 + (u32)s;
        int record = rec > 0 && (s + 1) % rec == 0;
        llna_idx f = record ? frame0 + (s + 1) / rec - 1 : 0;
#if LLNA_L < 32
        for (llna_idx v = i0; v < i1; v++) {
            u32 acc = 0;
            for (llna_idx i = v * (32 / LLNA_L); i < (v + 1) * (32 / LLNA_L) && i < n; i++) {
                WORD o = llna_update(cur, indptr, indices, seg_off, seg_thr, seg_cell, one, hasf, frac, stream,
                                     cm, cv, Wp, k0, k1, t, (int)i, 0, 0, 1);
                acc |= o[0] << ((i * LLNA_L) & 31);
            }
            nxt[v] = acc;
        }
        if (record) {
            llna_idx fb = (n * LLNA_L + 7) / 8, c0 = 4 * i0, c1 = 4 * i1 < fb ? 4 * i1 : fb;
            if (c1 > c0) memcpy(frames + f * fb + c0, (const unsigned char *)nxt + c0, c1 - c0);
        }
#else
        for (llna_idx i = i0; i < i1; i++)
            for (llna_idx w = w0; w < w1; w += VW) {
                WORD o = llna_update(cur, indptr, indices, seg_off, seg_thr, seg_cell, one, hasf, frac, stream,
                                     cm, cv, Wp, k0, k1, t, (int)i, w, 0, 1);
                memcpy(nxt + i * Wp + w, &o, sizeof o);
            }
        if (record) {
            llna_idx c0 = 4 * w0, c1 = 4 * w1 < nb ? 4 * w1 : nb;
            for (llna_idx i = i0; c1 > c0 && i < i1; i++)
                memcpy(frames + (f * n + i) * nb + c0, (const unsigned char *)(nxt + i * Wp) + c0, c1 - c0);
        }
#endif
    }
}
"""


def _compiler():
    return os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def available() -> bool:
    return _compiler() is not None


def _cache_dir() -> Path:
    base = os.environ.get("FAST_LLNA_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "fast_llna"
    )
    return Path(base)


@cache
def _machine_id(cc: str) -> str:
    version = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout
    cpu = platform.processor()
    for probe in (
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        ["grep", "-m1", "model name", "/proc/cpuinfo"],
    ):
        try:
            cpu += subprocess.run(probe, capture_output=True, text=True).stdout
        except OSError:
            pass
    return version + cpu + platform.machine()


@cache
def _load(source: str) -> ctypes.CDLL:
    cc = _compiler()
    arch = "-mcpu=native" if platform.machine() in ("arm64", "aarch64") else "-march=native"
    flags = ["-O3", arch, "-std=c11", "-shared", "-fPIC", "-w"]
    key = hashlib.sha256((source + " ".join(flags) + _machine_id(cc)).encode()).hexdigest()[:24]
    path = _cache_dir() / f"llna_{key}.so"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            src, out = Path(tmp) / "llna.c", Path(tmp) / "llna.so"
            src.write_text(source)
            res = subprocess.run([cc, *flags, "-o", str(out), str(src)], capture_output=True, text=True)
            if res.returncode:
                raise RuntimeError(f"compiling the CPU kernel failed:\n{res.stderr}")
            os.replace(out, path)  # atomic: concurrent jobs never see a partial file
    return ctypes.CDLL(str(path))


def _threads(threads):
    if threads:
        return threads
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def _ranges(stop, unit, parts):
    edges = np.unique(np.linspace(0, stop, min(stop, parts) + 1).astype(int)) * unit
    return list(zip(edges[:-1], edges[1:], strict=True))


def run(graph, rules, x0, steps, record, clamp, seed, t0, noise_period, threads=None):
    R, N = x0.n_replicas, graph.n
    W = (R + 31) // 32
    VW = min(8, 1 << (W - 1).bit_length())
    Wp = -(-W // VW) * VW
    tab = tables.build(graph, rules, x0, clamp, seed, noise_period, Wp)
    tab.defines["VW"] = VW
    narrow = tab.Lk < 32
    fn = _load(tab.source(PRELUDE, WRAPPER)).llna_cpu
    fn.restype = None

    nb = (R + 7) // 8
    rec = 0 if record == "final" else record
    out = np.empty((steps // rec + 1 if rec else 1, frame_bytes(N, R)), np.uint8)
    out[0] = x0.bits
    bufs = [tab.arrays["state"], np.empty_like(tab.arrays["state"])]
    nw = bufs[0].size  # narrow: output words per step

    # ctypes converts arguments under the GIL, so every task's argument tuple is built once; the per-step
    # scalars are shared objects whose .value is set before each step's tasks are submitted
    L_, U, C = ctypes.c_long, ctypes.c_uint32, ctypes.c_int
    fixed = [_ptr(tab.arrays[k]) for k in tables.ARRAYS]
    consts = [_ptr(out), L_(N), L_(nb), L_(Wp), U(tab.k0), U(tab.k1)]
    t_, rec_, frame0_ = U(0), C(0), L_(0)

    def calls(parts, nsteps, parity):  # argument tuples of one llna_cpu call per part, reading bufs[parity]
        a, b = bufs[parity], bufs[1 - parity]
        return [
            (*fixed, _ptr(a), _ptr(b), *consts, t_, L_(i0), L_(i1), L_(w0), L_(w1), C(nsteps), rec_, frame0_)
            for i0, i1, w0, w1 in parts
        ]

    nthreads = _threads(threads)
    groups = Wp // VW
    # nodes mode pays ~0.3 ms of thread synchronisation per step: worth it only for big steps (M4-measured:
    # from ~8e5 edges with one word per node; the multi-word bound is the v1 estimate, nodes mode can win
    # earlier, e.g. N=3e4, R=256)
    big = groups >= 2 * nthreads or (graph.indices.size < 800_000 if Wp == 1 else N * Wp < 1 << 21)
    mode = os.environ.get("FAST_LLNA_CPU_MODE") or ("words" if big else "nodes")
    with ThreadPoolExecutor(nthreads) as pool:
        if mode == "words":  # each task owns a word slice for all steps (narrow: one task)
            if narrow:
                parts = [(0, nw, 0, 1)]
            else:
                parts = [(0, N, w0, w1) for w0, w1 in _ranges(groups, VW, 4 * nthreads)]
            t_.value, rec_.value, frame0_.value = t0, rec, 1
            list(pool.map(lambda a: fn(*a), calls(parts, steps, 0)))
        else:  # every step split over node ranges (narrow: word ranges aligned to 128-byte lines)
            if narrow:
                parts = [(i0, min(i1, nw), 0, 1) for i0, i1 in _ranges(-(-nw // 32), 32, 4 * nthreads)]
            else:
                parts = [(i0, i1, 0, Wp) for i0, i1 in _ranges(N, 1, 4 * nthreads)]
            step = [calls(parts, 1, 0), calls(parts, 1, 1)]
            for s in range(steps):
                t_.value = t0 + s
                rec_.value = bool(rec) and (s + 1) % rec == 0
                frame0_.value = (s + 1) // rec if rec_.value else 0
                list(pool.map(lambda a: fn(*a), step[s & 1]))
    if record == "final":
        tables.to_frame(bufs[steps % 2].view(np.uint8), N, R, tab.Lk, out[0])
    return out
