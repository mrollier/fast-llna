"""CPU backend: llna.h compiled by the system C compiler, called through ctypes from a thread pool.

Two parallel modes (FAST_LLNA_CPU_MODE overrides the choice):
  words  each task owns a slice of replica words and runs all steps (no synchronisation), used when there
         are enough words to keep every thread busy;
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
/* Run nsteps updates of nodes [i0, i1) x words [w0, w1), ping-ponging between buffers a and b (step s reads
   a if s is even). If rec > 0, after every rec-th step the first nb bytes of each node row are copied to
   frames[frame0 + (s + 1) / rec - 1] (canonical layout [n_frames][n][nb]). */
void llna_cpu(const int *indptr, const int *indices, const int *seg_off, const int *seg_thr,
              const int *seg_cell, const u32 *one, const u32 *hasf, const u32 *frac, const u32 *stream,
              const u32 *cm, const u32 *cv, u32 *a, u32 *b, unsigned char *frames, llna_idx n, llna_idx nb,
              llna_idx Wp, u32 k0, u32 k1, u32 t0, llna_idx i0, llna_idx i1, llna_idx w0, llna_idx w1,
              int nsteps, int rec, llna_idx frame0) {
    for (int s = 0; s < nsteps; s++) {
        const u32 *cur = (s & 1) ? b : a;
        u32 *nxt = (s & 1) ? a : b;
        for (llna_idx i = i0; i < i1; i++)
            for (llna_idx w = w0; w < w1; w += VW) {
                WORD o = llna_update(cur, indptr, indices, seg_off, seg_thr, seg_cell, one, hasf, frac, stream,
                                     cm, cv, Wp, k0, k1, t0 + (u32)s, (int)i, w);
                memcpy(nxt + i * Wp + w, &o, sizeof o);
            }
        if (rec > 0 && (s + 1) % rec == 0) {
            llna_idx f = frame0 + (s + 1) / rec - 1, c0 = 4 * w0, c1 = 4 * w1 < nb ? 4 * w1 : nb;
            for (llna_idx i = i0; c1 > c0 && i < i1; i++)
                memcpy(frames + (f * n + i) * nb + c0, (const unsigned char *)(nxt + i * Wp) + c0, c1 - c0);
        }
    }
}
"""

ARRAYS = ("indptr", "indices", "seg_off", "seg_thr", "seg_cell", "one", "hasf", "frac", "stream", "cm", "cv")


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


def run(graph, rules, x0, steps, record, clamp, seed, t0, noise_period, threads=None):
    R, N = x0.n_replicas, graph.n
    W = (R + 31) // 32
    VW = min(8, 1 << (W - 1).bit_length())
    Wp = -(-W // VW) * VW
    tab = tables.build(graph, rules, x0, clamp, seed, noise_period, Wp)
    tab.defines["VW"] = VW
    fn = _load(tab.source(PRELUDE, WRAPPER)).llna_cpu
    fn.restype = None

    nb = (R + 7) // 8
    rec = 0 if record == "final" else record
    frames = np.empty((steps // rec + 1 if rec else 1, N, nb), np.uint8)
    frames[0] = x0.bits
    bufs = [np.ascontiguousarray(tab.arrays["state"]), np.empty((N, Wp), np.uint32)]
    fixed = [_ptr(tab.arrays[k]) for k in ARRAYS]
    L, U, C = ctypes.c_long, ctypes.c_uint32, ctypes.c_int

    def call(cur, nxt, i0, i1, w0, w1, nsteps, t, rec_, frame0):
        fn(
            *fixed,
            _ptr(cur),
            _ptr(nxt),
            _ptr(frames),
            L(N),
            L(nb),
            L(Wp),
            U(tab.k0),
            U(tab.k1),
            U(t),
            L(i0),
            L(i1),
            L(w0),
            L(w1),
            C(nsteps),
            C(rec_),
            L(frame0),
        )

    def ranges(stop, unit, parts):
        edges = np.unique(np.linspace(0, stop, min(stop, parts) + 1).astype(int)) * unit
        return list(zip(edges[:-1], edges[1:], strict=True))

    nthreads = _threads(threads)
    groups = Wp // VW
    # nodes mode pays ~0.3 ms of thread synchronisation per step: worth it only for big steps (M4-measured)
    big = groups >= 2 * nthreads or N * Wp < 1 << 21
    mode = os.environ.get("FAST_LLNA_CPU_MODE") or ("words" if big else "nodes")
    with ThreadPoolExecutor(nthreads) as pool:
        if mode == "words":  # each task owns a word slice for all steps
            tasks = [(*bufs, 0, N, w0, w1, steps, t0, rec, 1) for w0, w1 in ranges(groups, VW, 4 * nthreads)]
            list(pool.map(lambda a: call(*a), tasks))
            final = bufs[steps % 2]
        else:  # every step split over node ranges
            for s in range(steps):
                rec_ = 1 if rec and (s + 1) % rec == 0 else 0
                f = (s + 1) // rec if rec_ else 0
                tasks = [(*bufs, i0, i1, 0, Wp, 1, t0 + s, rec_, f) for i0, i1 in ranges(N, 1, 4 * nthreads)]
                list(pool.map(lambda a: call(*a), tasks))
                bufs.reverse()
            final = bufs[0]
    if record == "final":
        frames[0] = final.view(np.uint8)[:, :nb]
    return frames
