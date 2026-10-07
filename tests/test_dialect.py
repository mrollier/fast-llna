"""llna.h must stay in the C99 / CUDA C++ / Metal common subset: check it parses as C++14 with GPU-style
scalar macros (catches C-only constructs before they reach NVRTC or the Metal compiler)."""

import shutil
import subprocess

import pytest

from fast_llna.kernels.tables import HEADER

STUB = """
typedef unsigned int u32;
typedef long llna_idx;
typedef u32 WORD;
#define LLNA_FN inline
#define LLNA_PTR(T) const T *
#define MULHI(a, b) ((u32)(((unsigned long)(a) * (b)) >> 32))
#define ZEROW 0u
#define LOADW(ptr, off) ((ptr)[off])
#define ANYW(x) ((x) != 0u)
#define RANDW(k0, k1, t, i, st, w, d) llna_rand(k0, k1, t, (u32)(i), (st)[w], d)
#define PMAX 8
#define SEGMAX 16
#define NCELL 9
#define LLNA_D 4
#define LLNA_CLAMP 1
#define LLNA_NP 4
#define LLNA_REPL 0x11111111u
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="no C++ compiler")
@pytest.mark.parametrize(
    "lanes",
    [
        "#define LLNA_L 1\n#define LLNA_LMASK 0x1u\n#define LLNA_COOP 0\n",
        "#define LLNA_L 32\n#define LLNA_LMASK 0xffffffffu\n#define LLNA_COOP 0\n",
        "#define LLNA_L 4\n#define LLNA_LMASK 0xfu\n#define LLNA_COOP 1\n#define LLNA_SHFL_XOR(x, m) (x)\n",
    ],
)
def test_header_parses_as_cpp14(tmp_path, lanes):
    src = tmp_path / "k.cpp"
    src.write_text(STUB + lanes + HEADER)
    res = subprocess.run(
        ["c++", "-std=c++14", "-fsyntax-only", "-Wall", "-Werror", str(src)], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr


CUDA_STUB = """
#define __device__
#define __forceinline__ inline
#define __global__
struct dim3_ { unsigned x, y, z; };
static dim3_ blockIdx, threadIdx, blockDim;
static inline unsigned __umulhi(unsigned a, unsigned b) { return (unsigned)(((unsigned long)a * b) >> 32); }
template <class T> static inline T __ldg(const T *p) { return *p; }
static inline unsigned __ballot_sync(unsigned, int p) { return p ? 1u : 0u; }
template <class T> static inline T __shfl_xor_sync(unsigned, T v, int) { return v; }
static inline int __ffs(unsigned x) { return __builtin_ffs((int)x); }
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="no C++ compiler")
@pytest.mark.parametrize("R", [3, 257])
def test_generated_cuda_source_parses(tmp_path, R):
    """The CUDA host cannot run here; at least its full generated source (stochastic + clamp) must parse,
    with stand-ins for the CUDA builtins."""
    import numpy as np

    import fast_llna as fl
    from fast_llna.kernels import cuda, tables

    g = fl.moore_torus(5, 6)
    rules = fl.Rules(fl.symmetric(5), np.random.default_rng(0).choice([0, 1, 0.5, 0.3], size=(R, 2, 5)))
    clamp = (np.zeros((R, 30), bool), np.zeros((R, 30), bool))
    x0 = fl.random_states(30, R, seed=1)
    Wp = tables.gpu_words(R)
    tab = tables.build(g, rules, x0, clamp, 7, None, Wp, tables.lanes(R) if Wp == 1 else 32)
    tab.defines["LLNA_COOP"] = 1
    src = tmp_path / "k.cpp"
    src.write_text(CUDA_STUB + tab.source(cuda.PRELUDE, cuda.WRAPPER))
    res = subprocess.run(["c++", "-std=c++14", "-fsyntax-only", "-Wall", str(src)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
