# fast-llna: design and implementation plan

## Context

**Current performance.** The `llna` engine (network_automata_robustness) is built on torch float32, scatter-add, and a Python loop over the r intervals. It needs 35 s on CPU for 528 rules × 60 inits × N=900 × T=100. That is about 8e7 node-updates/s; MPS is slower. Ch. 7 of the thesis concluded that the GNN formulation is convenient but not fast, and it was never run on GPU.

**What we build.** `fast-llna` (import `fast_llna`) is a new standalone engine (numpy/scipy core). It simulates LLNAs at hardware speed on:
- multicore CPU;
- Apple GPU (this M4);
- NVIDIA GPU on the user's workstation (specs unknown, so the design is generic, sm_70+).

It picks a backend automatically, with a user override. It must stay extensible: α-async, output noise and fully async updating are planned for later.

**v1 success criteria (agreed):**
1. Every backend is bit-exact against an independent numpy reference, stochastic runs included.
2. At least 100× faster than `llna` on CPU for the 528-rule sweep (target < 1 s on the M4 CPU).
3. N = 1e7 runs T = 1000 on one GPU with packed output.

**Two bugs found in the existing `llna` engine** (verified; to report, not fix here):
- The `uniform` (iso=False) float32 lookup is wrong for r ≥ 22. For q=13, k=22, r=22, float32 gives r·ρ = 12.999999, so cell 12 instead of 13.
- `evolve_rules_batch(iso=True)` with even r puts ρ = ½ into cell 0.

The odd-r symmetric path is exact (checked for r ≤ 25, k ≤ 1000).

## Core reductions (why this is fast)

1. **Integer partitions, no floats.** φ depends only on (s, q, k), with q = Σ_j A_ij s_j. The cell index is exact integer arithmetic:
   - **symmetric(r), odd r:** `qr//k` if 2q<k, `r-1-(k-q)r//k` if 2q>k, `(r-1)/2` if 2q=k.
   - **symmetric(r), even r:** the two partitions differ only at 2q=k. R⁺ gives `r/2-1`, R⁻ gives `r/2`. The default `even="+-"` (B on R⁺, S on R⁻) and `"-+"` are both selectable.
   - **uniform(r):** `min(qr//k, r-1)`, the legacy partition.
   - **majority:** cells {2q<k}, {2q=k}, {2q>k}.

   These formulas were verified against Fraction definitions for r ≤ 25, k ≤ 199, both even conventions: 0 mismatches. For each degree k, the cell is monotone in q. So the rule reduces to a few q-thresholds per distinct degree, precomputed on the host.
2. **Replicas as lanes (multi-spin coding).** State is uint32 words in layout `[N][Wp]`, with replica r stored at word r>>5, bit r&31.
   - Neighbour counts use bit-sliced ripple adds into planes.
   - The rule is applied with bit-sliced `count ≥ threshold` compares plus per-word lane masks. **Every lane may carry a different rule at no extra cost**, so rule sweeps pack rules into lanes.
   - Cost is about 4 integer ops and about k/8 bytes per node-update.
3. **Coalesced, divergence-free GPU mapping.** A warp is one node × 32 consecutive words. That gives broadcast index loads, 128-byte coalesced state loads, and no degree divergence.

## Design decisions

**Single strategy in v1:** replicas as lanes, which also covers R=1 at a constant-factor cost.

**One C-subset kernel core, three thin hosts.** `kernels/llna.h` contains only pure value functions:
- bounded ripple add;
- the GE compare;
- Philox4x32-10;
- the MSB-first Bernoulli draw;
- `update_word`.

Each host owns indexing, loops and stores. Python concatenates `#defines + llna.h + wrapper`.

**Host-specific settings:**

| | CPU | CUDA | Metal |
|---|---|---|---|
| build | `cc -O3 -mcpu/-march=native -shared`, ctypes (GIL released) | CuPy RawModule (NVRTC), `extern "C"` | MLX `mx.fast.metal_kernel` (defines + core in `header=`) |
| `WORD` | `vector_size(32)` 8×u32 (explicit SIMD) | u32 | u32 |
| `LLNA_FN` | `static inline` | `__device__ __forceinline__` | `inline` |
| `LLNA_PTR` | `const u32*` | `const u32*` | `device const u32*` |
| `MULHI` | u64 multiply >>32 | `__umulhi` | `mulhi` |

**Constraints on the header:**
- Not allowed: `<stdint.h>`, `long long`, `restrict`, VLAs, `bool`, array parameters, global arrays, recursion.
- A test compiles the header as C++14 with `-fsyntax-only` to catch dialect slips locally.

**Specialisation and caching.**
- Specialise only on bucketed constants: PMAX ∈ {4, 8, 12, 16, 24}, D ∈ {0, 1, 2, 4, 8, 16, 32}, and the CLAMP flag.
- Per-step scalars (t, N, seed) are passed in a params array, never as template parameters.
- The cache key is the source hash plus the compiler version, flags, and CPU model, because SIGILL is a risk on shared HPC homes.
- Writes are atomic (temp file + `os.replace`).

**Kernel per (node i, word w).**
1. Count: ripple-add each neighbour word `S[col][w]` into planes c[0..Pi). Pi = bitlen(k_i) is a per-node bound, and the e-th add is bounded by bitlen(e+1). Loops are unrolled up to PMAX with a `break`.
2. Initial selection: from s and segment 0, pick ONE/FRAC masks with `b ^ (s & (a^b))`.
3. Segment boundaries: for each boundary `GE(c, thr)` (LSB to MSB, 1 op per plane), update telescopically: `sel ^= g & (new ^ prev)`.
4. Stochastic part: if any fractional bits apply, draw Philox words lazily and compute `out = one | (U < p)`, comparing MSB-first.
5. Clamp: if clamping is on, `out = (out & ~CM) | CV`.

**Mask layout.** `ONE[2][ncell][Wp]` and `FRAC[2][ncell][D][Wp]`, word-contiguous.

**Parallel decomposition.**
- **GPU:** block `(min(Wp,32), 256/bx)`, grid `(ceil(N/by), Wp/bx)`; one launch per step (~5 µs on CUDA, ~20 µs on MLX); MLX calls `mx.eval` every ~32 steps.
- **CPU:** if `Wp/VW ≥ 2·threads`, word-chunk tasks run all T steps with no barrier. Otherwise each step is split into node-range tasks. Both use the same C entry `(i0, i1, w0, w1, nsteps)`.
- **Threads:** the thread count is `sched_getaffinity`, and there are at least 4× more chunks than workers (the M4 has P- and E-cores).

**Padding.**
- GPU: `Wp = pow2ceil(W)` if W ≤ 32, else `roundup(W, 32)`.
- CPU: `roundup(W, 8)`.
- Padding lanes have zero masks and zero state.

**RNG contract** (`rng.py`, bit-identical everywhere):
```
word_d(t,i,w) = Philox4x32-10(ctr=(t, i, stream[w], purpose<<16 | d>>2), key=seed64)[d & 3]
U_r = Σ_d bit_{r&31}(word_d(t, i, r>>5)) · 2^-(d+1)    # MSB-first: independent of batch padding depth
next = 1 iff U_r < p_r ;  p rounded to 2^-32 ; p == 1 via its own ONE mask ; warn if 0 < p rounds to 0
```
- `purpose` is 0 for rule outputs; 1 and 2 are reserved for α-async and output noise.
- `stream[w] = w mod (noise_period/32)`, so defect twins share noise. Otherwise the δ of stochastic rules never relaxes to 0.
- `t0` lets a run continue: 2×50 steps equals 1×100 steps.
- Checked against the Random123 known-answer vectors, e.g. ctr=0, key=0 gives `6627e8d5 e169c58d bc57ac4c 9b00dbd8`.

**Canonical public format.**
- Packed states are uint8 `[..., N, ceil(R/8)]` (`packbits` over replicas, little bit order, padding bits 0).
- On little-endian machines this is the same memory as the u32 or vector word layout, so no conversion is needed. Import asserts little-endian.
- Known limit: R=1 wastes 7/8 of each byte. N=1e7 × T=1000 recorded every step is 10 GB, so use `record=n` stride or "final".

**Conventions.**
- `A[i,j] = 1` means j influences i: row i lists the in-neighbours, k_i is the in-degree. Undirected inputs must be symmetric (not enforced).
- Rejected inputs: k_i = 0, self-loops, and entries ∉ {0,1}.
- Clamping also applies to the initial state.
- One partition per `simulate` call; mixed r needs separate calls.

## Public API

```python
import fast_llna as fl
g = fl.Graph(A)                           # scipy.sparse or numpy 0/1 adjacency
g, offsets = fl.union(g1, g2, ...)        # many network realisations in one run
fl.ring(n, radius=1); fl.moore_torus(rows, cols)
fl.symmetric(r, even="+-"); fl.uniform(r); fl.MAJORITY                     # Partition (exact .cell(q,k,s))
rules = fl.life_like(r, beta, sigma, partition=...)   # beta/sigma scalars or arrays -> Rules (p: [K,2,ncell])
fl.majority(p_tie=0.5); fl.equivalent(r, beta, sigma); fl.nonequivalent(r)  # 528 rows at r=5
rules.take(idx)                           # explicit per-replica rule assignment
init = fl.random_states(N, R, density=0.5, seed=0)   # exactly round(density*N) alive -> States
init, pairs = fl.defect_twins(init, flips=1, seed=0) # twins at r+R; pairs [R,2]
fl.product(rules, init) -> (Rules, States)          # rules x inits ensemble helper
traj = fl.simulate(g, rules, init, steps, record=1 | n | "final", clamp=(mask, value),  # [N] or [R,N]
                   seed=0, t0=0, noise_period=None, backend="auto", threads=None,
                   max_bytes=4e9)                    # size guard -> clear error with projected size
traj.density(); traj.hamming(pairs); traj.states.to_bool(); traj.final()
fl.available_backends()                   # auto: cuda > metal > cpu (small problems -> cpu); FAST_LLNA_BACKEND env
```

`rules` has K=1 (broadcast) or K=R rows. Reducers run as chunked numpy `unpackbits` on the host. A C reducer is added only if reducers exceed ~30% of wall time.

## Files

```
pyproject.toml          # setuptools src layout; extras: metal (mlx, darwin-arm64), cuda (cupy-cuda12x), dev;
                        # package-data kernels/*.h; ruff line 110; pytest markers metal, cuda
environment.yml         # conda-forge, python 3.12
README.md
docs/design.md          # this design, as the repo spec
src/fast_llna/__init__.py, rules.py, graph.py, states.py, rng.py, reference.py, simulate.py
src/fast_llna/kernels/llna.h, cpu.py, metal.py, cuda.py
tests/test_rules.py, test_graph.py, test_rng.py, test_reference.py, test_states.py, test_backends.py, test_dialect.py
benchmarks/bench.py, benchmarks/graphs.py (ER, BA, rewired torus), benchmarks/workstation_job.sh
.github/workflows/ci.yml  # ubuntu(gcc) + macos(clang): ruff + pytest (CPU/reference)
```

**What to reuse from `network_automata_robustness/src/llna/`:**
- The vectorised non-equivalent rule enumeration and its bit-reverse trick (`rules.py`).
- The Moore-torus construction and rewiring semantics (`networks.py`), ported to numpy/scipy.

## Phases (each ends with tests green)

**P0: model and reference.**
- `git init`, scaffold, `docs/design.md`.
- `rules.py`, `graph.py`, `states.py`, `rng.py` (numpy Philox).
- `reference.py` (scipy.sparse counts plus table lookup, stochastic outputs, clamping, t0).
- `simulate(backend="reference")`.

**P1: C core and CPU host.**
- `llna.h`, `kernels/cpu.py` (compile cache, ctypes, both CPU modes).
- `test_backends.py`, `test_dialect.py`, `bench.py`, CI.
- **Gate:** the thesis case on the M4 CPU takes ≤ 1 s (target ~0.2 s).

**P2: Metal host.**
- `kernels/metal.py` (pinned MLX version, hashed kernel names, params array).
- The same bit-identity suite under the `metal` marker.
- Benchmark the M4 GPU against the CPU to set the `auto` small-problem threshold.

**P3: CUDA host.**
- `kernels/cuda.py`; scalars passed explicitly as `np.uint32` / `np.int32`.
- `workstation_job.sh` prints `nvidia-smi` and versions, runs `pytest -m cuda`, runs `bench.py --backend cuda`, and runs `ptxas -v` / SASS checks (spills, LOP3 count).
- The user runs it on the workstation and pastes the output back.

**P4: optimisations, only with evidence from benchmarks.** Each item has a trigger:

| Optimisation | Trigger |
|---|---|
| Degree-sorted node relabelling | Hub divergence when Wp < 32 |
| Persistent multi-step kernel, threadgroup slabs | Launch overhead > 20% |
| Carry-save adders | Counting dominates in SASS or profiles |
| Strategy "nodes as lanes" (node-packed bitsets, ballot output) | R=1, N ≥ 1e6 gathers dominate, or output size bites; the RNG contract already keeps it bit-identical |
| C reducer | Reducers > 30% of wall time |

**Deferred features**, each designed-in through the `purpose` RNG slot and the select/mask hook:
- α-async;
- output noise;
- fully async;
- cycle/fixed-point detection;
- dense/tensor-core path (it loses on sparse graphs per the research).

## Verification

**`test_rules.py`**
- A Fraction oracle built literally from the thesis intervals, covering symmetric odd and even (both conventions), uniform and majority. Range: r = 1..25, k ≤ 64, plus hub degrees {97, 199, 1000}.
- Equivalence is an involution.
- Non-equivalent count is 2^(2r-1) + 2^(r-1) for r = 1..9 (528 at r=5).
- Thesis examples: φ⁵₆,₂₈ ~ φ⁵₂₄,₁₉; φ⁵₁₁,₁₉ ~ φ⁵₆,₅; φ⁵₆,₁₉ is self-equivalent.
- GoL is φ⁹₈,₁₂.

**`test_reference.py`**
- A GoL glider on an 8×8 Moore torus moves by (1,1) every 4 steps.
- ECA 150 = φ³₂,₅ on a ring equals `roll ^ s ^ roll`.
- Complementation: `traj(φ', ¬x) = ¬traj(φ, x)` for r = 5, 9 (odd) and r = 4, 6 (even, both conventions), on irregular and directed graphs.
- Negative control: `uniform(5)` must break complementation at degrees that are multiples of 5.
- Majority is deterministic at odd degree; the tie mean is ≈ ½ within 5σ.
- Clamped nodes hold, including at t = 0.
- t0 continuation equals a single run.
- Every record mode works; k = 0 and self-loops raise.

**`test_rng.py`:** Random123 known-answer vectors.

**`test_states.py`**
- Pack/unpack for R ∈ {1, 7, 8, 9, 31, 33}.
- Exact-density initial states.
- Twins differ in exactly `flips` nodes.
- Reducers match naive numpy.

**`test_backends.py`:** every available backend must be bit-identical to the reference over a pairwise matrix (< 60 s):
- graphs: ring, Moore, ER, BA hub, star k=1023 (PMAX bucket), directed, union;
- R ∈ {1, 31, 32, 33, 64, 65, 257, 1056};
- partitions: symmetric 5/9/4, uniform 5, majority;
- options: clamp, stochastic, record 1/3/final, twins with `noise_period`, mixed rules per lane, threads 1 vs many, both CPU modes forced.

**`test_dialect.py`:** C++14 syntax-only compile of `llna.h` with stub macros.

**Benchmarks (`bench.py`, JSON lines; first call including compile reported separately from steady state):**
1. Thesis case: 528×60 on a rewired torus, p = 0.2, N = 900, T = 100; record every step vs final; compare with the 35 s baseline.
2. Scale R: N = 900, R from 32 to 1e6.
3. Scale N: R = 1024, N from 1e2 to 1e5.
4. Huge N: N from 1e5 to 1e7, ER and Moore, R ∈ {1, 32}.
5. Hubs: BA vs ER.
6. Overheads: stochastic vs deterministic; clamp on vs off.
7. Reducer timing.

Throughput is reported as node-updates/s = N·R·T / time.

**End to end:**
- `pytest -q` locally (reference, CPU, Metal).
- `python benchmarks/bench.py --scenario thesis` on the M4.
- `workstation_job.sh` on the NVIDIA workstation.
