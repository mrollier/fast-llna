# Narrow Lanes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store replicas of R ≤ 32 as L = min(32, pow2ceil(R)) bits per node (node-major), so huge graphs with few replicas run from cache. Add GPU-cooperative hub counting, a flat node-major output layout, and `noise_period` values that divide 32.

**Architecture:** `llna.h` stays the single kernel core. One gather macro (`LLNA_GET`) switches between the node-major L-bit layout and the v1 `[N][Wp]` layout. `llna_update` gains a strided edge loop and a shuffle butterfly, which lets a SIMD group count one hub together. The hosts change as follows:
- The CPU loops over the 32/L nodes of each output word.
- Metal and CUDA run one thread per node (R ≤ 32) and assemble words with XOR shuffles.
- R > 32 keeps the v1 2-D word kernel.

`States` becomes a flat little-endian bit array with replica r of node i at bit `i·Ls + r`.

**Tech Stack:**
- Python 3.12, numpy, scipy;
- C11 kernel through ctypes (CPU);
- MLX `mx.fast.metal_kernel` (Metal);
- CuPy RawModule/NVRTC (CUDA, which cannot be run here);
- pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-07-narrow-lanes-design.md`

## Global Constraints

**Environment and commands**
- Run everything in the conda env: `conda run -n fast-llna <cmd>` (Python 3.12, numpy ≥ 1.26, scipy ≥ 1.12). Do not use `np.bitwise_count`, which requires numpy ≥ 2.0.
- Ruff config: line length 110; rules E, F, W, I, UP, B, NPY. `ruff check src tests benchmarks` must pass after every task.

**Kernel header rules**
- `llna.h` stays in the common subset of C99, CUDA C++ and Metal. Not allowed:
  - standard headers, `long long`, `restrict`;
  - VLAs, `bool`, array parameters, globals, recursion;
  - modifying and reading a variable within one expression.

**Correctness**
- Every backend must stay bit-identical to `backend="reference"`. The reference must never import kernel code.
- Lane width: `L = min(32, pow2ceil(R))` for R ≤ 32. R > 32 keeps `[N][Wp]` words.
- Frame layout: `Ls = pow2ceil(R)` if R ≤ 8, else `8·ceil(R/8)`. Frame bytes are `ceil(N·Ls/8)`, and replica r of node i is bit `i·Ls + r`.
- Hub threshold: `LLNA_HUB = max(32, 4·ceil(mean degree))`.
- `noise_period` must divide 32 (1, 2, 4, 8, 16) or be a positive multiple of 32. Replica r draws the digits of replica `r mod noise_period`.

**Git**
- Commit after each task with a message ending in `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Do not push.

## Review Focus

1. **N not a multiple of 32/L, and N smaller than one word.** The last state word is partial, and the frame must have zero padding bits. Pinned by the `ring5-R1`, `twins-R2-noise1`, `even4-R3-clamp`, `stochastic-R5` and `uniform5-R9-record` cases (Task 3).
2. **A hub in the last, partial SIMD group.** Threads with i ≥ N still take part in the shuffles, and the hub's result lands in the right word. Pinned by `star-last-R4` and the `hubs-*` cases, whose hub is node 149 of 150 (Task 4).
3. **Degree exactly at the hub threshold.** k = 32 must take the normal path and k = 33 the cooperative one, and both must be exact. Pinned by the `HUBS` graph and `test_hub_threshold_scales_with_mean_degree` (Task 4).
4. **R where frame bits ≠ kernel bits (R = 17…24).** Recording copies 3 of 4 bytes per node, including `record="final"`. Pinned by `R20-strided-final` (Task 3).
5. **`noise_period` with narrow lanes and with the word kernel.** Twins at R = 2 with p = 1 share draws; p = 16 at R = 64 replicates lanes inside each word. Pinned by `twins-R2-noise1` (Task 3) and `majority-R64-noise16` (Task 1).

---

### Task 1: Shared noise for periods that divide 32

**Files:**
- Modify: `src/fast_llna/rng.py` (docstring, new `noise_streams`)
- Modify: `src/fast_llna/reference.py` (`_digits`, `run` signature)
- Modify: `src/fast_llna/simulate.py` (validation, pass `noise_period` to backends)
- Modify: `src/fast_llna/kernels/tables.py` (`build` signature, `LLNA_NP` / `LLNA_REPL`, stream)
- Modify: `src/fast_llna/kernels/llna.h` (replicate digit lanes)
- Modify: `src/fast_llna/kernels/cpu.py`, `metal.py`, `cuda.py` (`run` signature)
- Modify: `src/fast_llna/states.py` (`defect_twins` docstring)
- Test: `tests/test_reference.py`, `tests/test_backends.py`, `tests/test_dialect.py`

**Interfaces:**
- Produces `rng.noise_streams(n_words: int, noise_period: int | None) -> np.ndarray[uint32]`.
- Produces the backend signature `run(graph, rules, x0, steps, record, clamp, seed, t0, noise_period, threads=None)`, used by all later tasks. It replaces `stream`.
- Produces `tables.build(graph, rules, x0, clamp, seed, noise_period, Wp)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reference.py`:

```python
@pytest.mark.parametrize("period", [1, 2, 16])
def test_twins_share_noise_for_periods_dividing_32(period):
    g = fl.ring(64)
    x, pairs = fl.defect_twins(fl.random_states(64, period, seed=1), flips=0)
    shared = sim(g, fl.majority(), x, 8, seed=3, noise_period=period).hamming(pairs)
    indep = sim(g, fl.majority(), x, 8, seed=3).hamming(pairs)
    assert np.all(shared == 0) and indep.max() > 0


def test_noise_period_beyond_replica_count_changes_nothing():
    g, x = fl.ring(40), fl.random_states(40, 64, seed=2)
    a = sim(g, fl.majority(), x, 6, seed=4, noise_period=64).states.bits
    b = sim(g, fl.majority(), x, 6, seed=4).states.bits
    assert np.array_equal(a, b)


@pytest.mark.parametrize("period", [0, -32, 3, 24, 48])
def test_noise_period_must_divide_or_be_multiple_of_32(period):
    with pytest.raises(ValueError, match="noise_period"):
        sim(fl.ring(10), fl.majority(), fl.random_states(10, 2), 1, noise_period=period)
```

Add this entry to `CASES` in `tests/test_backends.py`, after `"stochastic-clamp-R257"`:

```python
    (
        "majority-R64-noise16",
        fl.ring(50, 2),
        64,
        lambda R: fl.majority(),
        6,
        {"noise_period": 16, "seed": 9},
    ),
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `conda run -n fast-llna pytest -q tests/test_reference.py tests/test_backends.py -k "noise or twins"`

Expected:
- `test_twins_share_noise_for_periods_dividing_32[1|2|16]` fails with `ValueError: noise_period must be a positive multiple of 32`.
- `majority-R64-noise16` fails with the same error.
- The other new tests pass already. They are regression guards.

- [ ] **Step 3: Implement**

In `src/fast_llna/rng.py`, replace the sentence ``t is the timestep being left (t -> t+1), i the node, w the 32-replica word (``stream[w] = w`` unless replicas share noise), ...`` with:

```
t is the timestep being left (t -> t+1), i the node, w the 32-replica word, purpose 0 the rule outputs (1 and
2 are reserved for alpha-asynchronous updating and output noise). Without shared noise, replica r reads bit
r % 32 of word_d(t, i, r // 32), d = 0, 1, ...;
```

Leave the rest of that paragraph unchanged. Append this paragraph to the module docstring:

```
Shared noise: with ``noise_period`` p, replica r reads exactly the digits of replica r' = r mod p, i.e. bit
r' % 32 of word_d(t, i, r' // 32). p must divide 32 or be a multiple of 32. Kernels implement multiples of 32
with stream[w] = w mod (p / 32), and divisors of 32 with stream[w] = 0 plus repeating the low p bits of each
digit word across the word.
```

Then add after `split_seed`:

```python
def noise_streams(n_words: int, noise_period: int | None) -> np.ndarray:
    """stream[w] of each 32-replica word w (see the module contract); periods below 32 use stream 0."""
    w = np.arange(n_words, dtype=np.uint32)
    return w if noise_period is None else w % np.uint32(max(1, noise_period // 32))
```

In `src/fast_llna/reference.py`, replace `_digits` and the line that calls it:

```python
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
```

Change `run` to `def run(graph, rules, x0: States, steps, record, clamp, seed, t0, noise_period, threads=None):`, and its draw to `u = _digits(seed, t0 + step, graph.n, R, depth, noise_period)`.

In `src/fast_llna/simulate.py`:
- Replace the block from `W = (R + 31) // 32` through `stream %= np.uint32(noise_period // 32)` with:

  ```python
      if noise_period is not None and (noise_period <= 0 or (32 % noise_period and noise_period % 32)):
          raise ValueError("noise_period must divide 32 or be a positive multiple of 32")
  ```

- Change the backend call to `bits = _module(backend).run(graph, rules, x0, steps, record, clamp, seed, t0, noise_period, threads)`.
- In the docstring, replace the `noise_period` text with: `Replicas r and r + noise_period draw the same random numbers (e.g. defect twins); must divide 32 or be a multiple of 32.`

In `src/fast_llna/states.py`, change the last sentence of `defect_twins`' docstring to: ``For stochastic rules pass ``noise_period=R`` to ``simulate`` so twins share their random draws (R must divide 32 or be a multiple of 32).``

In `src/fast_llna/kernels/tables.py`:
- Change `from ..rng import quantize, split_seed` to `from ..rng import noise_streams, quantize, split_seed`.
- Change the signature to `def build(graph, rules, x0, clamp, seed, noise_period, Wp) -> Tables:`.
- Inside, after `pmax = ...`, add:

  ```python
      period = noise_period if noise_period is not None and noise_period < 32 else 32
  ```

- Add these two entries to the `defines` dict:

  ```python
              "LLNA_NP": period,
              "LLNA_REPL": f"{sum(1 << s for s in range(0, 32, period)):#x}u",
  ```

- Replace the three `st = ...` / `a["stream"] = st` lines with `a["stream"] = noise_streams(Wp, noise_period)`.

In `src/fast_llna/kernels/llna.h`:
- In the header comment, extend the defines line with `LLNA_NP (noise period if it divides 32, else 32), LLNA_REPL (a 1 every LLNA_NP bits)`.
- In the digit loop, directly after `WORD u = RANDW(k0, k1, t, i, STREAM, w, d);`, insert:

  ```c
  #if LLNA_NP < 32 /* noise period p divides 32: lane r reads digit lane r % p */
              u = (u & ((1u << LLNA_NP) - 1u)) * LLNA_REPL;
  #endif
  ```

In each of `src/fast_llna/kernels/cpu.py`, `metal.py` and `cuda.py`:
- change `def run(graph, rules, x0, steps, record, clamp, seed, t0, stream, threads=None):` to `def run(graph, rules, x0, steps, record, clamp, seed, t0, noise_period, threads=None):`;
- change `tables.build(graph, rules, x0, clamp, seed, stream, Wp)` to `tables.build(graph, rules, x0, clamp, seed, noise_period, Wp)`.

In `tests/test_dialect.py`:
- Add `#define LLNA_NP 4` and `#define LLNA_REPL 0x11111111u` to `STUB`.
- In `test_generated_cuda_source_parses`, change the build call to `tables.build(g, rules, x0, clamp, 7, None, tables.gpu_words(R))`.

- [ ] **Step 4: Run the full suite and verify it passes**

Run: `conda run -n fast-llna ruff check src tests benchmarks && conda run -n fast-llna pytest -q`
Expected: `All checks passed!` and every test passes. The count is 165 plus the new ones; nothing is skipped beyond CUDA.

- [ ] **Step 5: Mutation check**

Change `* LLNA_REPL` to `* 1u` in `llna.h` and run `conda run -n fast-llna pytest -q tests/test_backends.py -k noise16`. Expected: FAIL on cpu and metal. Then revert with `git checkout -p src/fast_llna/kernels/llna.h`, or by undoing the edit.

- [ ] **Step 6: Commit**

```bash
git add -A src tests
git commit -m "Share noise for noise_period values dividing 32

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Flat node-major `States` layout

**Files:**
- Modify: `src/fast_llna/states.py` (layout helpers, `States`, `Trajectory`)
- Modify: `src/fast_llna/simulate.py` (size guard, `States(bits, R, N)`)
- Modify: `src/fast_llna/kernels/tables.py` (`to_kernel`, `to_frame`, `rows_to_frames`, `_words`; use them in `build`)
- Modify: `src/fast_llna/kernels/cpu.py`, `metal.py`, `cuda.py` (record via the new helpers)
- Test: `tests/test_states.py`, `tests/test_reference.py`

**Interfaces:**
- Consumes the Task 1 signatures.
- Produces in `fast_llna.states`:
  - `bits_per_node(R: int) -> int`;
  - `frame_bytes(n: int, R: int) -> int`;
  - `pack(x: bool[..., R, N], L: int) -> uint8[..., ceil(N·L/8)]`;
  - `unpack(bits, n: int, R: int, L: int) -> bool[..., R, n]`;
  - `States(bits, n_replicas, n)`, where `n` is now a field.
- Produces in `fast_llna.kernels.tables`:
  - `to_kernel(x0: States, Lk: int) -> uint32[ceil(N·Lk/32)]`, where Lk is the number of kernel bits per node;
  - `to_frame(state_u8, n, R, Lk) -> uint8[frame_bytes]`, which works on numpy or cupy for byte-aligned or equal layouts;
  - `rows_to_frames(rows: uint8[F, N, nb], R) -> uint8[F, frame_bytes]`;
  - `tab.arrays["state"]` is now flat.

- [ ] **Step 1: Write the failing tests**

In `tests/test_states.py`:
- Replace `test_pack_roundtrip_and_canonical_layout` and `test_padding_bits_are_zero`.
- Add `from fast_llna.states import bits_per_node, frame_bytes` below `import fast_llna as fl`.
- Parametrise the reducer test.

The new and changed tests:

```python
@pytest.mark.parametrize("R, L", [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8), (9, 16), (17, 24), (33, 40)])
def test_bits_per_node(R, L):
    assert bits_per_node(R) == L


@pytest.mark.parametrize("R", [1, 2, 3, 5, 7, 8, 9, 17, 31, 33])
def test_flat_node_major_layout(R):
    n, L = 13, bits_per_node(R)
    x = np.random.default_rng(R).integers(0, 2, size=(3, R, n)).astype(bool)
    s = fl.States.from_bool(x)
    assert s.bits.shape == (3, frame_bytes(n, R)) == (3, -(-n * L // 8))
    assert s.n_replicas == R and s.n == n
    assert np.array_equal(s.to_bool(), x)
    bit = np.unpackbits(s.bits, axis=-1, bitorder="little").astype(bool)  # [3, 8 * frame bytes]
    t, i, r = np.meshgrid(np.arange(3), np.arange(n), np.arange(R), indexing="ij")
    assert np.array_equal(bit[t, i * L + r], x[t, r, i])  # replica r of node i is bit i * L + r
    unused = np.ones(bit.shape[-1], bool)
    unused[(i * L + r)[0].ravel()] = False
    assert not bit[:, unused].any()  # padding lanes and trailing bits are 0


@pytest.mark.parametrize("R", [1, 3, 11])
def test_trajectory_reducers_match_naive(R):
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=(4, R, 25)).astype(bool)  # [T, R, N]
    traj = fl.Trajectory(fl.States.from_bool(x), np.arange(4))
    assert np.allclose(traj.density(), x.mean(axis=2))
    pairs = np.array([[0, R - 1], [R // 2, 0], [R - 1, R - 1]])
    assert np.allclose(traj.hamming(pairs), (x[:, pairs[:, 0]] ^ x[:, pairs[:, 1]]).mean(axis=2))
    assert np.array_equal(traj.final().to_bool(), x[-1])
```

Append to `tests/test_reference.py`:

```python
def test_size_guard_counts_bits_per_node_for_one_replica():
    g, x = fl.ring(800), fl.random_states(800, 1, seed=0)  # 100 bytes per frame, 100 frames
    sim(g, fl.life_like(3, 2, 5), x, 99, max_bytes=10_000)
    with pytest.raises(ValueError, match="bytes"):
        sim(g, fl.life_like(3, 2, 5), x, 99, max_bytes=9_999)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `conda run -n fast-llna pytest -q tests/test_states.py tests/test_reference.py -k "layout or bits_per_node or reducers or size_guard"`

Expected:
- `ImportError: cannot import name 'bits_per_node'`, or the shape assertion fails;
- the size-guard test fails, because 100 frames × 800 bytes exceed 10 000.

- [ ] **Step 3: Implement `states.py`**

Replace the module docstring, `States` and `Trajectory._chunks`/`final`, and add the helpers:

```python
"""Bit-packed configurations of R replicas and trajectories of them.

Layout: a configuration is a flat little-endian bit array, uint8 ``bits[..., ceil(N * L / 8)]``, in which
replica r of node i is bit i * L + r. L = bits_per_node(R) is the next power of two (1, 2, 4, 8) for R <= 8,
and whole bytes, 8 * ceil(R / 8), beyond. Padding bits are 0. For R >= 8 these are the bytes of the rows
[N, ceil(R / 8)].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rules import Rules

_CHUNK = 1 << 26  # unpacked bytes per reducer chunk


def bits_per_node(R: int) -> int:
    return 1 << (R - 1).bit_length() if R <= 8 else 8 * -(-R // 8)


def frame_bytes(n: int, R: int) -> int:
    return -(-n * bits_per_node(R) // 8)


def pack(x, L: int) -> np.ndarray:
    """bool [..., R, N] -> uint8 [..., ceil(N * L / 8)] with replica r of node i at bit i * L + r (R <= L)."""
    x = np.swapaxes(np.asarray(x, bool), -1, -2)  # [..., N, R]
    *lead, n, R = x.shape
    if L == 8 * -(-R // 8):  # whole bytes per node: packbits pads each node row itself
        return np.packbits(x, axis=-1, bitorder="little").reshape(*lead, -1)
    x = np.concatenate([x, np.zeros((*lead, n, L - R), bool)], axis=-1)
    return np.packbits(x.reshape(*lead, n * L), axis=-1, bitorder="little")


def unpack(bits, n: int, R: int, L: int) -> np.ndarray:
    """Inverse of :func:`pack`: uint8 [..., ceil(n * L / 8)] -> bool [..., R, n]."""
    x = np.unpackbits(bits, axis=-1, count=n * L, bitorder="little")
    return np.swapaxes(x.reshape(*bits.shape[:-1], n, L)[..., :R], -1, -2).astype(bool)


@dataclass(frozen=True, eq=False)
class States:
    bits: np.ndarray
    n_replicas: int
    n: int

    @classmethod
    def from_bool(cls, x) -> States:
        """From a boolean array ``[..., R, N]``."""
        x = np.asarray(x, dtype=bool)
        return cls(pack(x, bits_per_node(x.shape[-2])), x.shape[-2], x.shape[-1])

    def to_bool(self) -> np.ndarray:
        """Boolean array ``[..., R, N]``."""
        return unpack(self.bits, self.n, self.n_replicas, bits_per_node(self.n_replicas))
```

In `Trajectory`:
- change the docstring to `"""Recorded configurations ``states.bits[T', frame bytes]`` at timesteps ``times``."""`;
- replace `_chunks` and `final`:

```python
    def _chunks(self):
        s = self.states
        L = bits_per_node(s.n_replicas)
        step = max(1, _CHUNK // max(1, s.n * L))
        for t in range(0, s.bits.shape[0], step):
            x = np.unpackbits(s.bits[t : t + step], axis=-1, count=s.n * L, bitorder="little")
            yield x.reshape(-1, s.n, L)[..., : s.n_replicas]  # [t, N, R]
```

```python
    def final(self) -> States:
        return States(self.states.bits[-1], self.states.n_replicas, self.states.n)
```

`density` and `hamming` stay as they are, because they already reduce over axis 1 of `[t, N, R]`.

- [ ] **Step 4: Implement `simulate.py`**

Change `from .states import States, Trajectory` to `from .states import States, Trajectory, frame_bytes`. Replace `nbytes = len(times) * N * ((R + 7) // 8)` with `nbytes = len(times) * frame_bytes(N, R)`, and the return with `return Trajectory(States(bits, R, N), times)`.

- [ ] **Step 5: Implement the tables helpers**

In `src/fast_llna/kernels/tables.py`:
- add `from ..states import bits_per_node, frame_bytes, pack, unpack`;
- add the helpers below `_pack_words`;
- in `build`, replace the last three state lines (`state = np.zeros(...)` through `a["state"] = state.view("<u4")`) with `a["state"] = to_kernel(x0, 32 * Wp)  # flat [N * Wp]`.

```python
def _words(b: np.ndarray, n_words: int) -> np.ndarray:
    """uint8 bytes -> uint32 words, zero-padded to n_words."""
    return np.pad(b, (0, 4 * n_words - b.size)).view("<u4")


def to_kernel(x0, Lk: int) -> np.ndarray:
    """States -> kernel state words with Lk bits per node (node-major), ceil(N * Lk / 32) words."""
    N, R, Ls = x0.n, x0.n_replicas, bits_per_node(x0.n_replicas)
    if Ls == Lk:
        b = x0.bits
    elif Ls % 8 == 0 and Lk % 8 == 0:  # whole bytes per node: pad each node row
        b = np.pad(x0.bits.reshape(N, Ls // 8), ((0, 0), (0, (Lk - Ls) // 8))).ravel()
    else:
        b = pack(x0.to_bool(), Lk)
    return _words(b, -(-N * Lk // 32))


def to_frame(state, n: int, R: int, Lk: int):
    """Kernel state bytes (Lk bits per node, trailing padding allowed) -> one frame in the States layout.
    The first two cases also work on cupy arrays."""
    Ls = bits_per_node(R)
    if Ls == Lk:
        return state[: frame_bytes(n, R)]
    if Ls % 8 == 0 and Lk % 8 == 0:
        return state[: n * Lk // 8].reshape(n, Lk // 8)[:, : Ls // 8].reshape(-1)
    return pack(unpack(state[: -(-n * Lk // 8)], n, R, Lk), Ls)


def rows_to_frames(rows: np.ndarray, R: int) -> np.ndarray:
    """v1 rows [F, N, ceil(R / 8)] -> frames [F, frame bytes]: a free reshape for R >= 5."""
    F, n, nb = rows.shape
    if bits_per_node(R) == 8 * nb:
        return rows.reshape(F, n * nb)
    x = np.unpackbits(rows, axis=-1, count=R, bitorder="little").swapaxes(-1, -2).astype(bool)
    return pack(x, bits_per_node(R))
```

- [ ] **Step 6: Adapt the hosts**

`src/fast_llna/kernels/cpu.py`, in `run`:
1. Replace `frames = np.empty(...)` and `frames[0] = x0.bits` with:

   ```python
       rows = np.empty((steps // rec + 1 if rec else 1, N, nb), np.uint8)  # the kernel records [N, nb] rows
       state = tab.arrays["state"]
       rows[0] = state.view(np.uint8).reshape(N, 4 * Wp)[:, :nb]
   ```

2. Change `bufs = [...]` to `bufs = [state.copy(), np.empty_like(state)]`.
3. Change `_ptr(frames)` to `_ptr(rows)`.
4. Replace the final block with:

   ```python
       if record == "final":
           rows[0] = final.view(np.uint8).reshape(N, 4 * Wp)[:, :nb]
       return tables.rows_to_frames(rows, R)
   ```

`src/fast_llna/kernels/metal.py`:
1. Add `from ..states import frame_bytes` below `from . import tables`.
2. In `run`, remove `nb = (R + 7) // 8`.
3. Replace the frames allocation:

   ```python
       frames = np.empty((steps // rec + 1 if rec else 1, frame_bytes(N, R)), np.uint8)
       frames[0] = x0.bits
       pending = []  # (frame index, lazy mx array)

       def frame(a):
           return tables.to_frame(np.asarray(a).view(np.uint8), N, R, 32 * Wp)

       def flush():
           mx.eval(cur, *(a for _, a in pending))
           for f, a in pending:
               frames[f] = frame(a)
           pending.clear()
   ```

4. Change `cur = mx.array(_pad(tab.arrays["state"].ravel()))` to `cur = mx.array(_pad(tab.arrays["state"]))`.
5. Change the final block to `frames[0] = frame(cur)`.

`src/fast_llna/kernels/cuda.py`, in `run`:
1. Rename `frames` to `rows` throughout and keep its v1 shape.
2. Change `rows[0] = x0.bits` to:

   ```python
       rows[0] = tab.arrays["state"].view(np.uint8).reshape(N, 4 * Wp)[:, :nb]
   ```

3. Change `frame = cur.view(np.uint8)[:, :nb]` and the final `cur.view(np.uint8)[:, :nb]` to `cur.view(np.uint8).reshape(N, 4 * Wp)[:, :nb]`.
4. Return `tables.rows_to_frames(rows, R)`.

- [ ] **Step 7: Run the full suite and verify it passes**

Run: `conda run -n fast-llna ruff check src tests benchmarks && conda run -n fast-llna pytest -q`
Expected: all pass, and the backends are still bit-identical. R ≤ 4 records through the repacking fallback for now.

- [ ] **Step 8: Commit**

```bash
git add -A src tests
git commit -m "Flat node-major States layout (bits_per_node(R) bits per node)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Narrow gathers in the kernel core and the CPU host

**Files:**
- Modify: `src/fast_llna/kernels/llna.h` (`LLNA_GET`, `lane`/`coop` parameters, strided loop)
- Modify: `src/fast_llna/kernels/tables.py` (`lanes`, `L` parameter, `LLNA_L`, `LLNA_LMASK`, clamp packing)
- Modify: `src/fast_llna/kernels/cpu.py` (narrow wrapper and run)
- Modify: `src/fast_llna/kernels/metal.py`, `cuda.py` (pass `0, 1` for the new parameters)
- Test: `tests/test_backends.py`, `tests/test_dialect.py`

**Interfaces:**
- Consumes `to_kernel`, `to_frame`, `pack` and `frame_bytes` from Task 2.
- Produces:
  - `tables.lanes(R: int) -> int`, equal to `min(32, pow2ceil(R))`;
  - `tables.build(graph, rules, x0, clamp, seed, noise_period, Wp, L=32)`, where the kernel has `L·Wp` bits per node;
  - `llna_update(..., int i, llna_idx w, int lane, int coop)`;
  - the C function `llna_cpu`, whose signature is unchanged. In narrow builds `i0`/`i1` are output word indices and `frames` is `[F][frame bytes]`.

- [ ] **Step 1: Write the failing tests**

Add these entries to `CASES` in `tests/test_backends.py`:

```python
    ("ring5-R1", fl.ring(5), 1, lambda R: fl.life_like(3, 2, 5), 7, {}),
    (
        "twins-R2-noise1",
        random_digraph(60, 2, 9, 11),
        2,
        lambda R: fl.majority(),
        9,
        {"noise_period": 1, "init": twins(1, 11)},
    ),
    (
        "even4-R3-clamp",
        random_digraph(77, 1, 20, 12),
        3,
        lambda R: mixed(4, R, 12, fl.symmetric(4)),
        8,
        {"clamp": clamp(3, 77, 12), "record": 2},
    ),
    ("stochastic-R5", random_digraph(61, 1, 12, 13), 5, lambda R: stochastic(fl.symmetric(5), R, 13), 7, {"seed": 5}),
    (
        "uniform5-R9-record",
        random_digraph(45, 1, 16, 14),
        9,
        lambda R: mixed(5, R, 14, fl.uniform(5)),
        6,
        {"record": 3},
    ),
    ("R20-strided-final", random_digraph(50, 1, 10, 15), 20, lambda R: mixed(9, R, 15), 5, {"record": "final"}),
```

Replace `test_cpu_modes_and_thread_counts` so that it also covers the narrow layout with several aligned word ranges. With N = 700 and L = 4 there are 88 words, which gives 3 tasks:

```python
@pytest.mark.skipif("cpu" not in fl.available_backends(), reason="no C compiler")
@pytest.mark.parametrize("mode", ["words", "nodes"])
@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("R", [3, 300])
def test_cpu_modes_and_thread_counts(mode, threads, R, monkeypatch):
    monkeypatch.setenv("FAST_LLNA_CPU_MODE", mode)
    graph = random_digraph(700, 1, 15, 9)
    rules = stochastic(fl.symmetric(5), R, 9)
    init = fl.random_states(700, R, seed=9)
    want = fl.simulate(graph, rules, init, 9, backend="reference", seed=3, record=3)
    got = fl.simulate(graph, rules, init, 9, backend="cpu", seed=3, record=3, threads=threads)
    assert np.array_equal(got.states.bits, want.states.bits)
```

In `tests/test_dialect.py`, keep `STUB` as it is and replace `test_header_parses_as_cpp14` with this version, parametrised over the lane width:

```python
@pytest.mark.skipif(shutil.which("c++") is None, reason="no C++ compiler")
@pytest.mark.parametrize("lanes", ["#define LLNA_L 1\n#define LLNA_LMASK 0x1u\n", "#define LLNA_L 32\n#define LLNA_LMASK 0xffffffffu\n"])
def test_header_parses_as_cpp14(tmp_path, lanes):
    src = tmp_path / "k.cpp"
    src.write_text(STUB + lanes + HEADER)
    res = subprocess.run(
        ["c++", "-std=c++14", "-fsyntax-only", "-Wall", "-Werror", str(src)], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr
```

- [ ] **Step 2: Run the tests and record the baseline**

Run: `conda run -n fast-llna pytest -q tests/test_backends.py tests/test_dialect.py`

Expected: everything passes. The new CASES go through the v1 kernels plus the Task 2 conversions, which are exact, so they pin behaviour before the refactor. The dialect variants pass because the v1 header ignores `LLNA_L`.

This task is a refactor that must not change results. Its RED step is therefore the mutation check in Step 6: the narrow gather must be able to break these cases.

- [ ] **Step 3: Implement the header**

In `src/fast_llna/kernels/llna.h`:

1. Add to the header comment's defines list: `LLNA_L (bits per node: 1..16 node-major fields with Wp == 1, or 32 for [N][Wp] words), LLNA_LMASK ((1 << LLNA_L) - 1)`.

2. After the `LLNA_OFFD` macro, add:

   ```c
   #if LLNA_L < 32 /* node j's replicas are bits j*L .. j*L + L - 1 of the flat word array (Wp == 1, w == 0) */
   #define LLNA_GET(P, j)                                                                                 \
       ((LOADW(P, ((llna_idx)(j) * LLNA_L) >> 5) >> (u32)(((llna_idx)(j) * LLNA_L) & 31)) & LLNA_LMASK)
   #else
   #define LLNA_GET(P, j) LOADW(P, (llna_idx)(j) * Wp + w)
   #endif
   ```

3. Change the `llna_update` signature's last line to `llna_idx w, int lane, int coop) {`.

4. Replace the counting block, from `/* bit-sliced count ...` through the closing brace of the edge loop, with:

   ```c
       /* bit-sliced count of living in-neighbours e = lane, lane + coop, ...: c[p] holds bit p of the count */
       int lo = indptr[i], k = indptr[i + 1] - lo, lim = 0, m = 0;
       WORD c[PMAX];
       for (int p = 0; p < PMAX; p++) c[p] = ZEROW;
       for (int e = lane; e < k; e += coop) {
           WORD x = LLNA_GET(S, indices[lo + e]);
           m++;
           if ((m & (m - 1)) == 0) lim++; /* lim = bitlen(m): m added counts still fit in lim planes */
           for (int p = 0; p < PMAX; p++) {
               if (p >= lim) break;
               WORD carry = c[p] & x;
               c[p] ^= x;
               x = carry;
           }
       }
       WORD s = LLNA_GET(S, i);
   ```

   This removes the old `WORD s = LOADW(S, (llna_idx)i * Wp + w);` line.

5. Replace the clamp line with `out = (out & ~LLNA_GET(CM, i)) | LLNA_GET(CV, i);`.

- [ ] **Step 4: Implement the tables and hosts**

In `src/fast_llna/kernels/tables.py`:

1. Add the `lanes` helper:

   ```python
   def lanes(R: int) -> int:
       """Bits per node in the kernel state for R <= 32 replicas."""
       return min(32, 1 << (R - 1).bit_length())
   ```

2. Change the signature to `def build(graph, rules, x0, clamp, seed, noise_period, Wp, L=32) -> Tables:`. Directly after `R, N = ...` add `Lk = L * Wp  # kernel bits per node`.

3. Add these entries to `defines`:

   ```python
               "LLNA_L": L,
               "LLNA_LMASK": f"{(1 << L) - 1:#x}u",
   ```

4. Replace the clamp packing and the state line:

   ```python
       if clamp is not None:
           n_words = -(-N * Lk // 32)
           a["cm"] = _words(pack(clamp[0], Lk), n_words)
           a["cv"] = _words(pack(clamp[0] & clamp[1], Lk), n_words)
       else:
           a["cm"] = a["cv"] = np.zeros(1, np.uint32)
       a["state"] = to_kernel(x0, Lk)  # flat, ceil(N * Lk / 32) words
   ```

   `clamp` arrays are `[R, N]`.

In `src/fast_llna/kernels/metal.py`, change the BODY call's last line to `(int)i, (llna_idx)w, 0, 1);`. In `src/fast_llna/kernels/cuda.py`, change the WRAPPER call's last argument line to `cv, Wp, k0, k1, t, (int)i, w, 0, 1);`.

In `src/fast_llna/kernels/cpu.py`:

1. Wrap the existing `llna_cpu` definition in `#else` … `#endif`, preceded by this narrow version. The `llna_update` call in the existing version gains `, 0, 1` before `);`.

   ```c
   #if LLNA_L < 32
   /* Narrow layout: update output words [i0, i1) (nodes 32/L * v .. 32/L * v + 32/L - 1 of word v) for nsteps
      steps; after every rec-th step copy the words' bytes into frames[frame0 + (s + 1) / rec - 1] ([F][fb]). */
   void llna_cpu(const int *indptr, const int *indices, const int *seg_off, const int *seg_thr,
                 const int *seg_cell, const u32 *one, const u32 *hasf, const u32 *frac, const u32 *stream,
                 const u32 *cm, const u32 *cv, u32 *a, u32 *b, unsigned char *frames, llna_idx n, llna_idx nb,
                 llna_idx Wp, u32 k0, u32 k1, u32 t0, llna_idx i0, llna_idx i1, llna_idx w0, llna_idx w1,
                 int nsteps, int rec, llna_idx frame0) {
       llna_idx fb = (n * LLNA_L + 7) / 8;
       for (int s = 0; s < nsteps; s++) {
           const u32 *cur = (s & 1) ? b : a;
           u32 *nxt = (s & 1) ? a : b;
           for (llna_idx v = i0; v < i1; v++) {
               u32 acc = 0;
               for (llna_idx i = v * (32 / LLNA_L); i < (v + 1) * (32 / LLNA_L) && i < n; i++) {
                   WORD o = llna_update(cur, indptr, indices, seg_off, seg_thr, seg_cell, one, hasf, frac, stream,
                                        cm, cv, Wp, k0, k1, t0 + (u32)s, (int)i, 0, 0, 1);
                   acc |= o[0] << ((i * LLNA_L) & 31);
               }
               nxt[v] = acc;
           }
           if (rec > 0 && (s + 1) % rec == 0) {
               llna_idx f = frame0 + (s + 1) / rec - 1, c0 = 4 * i0, c1 = 4 * i1 < fb ? 4 * i1 : fb;
               if (c1 > c0) memcpy(frames + f * fb + c0, (const unsigned char *)nxt + c0, c1 - c0);
           }
       }
   }
   #else
   ```

2. Replace `run` with:

   ```python
   def run(graph, rules, x0, steps, record, clamp, seed, t0, noise_period, threads=None):
       R, N = x0.n_replicas, graph.n
       W = (R + 31) // 32
       VW = min(8, 1 << (W - 1).bit_length())
       Wp = -(-W // VW) * VW
       L = tables.lanes(R) if R <= 32 else 32
       narrow = L < 32
       tab = tables.build(graph, rules, x0, clamp, seed, noise_period, Wp, L)
       tab.defines["VW"] = VW
       fn = _load(tab.source(PRELUDE, WRAPPER)).llna_cpu
       fn.restype = None

       nb = (R + 7) // 8
       rec = 0 if record == "final" else record
       F = steps // rec + 1 if rec else 1
       # narrow: the kernel writes frames itself; else rows [N, nb], which are the frame bytes for R >= 17
       out = np.empty((F, frame_bytes(N, R)) if narrow else (F, N, nb), np.uint8)
       out[0] = x0.bits.reshape(out.shape[1:])
       state = tab.arrays["state"]
       bufs = [state.copy(), np.empty_like(state)]
       fixed = [_ptr(tab.arrays[k]) for k in ARRAYS]
       L_, U, C = ctypes.c_long, ctypes.c_uint32, ctypes.c_int

       def call(cur, nxt, i0, i1, w0, w1, nsteps, t, rec_, frame0):
           fn(*fixed, _ptr(cur), _ptr(nxt), _ptr(out), L_(N), L_(nb), L_(Wp), U(tab.k0), U(tab.k1), U(t),
              L_(i0), L_(i1), L_(w0), L_(w1), C(nsteps), C(rec_), L_(frame0))

       def ranges(stop, unit, parts):
           edges = np.unique(np.linspace(0, stop, min(stop, parts) + 1).astype(int)) * unit
           return list(zip(edges[:-1], edges[1:], strict=True))

       nthreads = _threads(threads)
       groups = Wp // VW
       # nodes mode pays ~0.3 ms of thread synchronisation per step: worth it only for big steps (M4-measured)
       big = groups >= 2 * nthreads or N * Wp < 1 << 21
       mode = os.environ.get("FAST_LLNA_CPU_MODE") or ("words" if big else "nodes")
       nw = state.size  # narrow: output words per step
       with ThreadPoolExecutor(nthreads) as pool:
           if mode == "words":  # each task owns a word slice for all steps (narrow: one task)
               parts = [(0, nw, 0, 1)] if narrow else [(0, N, w0, w1) for w0, w1 in ranges(groups, VW, 4 * nthreads)]
               tasks = [(*bufs, *p, steps, t0, rec, 1) for p in parts]
               list(pool.map(lambda a: call(*a), tasks))
               final = bufs[steps % 2]
           else:  # every step split over node ranges (narrow: word ranges aligned to 128-byte lines)
               if narrow:
                   parts = [(i0, min(i1, nw), 0, 1) for i0, i1 in ranges(-(-nw // 32), 32, 4 * nthreads)]
               else:
                   parts = [(i0, i1, 0, Wp) for i0, i1 in ranges(N, 1, 4 * nthreads)]
               for s in range(steps):
                   rec_ = 1 if rec and (s + 1) % rec == 0 else 0
                   f = (s + 1) // rec if rec_ else 0
                   list(pool.map(lambda a: call(*a), [(*bufs, *p, 1, t0 + s, rec_, f) for p in parts]))
                   bufs.reverse()
               final = bufs[0]
       if record == "final":
           fin = final.view(np.uint8)
           out[0] = fin[: out.shape[1]] if narrow else fin.reshape(N, 4 * Wp)[:, :nb]
       return out if narrow else out.reshape(F, -1)
   ```

3. Add `from ..states import frame_bytes` to the imports.

- [ ] **Step 5: Run the full suite and verify it passes**

Run: `conda run -n fast-llna ruff check src tests benchmarks && conda run -n fast-llna pytest -q`
Expected: all pass. CPU now runs R ≤ 16 in the narrow layout; Metal and CUDA still use L = 32.

- [ ] **Step 6: Mutation check (RED for the narrow gather)**

Temporarily change `& 31)) & LLNA_LMASK)` in `LLNA_GET` to `& 30)) & LLNA_LMASK)`.

Run: `conda run -n fast-llna pytest -q tests/test_backends.py -k cpu`

Expected: FAIL in `ring5-R1`, `twins-R2-noise1`, `even4-R3-clamp`, `stochastic-R5` and `uniform5-R9-record`. Revert the edit and re-run to confirm PASS.

- [ ] **Step 7: Commit**

```bash
git add -A src tests
git commit -m "Narrow node-major lanes in llna.h and the CPU host

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Metal node kernel with cooperative hubs

**Files:**
- Modify: `src/fast_llna/kernels/llna.h` (`#if LLNA_COOP` butterfly)
- Modify: `src/fast_llna/kernels/tables.py` (`LLNA_HUB`, `LLNA_COOP` defines)
- Modify: `src/fast_llna/kernels/metal.py` (`NODE_BODY`, prelude macro, run)
- Test: `tests/_graphs.py`, `tests/test_backends.py`, `tests/test_dialect.py`

**Interfaces:**
- Consumes `tables.lanes` and `build(..., L)` from Task 3, and `to_frame` from Task 2.
- Produces:
  - the defines `LLNA_HUB` and `LLNA_COOP` (0 by default; GPU hosts set 1);
  - the host macro `LLNA_SHFL_XOR(x, m)`;
  - `_graphs.degree_graph(degrees, seed)` and `_graphs.star_hub(n, seed, hub=0)`.

- [ ] **Step 1: Write the failing tests**

In `tests/_graphs.py`, change `star_hub` to accept `hub=0`, and add `degree_graph`:

```python
def star_hub(n, seed, hub=0):
    """Node `hub` reads every other node (in-degree n - 1); the others read it and 0-3 random nodes."""
    rng = np.random.default_rng(seed)
    A = sp.lil_array((n, n), dtype=np.int8)
    A[0, 1:] = 1
    for i in range(1, n):
        A[i, 0] = 1
        A[i, rng.choice(np.delete(np.arange(1, n), i - 1), size=rng.integers(0, 4), replace=False)] = 1
    p = np.arange(n)
    p[[0, hub]] = p[[hub, 0]]
    return fl.Graph(sp.csr_array(A)[p][:, p])


def degree_graph(degrees, seed):
    """Directed graph in which node i reads degrees[i] distinct random other nodes."""
    rng = np.random.default_rng(seed)
    n = len(degrees)
    rows = np.repeat(np.arange(n), degrees)
    cols = np.concatenate(
        [rng.choice(np.delete(np.arange(n), i), size=d, replace=False) for i, d in enumerate(degrees)]
    )
    return fl.Graph(sp.coo_array((np.ones(len(rows)), (rows, cols)), shape=(n, n)))
```

In `tests/test_backends.py`:
- change the import to `from _graphs import degree_graph, random_digraph, star_hub`;
- add the following below `clamp`;
- append the hub cases to `CASES`.

```python
# mean degree 3.95 -> LLNA_HUB = 32: degree 32 takes the normal path, 33, 34, 64 and 140 the cooperative one;
# the 140-degree hub is node 149 of 150, in the last (partial) SIMD group
HUBS = degree_graph([2] * 7 + [32] + [2] * 32 + [33, 34] + [2] * 57 + [64] + [2] * 49 + [140], seed=21)


def test_hub_threshold_scales_with_mean_degree():
    from fast_llna.kernels import tables

    rule = fl.life_like(5, 6, 28)
    assert tables.build(HUBS, rule, fl.random_states(150, 1), None, 0, None, 1, 1).defines["LLNA_HUB"] == 32
    dense = fl.ring(300, 20)  # degree 40
    assert tables.build(dense, rule, fl.random_states(300, 1), None, 0, None, 1, 1).defines["LLNA_HUB"] == 160
```

```python
    (
        "hubs-R1-stochastic-clamp",
        HUBS,
        1,
        lambda R: stochastic(fl.symmetric(5), R, 21),
        8,
        {"clamp": clamp(1, 150, 21), "seed": 21},
    ),
    ("hubs-R3-mixed", HUBS, 3, lambda R: mixed(9, R, 22), 8, {"record": 2}),
    ("hubs-R32-majority", HUBS, 32, lambda R: stochastic(fl.MAJORITY, R, 23), 8, {"seed": 23}),
    ("star-last-R4", star_hub(200, 24, hub=199), 4, lambda R: fl.life_like(9, 72, 12), 6, {}),
```

`HUBS` is used inside `CASES`, so define it before `CASES`.

In `tests/test_dialect.py`, add the cooperative variant to the parametrisation:

```python
@pytest.mark.parametrize(
    "lanes",
    [
        "#define LLNA_L 1\n#define LLNA_LMASK 0x1u\n#define LLNA_COOP 0\n",
        "#define LLNA_L 32\n#define LLNA_LMASK 0xffffffffu\n#define LLNA_COOP 0\n",
        "#define LLNA_L 4\n#define LLNA_LMASK 0xfu\n#define LLNA_COOP 1\n#define LLNA_SHFL_XOR(x, m) (x)\n",
    ],
)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `conda run -n fast-llna pytest -q tests/test_backends.py tests/test_dialect.py -k "hub or star-last or parses"`

Expected:
- `test_hub_threshold_scales_with_mean_degree` fails with `KeyError: 'LLNA_HUB'`.
- The hub CASES pass on cpu and on metal (Metal is still v1), so they pin behaviour.
- The COOP dialect variant passes trivially until the header uses `LLNA_COOP`.

- [ ] **Step 3: Implement the header butterfly**

In `src/fast_llna/kernels/llna.h`, directly after the edge loop's closing brace (before `WORD s = LLNA_GET(S, i);`), insert:

```c
#if LLNA_COOP
    if (coop > 1) { /* sum the coop lanes' partial counts: XOR butterfly of bit-sliced ripple adds */
        int kb = 0;
        while ((k >> kb) != 0) kb++; /* bitlen(k): every partial sum and the total fit in kb planes */
        for (int sh = 1; sh < coop; sh <<= 1) {
            WORD carry = ZEROW;
            for (int p = 0; p < PMAX; p++) {
                if (p >= kb) break;
                WORD a_ = c[p], y_ = LLNA_SHFL_XOR(a_, sh), ab_ = a_ ^ y_;
                c[p] = ab_ ^ carry;
                carry = (a_ & y_) | (carry & ab_);
            }
        }
        lim = kb;
    }
#endif
```

Add to the header comment: `LLNA_COOP (1: coop > 1 lanes may share a node; needs LLNA_SHFL_XOR(x, m), the value of x in lane ^ m)`.

- [ ] **Step 4: Implement the tables defines**

In `tables.build`'s `defines`, add:

```python
            "LLNA_COOP": 0,
            "LLNA_HUB": max(32, 4 * int(np.ceil(deg.mean()))),
```

- [ ] **Step 5: Implement the Metal node kernel**

In `src/fast_llna/kernels/metal.py`:

1. Update the module docstring:

   ```
   """Apple GPU backend: llna.h as MLX custom Metal kernels, one launch per step.

   R <= 32: one thread per node; a SIMD group counts each hub (degree > LLNA_HUB) together, then assembles
   32/L nodes per output word with XOR shuffles. R > 32: thread (x, y) updates replica word x of node y.
   """
   ```

2. Add to `PRELUDE`:

   ```
   #define LLNA_SHFL_XOR(x, m) simd_shuffle_xor((x), (ushort)(m))
   ```

3. Change the BODY call to keep `0, 1`, and add:

   ```python
   NODE_BODY = """
       uint i = thread_position_in_grid.x, lane = thread_index_in_simdgroup, n = params[0];
       bool hub = i < n && indptr[i + 1] - indptr[i] > LLNA_HUB;
       uint hubs = (uint)((simd_vote::vote_t)simd_ballot(hub));
       WORD out = 0u;
       while (hubs != 0u) {  /* the whole SIMD group counts each hub; its own lane keeps the result */
           uint h = ctz(hubs);
           hubs &= hubs - 1u;
           WORD o = llna_update(S, indptr, indices, seg_off, seg_thr, seg_cell, one, hasf, frac, stream, cm, cv,
                                (llna_idx)1, params[2], params[3], params[4], (int)(i - lane + h), 0, (int)lane, 32);
           if (lane == h) out = o;
       }
       if (i < n && !hub)
           out = llna_update(S, indptr, indices, seg_off, seg_thr, seg_cell, one, hasf, frac, stream, cm, cv,
                             (llna_idx)1, params[2], params[3], params[4], (int)i, 0, 0, 1);
       WORD v = out << ((i * LLNA_L) & 31u);
       for (uint sh = 1u; sh < 32u / LLNA_L; sh <<= 1u) v |= simd_shuffle_xor(v, (ushort)sh);
       ulong wi = ((ulong)i * LLNA_L) >> 5;
       if ((lane & (32u / LLNA_L - 1u)) == 0u && wi < params[1]) nxt[wi] = v;
   """
   ```

4. Change `_kernel(header)` to `_kernel(header, body)`, hash `header + body`, and pass `source=body`.

5. Replace `run`'s setup and frame code. The step loop stays, with `params` built from `p1`:

   ```python
   def run(graph, rules, x0, steps, record, clamp, seed, t0, noise_period, threads=None):
       import mlx.core as mx

       R, N = x0.n_replicas, graph.n
       Wp = tables.gpu_words(R)
       node = Wp == 1  # R <= 32: one thread per node, L bits per node
       L = tables.lanes(R) if node else 32
       tab = tables.build(graph, rules, x0, clamp, seed, noise_period, Wp, L)
       tab.defines["LLNA_COOP"] = 1
       kernel = _kernel(tab.source(PRELUDE, ""), NODE_BODY if node else BODY)
       fixed = [mx.array(_pad(tab.arrays[k])) for k in INPUTS[1:-1]]
       state = tab.arrays["state"]
       if node:
           grid, group, p1 = (-(-N // 256) * 256, 1, 1), (256, 1, 1), state.size
       else:
           bx = min(Wp, 32)
           grid, group, p1 = (Wp, N, 1), (bx, max(1, min(256 // bx, N)), 1), Wp

       rec = 0 if record == "final" else record
       frames = np.empty((steps // rec + 1 if rec else 1, frame_bytes(N, R)), np.uint8)
       frames[0] = x0.bits
       pending = []  # (frame index, lazy mx array)

       def frame(a):
           return tables.to_frame(np.asarray(a).view(np.uint8), N, R, L * Wp)

       def flush():
           mx.eval(cur, *(a for _, a in pending))
           for f, a in pending:
               frames[f] = frame(a)
           pending.clear()

       cur = mx.array(_pad(state))
       size = cur.size
       for s in range(steps):
           params = mx.array(np.array([N, p1, tab.k0, tab.k1, (t0 + s) % 2**32], np.uint32))
           (cur,) = kernel(
               inputs=[cur, *fixed, params],
               grid=grid,
               threadgroup=group,
               output_shapes=[(size,)],
               output_dtypes=[mx.uint32],
           )
           if rec and (s + 1) % rec == 0:
               pending.append(((s + 1) // rec, cur))
           if (s + 1) % EVAL_EVERY == 0:
               flush()
       flush()
       if record == "final":
           frames[0] = frame(cur)
       return frames
   ```

- [ ] **Step 6: Run the full suite and verify it passes**

Run: `conda run -n fast-llna ruff check src tests benchmarks && conda run -n fast-llna pytest -q`
Expected: all pass, with every metal case bit-exact.

- [ ] **Step 7: Mutation checks**

For each of the four mutations below:
1. Apply the edit.
2. Run `conda run -n fast-llna pytest -q tests/test_backends.py -k metal`.
3. Confirm the expected failure.
4. Revert.

| # | Mutation | Expected failures |
|---|---|---|
| 1 | In the butterfly, change `carry = (a_ & y_) \| (carry & ab_);` to `carry = a_ & y_;` | the `hubs-*` and `star-last-R4` cases |
| 2 | In `NODE_BODY`, change `(int)(i - lane + h)` to `(int)(i - lane + h + 1u)` | the hub cases |
| 3 | In `NODE_BODY`, change `sh < 32u / LLNA_L` to `sh < 16u / LLNA_L` | the R ≤ 2 cases: `ring5-R1`, `twins-R2-noise1`, `hubs-R1-*` |
| 4 | Change the hub test to `hub = false` | no failure (hubs only affect speed) |

Mutation 4 is a control. Confirm it passes, and record in the commit message that hub cooperation is exercised (mutations 1 and 2 failed).

- [ ] **Step 8: Commit**

```bash
git add -A src tests
git commit -m "Metal node kernel: narrow lanes, SIMD-cooperative hubs

Mutation-checked: butterfly carry and hub index break the hub cases.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: CUDA node kernel, and removal of transitional conversions

**Files:**
- Modify: `src/fast_llna/kernels/cuda.py`
- Modify: `src/fast_llna/kernels/tables.py` (delete `rows_to_frames`, and the generic branches of `to_kernel`/`to_frame`)
- Test: `tests/test_dialect.py`

**Interfaces:**
- Consumes everything from Tasks 1–4.
- Produces the CUDA kernels `llna_node(S, ..., nxt, llna_idx n, llna_idx nw, u32 k0, u32 k1, u32 t)` and `llna_step` (v1 word kernel, with the same parameter list where the sixth-to-last argument is Wp).

- [ ] **Step 1: Write the failing test**

In `tests/test_dialect.py`:

1. Extend `CUDA_STUB`:

   ```python
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
   ```

2. Parametrise `test_generated_cuda_source_parses` over `R` ∈ {3, 257}, building with `L = tables.lanes(R) if R <= 32 else 32`:

   ```python
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
   ```

- [ ] **Step 2: Run the test and verify it fails**

Run: `conda run -n fast-llna pytest -q tests/test_dialect.py -k cuda`
Expected: FAIL with `use of undeclared identifier 'LLNA_SHFL_XOR'`, because the CUDA prelude lacks it while `LLNA_COOP` is 1.

- [ ] **Step 3: Implement the CUDA host**

In `src/fast_llna/kernels/cuda.py`:

1. Replace the module docstring's first paragraph:

   ```
   """NVIDIA GPU backend: llna.h compiled by NVRTC through CuPy, one launch per step.

   R <= 32 (llna_node): one thread per node in blocks of 256; a warp counts each hub (degree > LLNA_HUB)
   together and assembles 32/L nodes per output word with __shfl_xor_sync. R > 32 (llna_step): block
   (bx, by) = (min(Wp, 32), 256 / bx), a warp is one node x 32 consecutive replica words once Wp >= 32.
   Set FAST_LLNA_DUMP=path to write the generated CUDA source (for nvcc -Xptxas -v / SASS inspection).
   """
   ```

2. Add to `PRELUDE`: `#define LLNA_SHFL_XOR(x, m) __shfl_xor_sync(0xffffffffu, (x), (m))`.

3. Append the node kernel to `WRAPPER`. The existing `llna_step` already passes `w, 0, 1` from Task 3.

   ```c
   extern "C" __global__ void llna_node(const u32 *S, const int *indptr, const int *indices, const int *seg_off,
                                        const int *seg_thr, const int *seg_cell, const u32 *one, const u32 *hasf,
                                        const u32 *frac, const u32 *stream, const u32 *cm, const u32 *cv, u32 *nxt,
                                        llna_idx n, llna_idx nw, u32 k0, u32 k1, u32 t) {
       llna_idx i = (llna_idx)blockIdx.x * blockDim.x + threadIdx.x;
       int lane = (int)(threadIdx.x & 31u);
       int hub = i < n && indptr[i + 1] - indptr[i] > LLNA_HUB;
       unsigned hubs = __ballot_sync(0xffffffffu, hub);
       WORD out = 0u;
       while (hubs) { /* the whole warp counts each hub; its own lane keeps the result */
           int h = __ffs(hubs) - 1;
           hubs &= hubs - 1u;
           WORD o = llna_update(S, indptr, indices, seg_off, seg_thr, seg_cell, one, hasf, frac, stream, cm, cv, 1,
                                k0, k1, t, (int)(i - lane + h), 0, lane, 32);
           if (lane == h) out = o;
       }
       if (i < n && !hub)
           out = llna_update(S, indptr, indices, seg_off, seg_thr, seg_cell, one, hasf, frac, stream, cm, cv, 1,
                             k0, k1, t, (int)i, 0, 0, 1);
       WORD v = out << ((i * LLNA_L) & 31);
       for (int sh = 1; sh < 32 / LLNA_L; sh <<= 1) v |= __shfl_xor_sync(0xffffffffu, v, sh);
       llna_idx wi = (i * LLNA_L) >> 5;
       if ((lane & (32 / LLNA_L - 1)) == 0 && wi < nw) nxt[wi] = v;
   }
   ```

4. Change `_kernel` to return the module: `return cp.RawModule(code=source, options=("-std=c++14",))`.

5. Replace `run`:

   ```python
   def run(graph, rules, x0, steps, record, clamp, seed, t0, noise_period, threads=None):
       import cupy as cp

       R, N = x0.n_replicas, graph.n
       Wp = tables.gpu_words(R)
       node = Wp == 1  # R <= 32: one thread per node, L bits per node
       L = tables.lanes(R) if node else 32
       tab = tables.build(graph, rules, x0, clamp, seed, noise_period, Wp, L)
       tab.defines["LLNA_COOP"] = 1
       source = tab.source(PRELUDE, WRAPPER)
       if os.environ.get("FAST_LLNA_DUMP"):
           with open(os.environ["FAST_LLNA_DUMP"], "w") as f:
               f.write(source)
       kernel = _kernel(source).get_function("llna_node" if node else "llna_step")
       fixed = [cp.asarray(tab.arrays[k]) for k in ARRAYS]
       cur = cp.asarray(tab.arrays["state"])
       nxt = cp.empty_like(cur)
       if node:
           grid, block, p1 = (-(-N // 256),), (256,), cur.size
       else:
           bx = min(Wp, 32)
           by = 256 // bx
           grid, block, p1 = ((N + by - 1) // by, Wp // bx), (bx, by), Wp

       rec = 0 if record == "final" else record
       frames = np.empty((steps // rec + 1 if rec else 1, frame_bytes(N, R)), np.uint8)
       frames[0] = x0.bits
       # keep recorded frames on the GPU when they fit comfortably, else copy each one to the host
       on_device = rec and frames.nbytes < 0.5 * cp.cuda.Device().mem_info[0]
       dframes = cp.empty(frames.shape, np.uint8) if on_device else None

       def frame(a):
           return tables.to_frame(a.view(np.uint8), N, R, L * Wp)

       for s in range(steps):
           args = (cur, *fixed, nxt, np.int64(N), np.int64(p1), np.uint32(tab.k0), np.uint32(tab.k1),
                   np.uint32((t0 + s) % 2**32))
           kernel(grid, block, args)
           cur, nxt = nxt, cur
           if rec and (s + 1) % rec == 0:
               if on_device:
                   dframes[(s + 1) // rec] = frame(cur)
               else:
                   frames[(s + 1) // rec] = cp.asnumpy(frame(cur))
       if on_device:
           frames[1:] = cp.asnumpy(dframes[1:])
       if record == "final":
           frames[0] = cp.asnumpy(frame(cur))
       return frames
   ```

6. Add `from ..states import frame_bytes`.

- [ ] **Step 4: Remove the transitional conversions**

Every host now uses `L = lanes(R)` for R ≤ 32. Kernel bits therefore equal frame bits, or both are whole bytes. In `src/fast_llna/kernels/tables.py`:
- delete `rows_to_frames`;
- in `to_kernel`, replace the `else:` branch with `else: raise AssertionError(f"no byte-aligned path from {Ls} to {Lk} bits per node")`;
- in `to_frame`, replace the last `return` with the same `raise`;
- delete `unpack` from the `..states` import if it is now unused, and run `ruff check` to confirm.

- [ ] **Step 5: Run the full suite and verify it passes**

Run: `conda run -n fast-llna ruff check src tests benchmarks && conda run -n fast-llna pytest -q`
Expected: all pass, including both CUDA dialect parametrisations. CUDA cannot run here.

- [ ] **Step 6: Commit**

```bash
git add -A src tests
git commit -m "CUDA node kernel with warp-cooperative hubs; drop transitional layout conversions

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Benchmarks, tuning by evidence, acceptance, docs

**Files:**
- Modify: `benchmarks/graphs.py` (`chung_lu`)
- Modify: `benchmarks/bench.py` (`small_R` scenario)
- Possibly modify: `src/fast_llna/kernels/cpu.py` (mode threshold) and `llna.h` (L = 1 scalar count), only per the decision rules in Steps 4–5
- Modify: `README.md`, `docs/design.md`

**Interfaces:**
- Produces `graphs.chung_lu(n, gamma, mean_degree, seed) -> fl.Graph` and `bench.py --scenario small_R`.

- [ ] **Step 1: Add the generator and the scenario**

In `benchmarks/graphs.py`:

```python
def chung_lu(n, gamma, mean_degree, seed=0):
    """Power-law graph (Chung-Lu): endpoints drawn proportionally to weights i^(-1 / (gamma - 1)); undirected,
    nodes relabelled at random so hubs are not adjacent in memory; isolated nodes get one edge."""
    rng = np.random.default_rng(seed)
    w = np.arange(1, n + 1) ** (-1 / (gamma - 1))
    p = w / w.sum()
    m = int(n * mean_degree / 2)
    i, j = rng.choice(n, m, p=p), rng.choice(n, m, p=p)
    keep = i != j
    i, j = i[keep], j[keep]
    lonely = np.setdiff1d(np.arange(n), np.r_[i, j])
    perm = rng.permutation(n)
    A = _undirected(n, perm[np.r_[i, lonely]], perm[np.r_[j, (lonely + 1) % n]])
    return fl.Graph(A)
```

In `benchmarks/bench.py`:
1. Change the import to `from graphs import barabasi_albert, chung_lu, erdos_renyi, rewired_torus  # noqa: E402`.
2. Add the scenario.
3. Register it in `SCENARIOS`, after `huge_N`.

```python
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
```

Run: `conda run -n fast-llna ruff check benchmarks`. Expected: clean.

- [ ] **Step 2: Measure the baseline (commit 46f8e0d) in a worktree**

```bash
SP=/private/tmp/claude-501/-Users-michielrollier-Developer-GitHub-fast-llna/a13a5ed9-6f49-48af-9e37-77cd99ac6578/scratchpad
git worktree add "$SP/baseline" 46f8e0d
cp benchmarks/bench.py benchmarks/graphs.py "$SP/baseline/benchmarks/"
cd "$SP/baseline" && PYTHONPATH=src conda run -n fast-llna python -c "import fast_llna; print(fast_llna.__file__)"
```

Expected: a path under `$SP/baseline/src`. If it prints the main repository path, the editable install shadows `PYTHONPATH`. In that case run with `conda run -n fast-llna python -c "import sys; sys.path.insert(0, 'src'); import runpy; sys.argv = ['bench.py', '--scenario', 'small_R', '--backend', 'cpu']; runpy.run_path('benchmarks/bench.py', run_name='__main__')"` instead.

```bash
for b in cpu metal; do for s in small_R thesis; do
  PYTHONPATH=src conda run -n fast-llna python benchmarks/bench.py --scenario $s --backend $b >> "$SP/base.jsonl"
done; done
cd - && git worktree remove "$SP/baseline"
```

- [ ] **Step 3: Measure the new code and compare**

```bash
SP=/private/tmp/claude-501/-Users-michielrollier-Developer-GitHub-fast-llna/a13a5ed9-6f49-48af-9e37-77cd99ac6578/scratchpad
for b in cpu metal; do for s in small_R thesis; do
  conda run -n fast-llna python benchmarks/bench.py --scenario $s --backend $b >> "$SP/new.jsonl"
done; done
conda run -n fast-llna python - "$SP/base.jsonl" "$SP/new.jsonl" <<'PY'
import json, sys
load = lambda p: {(r["scenario"], r["backend"], r["N"], r["R"], r["record"]): r for r in map(json.loads, open(p)) if "N" in r}
base, new = load(sys.argv[1]), load(sys.argv[2])
for k in sorted(new):
    b, n = base.get(k), new[k]
    ms = lambda r: 1e3 * r["steady_s"] / r["T"]
    print(f"{k[0]:18s} {k[1]:6s} N={k[2]:>9,} R={k[3]:>6} rec={k[4]!s:6s} base {ms(b) if b else float('nan'):9.3f} ms/step  new {ms(n):9.3f}  ratio {ms(n) / ms(b) if b else float('nan'):6.2f}")
PY
```

Acceptance criteria. Check each and write the numbers into the commit message:
1. Metal, `small_R` at N = 1e7, R = 1: ≤ 10 ms/step. `small_R_power_law` on Metal: ≤ 10 ms/step.
2. CPU, `small_R` at N = 1e7, R = 1: ≤ 30 ms/step.
3. No row has a ratio > 1.10. `thesis` rows have a ratio ≤ 1.10.

If criterion 3 fails for a cell, stop and report the cell with both numbers. Do not tune silently; the spec says an override or heuristic needs evidence and the user's decision.

- [ ] **Step 4: CPU mode threshold (decision by evidence)**

```bash
for mode in words nodes; do FAST_LLNA_CPU_MODE=$mode conda run -n fast-llna python - <<'PY'
import os, sys, time; sys.path.insert(0, "benchmarks")
import numpy as np, fast_llna as fl
from graphs import erdos_renyi
for N in (10_000, 100_000, 1_000_000):
    g = erdos_renyi(N, 8, seed=N); x = fl.States.from_bool(np.random.default_rng(0).random((1, N)) < 0.5)
    fl.simulate(g, fl.life_like(5, 6, 28), x, 5, record="final", backend="cpu")
    t = time.perf_counter(); fl.simulate(g, fl.life_like(5, 6, 28), x, 50, record="final", backend="cpu")
    print(os.environ["FAST_LLNA_CPU_MODE"], N, f"{(time.perf_counter() - t) / 50 * 1e3:.3f} ms/step")
PY
done
```

Decision rule: let N* be the smallest N at which `nodes` is ≥ 20% faster than `words`.
- If N* < 2^21 (≈ 2.1e6), change `big = groups >= 2 * nthreads or N * Wp < 1 << 21` in `cpu.py` to `big = groups >= 2 * nthreads or graph.indices.size * Wp < E` with `E = 8 · N*` (8 = mean degree here). Then update the comment to `# nodes mode pays ~0.3 ms of thread synchronisation per step: worth it from ~E edge-words (M4-measured)`.
- Otherwise leave the line unchanged.

Re-run `pytest -q` after any change.

- [ ] **Step 5: L = 1 scalar count (decision by evidence)**

The spike's scalar bitset kernel ran CPU ER N = 1e7 R = 1 at 26.6 ms/step. If Step 3's CPU N = 1e7 R = 1 figure is > 30.6 ms/step (more than 15% slower), replace the edge loop in `llna.h` with this `#if`. Otherwise skip this step.

```c
#if LLNA_L == 1 /* one replica: count in a WORD integer, then spread into planes */
    WORD q = ZEROW;
    for (int e = lane; e < k; e += coop) {
        q += LLNA_GET(S, indices[lo + e]);
        m++;
    }
    while ((m >> lim) != 0) lim++;
    for (int p = 0; p < PMAX; p++) c[p] = (q >> p) & 1u;
#else
    /* existing bit-sliced loop */
#endif
```

Then re-run `pytest -q` (all backends must stay bit-exact) and re-measure.

- [ ] **Step 6: Update the docs**

In `README.md`:

1. Replace the **Output** paragraph with:

   ```
   **Output**

   Each recorded configuration is a flat little-endian bit array `traj.states.bits[t]` in which replica r of node
   i is bit `i·L + r`. L, the bits per node, is the next power of two for R ≤ 8 (1, 2, 4 or 8) and whole bytes,
   8·⌈R/8⌉, beyond: one replica of N = 1e7 nodes takes 1.25 MB per frame. `traj.states.to_bool()` unpacks to
   `[T, R, N]`.
   ```

2. In **Options**, add: `` - `noise_period`: replicas r and r + p share random draws (defect twins under stochastic rules); p divides 32 or is a multiple of 32. ``

3. In **How it gets there**, add these bullets:

   ```
   - With R ≤ 32 a node holds only L = pow2ceil(R) bits, so huge graphs with few replicas run from cache.
   - On GPUs a SIMD group counts each hub's edges together, so hubs do not serialise one thread.
   ```

4. Replace the "Huge single networks with few replicas are memory-bound…" paragraph with one sentence giving the measured N = 1e7, R = 1 ER, power-law and Moore numbers from Step 3. Add a row to the performance table for `ER N=1e7, R=1, per step`, with the new CPU and Metal numbers.

In `docs/design.md`, change the line `**Single strategy in v1:** replicas as lanes, which also covers R=1 at a constant-factor cost.` to:

```
**Strategy:** replicas as lanes. Since 2026-10-07, R ≤ 32 uses narrow node-major lanes (L = pow2ceil(R) bits per node)
and GPU-cooperative hubs; see `docs/superpowers/specs/2026-10-07-narrow-lanes-design.md`.
```

Also add one line under **Canonical public format**: `Superseded by the flat node-major layout of the narrow-lanes spec (frames of ceil(N·L/8) bytes).`

- [ ] **Step 7: Final verification and commit**

Run: `conda run -n fast-llna ruff check src tests benchmarks && conda run -n fast-llna pytest -q`
Expected: all pass.

```bash
git add -A benchmarks src README.md docs/design.md
git commit -m "Benchmarks for few replicas; docs for narrow lanes

<paste the acceptance numbers from Step 3 and the decisions from Steps 4-5>

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
