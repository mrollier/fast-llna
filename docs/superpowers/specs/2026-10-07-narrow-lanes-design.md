# Narrow lanes: node-major state with L bits per node

Status: design approved 2026-10-07; spec under review.

## Problem

The v1 kernels store the state as uint32 words `[N][Wp]`: replica r of node i is bit r % 32 of word
`i * Wp + r // 32`. A node therefore occupies at least 32 bits even when R = 1. For huge graphs with few
replicas, that state no longer fits in cache, and every neighbour read becomes a DRAM access.

At N = 1e7 with R ≤ 32, the state is 40 MB. On the M4, Metal takes 61 ms per step on an Erdős–Rényi graph
with mean degree 8, which is 19× the time needed just to stream the graph's index array.

## Evidence (spike, Apple M4, N = 1e7, deterministic rule φ⁵₆,₂₈)

All prototypes were bit-identical to the current CPU backend.

Metal, ms per step, by lane width L (bits per node) on the ER graph. The rows emulate the gather cost of each
footprint:

| L | state | ms/step |
|---|---|---|
| 1 | 1.25 MB | 8.6 |
| 2 | 2.5 MB | 14.3 |
| 4 | 5 MB | 26.3 |
| 8 | 10 MB | 36.7 |
| 16 | 20 MB | 50.1 |
| 32 (v1) | 40 MB | 65.5 |
| floor: stream `indptr` + `indices` only | | 3.25 |

Other measurements:

- **Moore torus (3163²), R = 1.** Current Metal takes 13.2 ms; a bitset kernel takes 3.3 ms, which is the
  index-stream floor.
- **Power-law graph (Chung–Lu, γ = 2.5, max degree 116k), R = 1.**
  - A bitset kernel with one thread per node takes 21.3 ms: hub rows serialise one thread, and the floor
    kernel shows the same effect (9.4 ms).
  - With SIMD-group-cooperative counting of nodes whose degree exceeds 32, it takes 8.4 ms. ER is unaffected
    (8.9 ms).
- **GPU output assembly.** One thread per node with a SIMD ballot/shuffle beats one thread per output word
  looping over 32 nodes: 8.5 ms vs 14.9 ms.
- **CPU, 10 threads.** ER goes from 160 to 26.6 ms, and Moore from 49 to 21.8 ms. The CPU per-node loop is
  instruction-bound: one thread streams the indices at 5 GB/s.

Conclusion: footprint, not bit width or gather count, sets the cost. A layout whose footprint scales with R
helps every R < 32, not only R = 1.

## Design

### 1. Layout

For R ≤ 32, every node holds `L = min(32, pow2ceil(R))` bits. Replica r of node i is bit `i·L + r` of a
little-endian bit array, stored as uint32 words. L = 1 is the node bitset ("nodes as lanes"), and L = 32
equals the v1 layout with Wp = 1. Padding lanes (R ≤ r < L) are 0.

For R > 32, the v1 layout `[N][Wp]` is unchanged.

L follows from R. There is no strategy heuristic and no override, because L ≤ 32 never loads more data than
v1. The benchmark gate below checks that claim, and an override is added only if it fails.

### 2. Kernel core (`llna.h`)

`llna_update` keeps its structure. The changes are:

- **State gathers.** All gathers of state and clamp words (neighbours, own state, `CM`, `CV`) go through one
  macro:
  ```c
  #if LLNA_L < 32   /* node-major L-bit fields; Wp == 1, w == 0 */
  #define LLNA_GET(P, j) ((LOADW(P, ((llna_idx)(j) * LLNA_L) >> 5) >> (u32)(((llna_idx)(j) * LLNA_L) & 31)) & LLNA_LMASK)
  #else
  #define LLNA_GET(P, j) LOADW(P, (llna_idx)(j) * Wp + w)
  #endif
  ```
- **Strided edge loop.** `llna_update` gains `(int lane, int coop)` and counts edges `lo + lane, lo + lane +
  coop, …`. The plane bound `lim` follows the number of edges this lane has added. With `coop = 1`, this is
  exactly the v1 loop.
- **Cooperative sum (`#if LLNA_COOP`).** When `coop > 1`, the lanes' bit-sliced partial counts are summed
  with an XOR butterfly. For m = 1, 2, …, coop/2, each plane is exchanged with `LLNA_SHFL_XOR(c[p], m)` and
  added with a ripple carry over `bitlen(k)` planes. Every lane ends with the full count. The rest of the
  update (threshold compares, selection, Bernoulli digits, clamp) is unchanged and runs on that count.
- **Unchanged.** The masks `ONE`, `HASF`, `FRAC` and `STREAM` keep their v1 meaning; with Wp = 1, replica r
  is bit r of word 0. Randomness uses the same Philox digit words, so v1 and narrow runs are bit-identical
  and both match the reference.

Host defines: `LLNA_L`, `LLNA_LMASK`, `LLNA_COOP` (0 on CPU), `LLNA_HUB` (see below), and the host macro
`LLNA_SHFL_XOR`. On the CPU, WORD stays a 1-lane vector for R ≤ 32, as in v1.

### 3. GPU hosts (Metal, CUDA): node kernel for R ≤ 32

- **Grid.** The grid is 1-D with one thread per node, padded to a multiple of 256, and threadgroups of 256.
  All SIMD groups are therefore full and lane = i % 32. Threads with i ≥ N take part in shuffles with k = 0.
- **Hubs.**
  1. A ballot marks the lanes whose node has degree > `LLNA_HUB`.
  2. For each marked lane h (ascending), the whole SIMD group calls `llna_update` for node
     `i − lane + h` with `coop = 32`, and lane h keeps the result.
  3. Then every non-hub lane with i < N updates its own node with `coop = 1`.
- **Hub threshold.** `LLNA_HUB = max(32, 4·⌈mean degree⌉)`. Cooperation costs a 5-step butterfly per hub,
  so the threshold scales with typical degrees. The spike's optimum on ER and power-law graphs was 32.
- **Output assembly.** Each lane shifts its L output bits to `(i·L) % 32`. A butterfly over
  `log2(32/L)` steps (`simd_shuffle_xor`, `__shfl_xor_sync`) ORs them together, and lanes with
  `lane % (32/L) == 0` store word `i·L / 32` if it lies inside the state.
- **R > 32.** The v1 2-D word kernel is unchanged. Cooperative hubs for it are a follow-up.

### 4. CPU host

- **Tasks.** For R ≤ 16 (L < 32), tasks own ranges of output words, aligned to 32 words (one 128-byte line),
  so no two threads write the same cache line. A task loops over the 32/L nodes of each word, ORs their
  outputs into a register and stores whole words. R = 17…32 keeps the v1 per-node loop (L = 32, Wp = 1).
- **Modes.** Nodes mode (per-step tasks) and words mode (one task for all steps) keep the v1 heuristic.
- **No hub handling.** It is not needed on the CPU.
- **Scalar special case.** An `LLNA_L == 1` scalar-count path is added only if benchmarks show that
  bit-sliced counting costs more than 15% at L = 1.

### 5. Output layout (public)

A recorded frame is a flat little-endian bit array. Replica r of node i is bit `i·Ls + r`, with

```
Ls = pow2ceil(R)      if R ≤ 8      (1, 2, 4 or 8 bits per node)
Ls = 8·ceil(R/8)      if R > 8      (whole bytes per node: the v1 layout, flattened)
```

- `States(bits, n_replicas, n)` stores `bits[..., ceil(N·Ls/8)]`. The node count becomes a field, because it
  can no longer be read off the shape.
- `from_bool`, `to_bool`, `Trajectory.density`, `hamming` and `final` keep their signatures and results.
- For R ≥ 8 the bytes are the v1 bytes. For R = 1 a frame is the node bitset: N = 1e7 with T = 1000 recorded
  every step takes 1.25 GB, against 10 GB in v1.
- The kernel state equals the frame bytes whenever `Ls == L` (R ≤ 16 or 25 ≤ R ≤ 32), in which case
  recording is a contiguous copy. For R = 17…24 and R > 32, recording copies `ceil(R/8)` bytes per node, as
  in v1. `tables.frame()` implements both cases for all hosts.
- `simulate`'s size guard uses the new frame size.

### 6. Shared noise for small replica counts (addition, needs approval)

In v1, `noise_period` must be a multiple of 32, so defect twins of R < 32 replicas cannot share random draws.
That matters for twins under stochastic rules (Watts majority) on huge graphs, which the narrow layout makes
cheap.

Generalised contract: replica r draws its digits from replica `r' = r mod noise_period`. That is bit
`r' % 32` of the Philox word with stream `r' // 32`.

- **Allowed periods.** `noise_period` may divide 32 (1, 2, 4, 8, 16) or be a multiple of 32.
- **Backward compatibility.** For multiples of 32 this is exactly the v1 contract.
- **Kernel cost.** For p | 32, one multiply after each digit word,
  `u = (u & (2^p − 1)) · REPL_p` (where `REPL_p` has a 1 every p bits), and `stream[w] = 0`.
- **Reference.** Implements the definition directly.
- **Helpers.** `defect_twins` documents `noise_period=R` for any R that divides 32 or is a multiple of 32.

## Testing

Test-first, with each test watched failing.

- `test_states.py`:
  - the bit position `i·Ls + r` for R ∈ {1, 2, 3, 5, 8, 9, 17, 33} and N not a multiple of 8;
  - round trips;
  - reducers against naive numpy;
  - shapes.
- `test_reference.py`: `noise_period` ∈ {1, 2, 16} makes twins share noise; other values raise.
- `test_backends.py`, new cases, each bit-exact against the reference on every backend:
  - R ∈ {2, 3, 5, 9}, covering L = 2, 4, 8, 16, with stochastic rules, clamping and record strides;
  - R = 2 twins with `noise_period=1` under the majority rule;
  - hubs with R ∈ {1, 3, 32}, using `star_hub` and a power-law graph with degrees above `LLNA_HUB`, so the
    cooperative path runs for L = 1, 4 and 32 with stochastic rules and clamping;
  - N not a multiple of 32/L, to cover partial last words.
- `test_dialect.py`: the header compiles as C++14 with `LLNA_L` ∈ {1, 32} and `LLNA_COOP` ∈ {0, 1}, and the
  generated CUDA node kernel compiles with stubs.
- Mutation checks: break the gather shift, the butterfly carry, the hub lane selection and the output
  shuffle; each must fail a test.

## Benchmarks and acceptance

New `benchmarks/graphs.py` generator: `chung_lu(n, gamma, mean_degree, seed)`.

New `bench.py` scenario `small_R`: N ∈ {1e3, 1e5, 1e6, 1e7} × R ∈ {1, 2, 4, 8, 16, 32} on ER, plus
power-law and Moore at N = 1e7, R = 1. The baseline is commit `46f8e0d`, run from a git worktree.

Acceptance on the M4:

1. All tests pass on reference, CPU and Metal; ruff is clean.
2. Metal at N = 1e7, R = 1: ≤ 10 ms/step on ER (v1: 61) and ≤ 10 ms/step on the power-law graph. CPU on ER
   at R = 1: ≤ 30 ms/step (v1: 160).
3. No (N, R) cell of `small_R` is more than 10% slower than baseline on either backend. The thesis scenario
   (R > 32) is unchanged within noise.

CUDA is verified by `benchmarks/workstation_job.sh`, which gains the new scenario.

## Out of scope (follow-ups, each needs its own evidence)

- Index-stream compression (delta-coded neighbour lists; estimated 2–4× on lattices, ~1.3× on ER) and node
  relabelling for empirical graphs. Narrow lanes leave the index stream as the main cost.
- Cooperative hubs in the R > 32 word kernel.
- A C-side barrier for CPU nodes mode (~0.3 ms per step, under 3% at N = 1e7).
