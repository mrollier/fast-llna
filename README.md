# fast-llna

Fast, exact simulation of Life-like network automata (LLNAs) on multicore CPUs, Apple GPUs (Metal) and NVIDIA GPUs (CUDA).

A node with state `s`, in-degree `k` and `q` living in-neighbours updates according to the cell of its density `q/k`. Rules are written φ^r_{β,σ}: the node is born (s=0) if bit j of β is set, and survives (s=1) if bit j of σ is set, where j is the density cell. Rules can also be stochastic, for example Watts' majority rule with a fair coin at ρ = ½.

Every backend reproduces an independent numpy reference **bit for bit**, including stochastic runs: randomness is counter-based (Philox4x32-10), so a given seed gives identical trajectories on every device.

## Install

```bash
conda env create -f environment.yml && conda activate fast-llna
pip install -e ".[metal,dev]"   # Apple Silicon
pip install -e ".[cuda,dev]"    # NVIDIA (cupy-cuda12x)
```

Example notebook: [notebooks/consensus_r9.ipynb](notebooks/consensus_r9.ipynb) shows how fast the 512 self-symmetric r = 9 rules and Watts' majority rule reach consensus on small-world networks. It needs `pip install -e ".[notebook]"`.

The CPU backend compiles its kernel at first use with the system C compiler (`cc`, or `$CC`) and caches it in `~/.cache/fast_llna` (`$XDG_CACHE_HOME/fast_llna`, or `$FAST_LLNA_CACHE`).

## Quick start

```python
import fast_llna as fl

g = fl.moore_torus(30, 30)                       # or fl.Graph(A) for any 0/1 adjacency (scipy.sparse / numpy)
rules = fl.life_like(5, *fl.nonequivalent(5).T)  # all 528 non-equivalent r=5 rules
init = fl.random_states(g.n, 60, density=0.5, seed=0)
rules, init = fl.product(rules, init)            # 528 x 60 = 31,680 replicas, one rule each
traj = fl.simulate(g, rules, init, steps=100)    # backend="auto": cuda > metal > cpu
rho = traj.density()                             # [T+1, R]

x, pairs = fl.defect_twins(fl.random_states(g.n, 128, seed=1), flips=1, seed=2)
traj = fl.simulate(g, fl.majority(), x, 100, seed=3, noise_period=128)  # twins share their coin flips
delta = traj.hamming(pairs)                      # [T+1, 128]
```

**Specifying rules**
- Partitions:
  - `fl.symmetric(r)`: the complementation-symmetric partition. Even r uses R⁺ for dead nodes and R⁻ for living nodes; pass `even="-+"` for the swapped convention.
  - `fl.uniform(r)`: the legacy left-closed partition.
  - `fl.MAJORITY`.
- Helpers:
  - `fl.equivalent(r, β, σ)` returns the complement-equivalent rule.
  - Custom stochastic rules are `fl.Rules(partition, p)`, where `p[i, s, j]` is the probability of being alive next.

**Options**
- `record`: `n` records every n-th step (`steps` must be a multiple of n); `"final"` records only the last state.
- `clamp`: `(mask, value)` pins nodes, per replica or for all replicas.
- `t0`: continues a run.
- `noise_period`: replicas r and r + p share their random draws (defect twins under stochastic rules); p divides 32 or is a multiple of 32.
- Directed graphs: row i of the adjacency matrix lists the nodes that node i reads.
- `backend`: `"auto"` (cuda > metal > cpu) or an explicit name; `threads` sets the CPU worker count.

**Environment variables**
- `FAST_LLNA_BACKEND`: replaces `backend="auto"`.
- `FAST_LLNA_CACHE`: directory for the compiled CPU kernels.
- `FAST_LLNA_CPU_MODE`: `words` or `nodes`, forcing the CPU threading mode.
- `FAST_LLNA_DUMP`: path to which the CUDA host writes its generated source.

**Output**

Each recorded configuration is a flat little-endian bit array `traj.states.bits[t]`, in which replica r of node i is bit `i·L + r`. L is the number of bits per node: the next power of two for R ≤ 8 (1, 2, 4 or 8), and whole bytes, 8·⌈R/8⌉, beyond. One replica of N = 1e7 nodes therefore takes 1.25 MB per frame. `traj.states.to_bool()` unpacks to `[T, R, N]`.

## Performance (Apple M4, 10 CPU cores, 10-core GPU)

| case | old `llna` (torch, CPU) | fast-llna CPU | fast-llna Metal |
|---|---|---|---|
| 528 rules × 60 inits, N=900, T=100, all steps recorded | 35 s | 0.17 s | 0.09 s |
| same, final state only | – | 0.08 s | 0.05 s |
| ER N=1e5, R=1024, T=100 | – | 0.36 s | 0.21 s |
| ER N=1e7, R=1, T=1000, every 10th step recorded (126 MB) | – | 24.9 s | 8.7 s |
| N=1e7, R=1, per step (setup excluded, `bench.py --scenario per_step`): ER / power-law (γ=2.5) / Moore | – | 24 / 24 / 14 ms | 8.6 / 8.4 / 3.4 ms |

**Throughput** is 2–5·10¹⁰ node-updates/s once there are ≥ 32 replicas per node.

**How it gets there**
- Replicas are packed 32 per machine word.
- Neighbours are counted with bit-sliced adders.
- Rules are applied through exact integer density thresholds per degree, with no floats.
- Every replica can carry a different rule at no extra cost.
- With R ≤ 32 a node holds only L = pow2ceil(R) bits, so huge graphs with few replicas run from cache.
- On GPUs a SIMD group counts each hub's edges together, so a hub does not stall one thread.

**Huge single networks with few replicas** now run from cache: at N=1e7 and R=1, the state is a 1.25 MB node bitset. On Metal a step then costs about as much as streaming the graph's index arrays once (≈3 ms at the M4's bandwidth for the lattice). Each call also pays a one-off setup of 50–120 ms for tables and device copies. The v1 layout needed 60 ms (ER) and 295 ms (power-law) per step on Metal. The NVIDIA numbers will come from `benchmarks/workstation_job.sh`.

## Layout

| path | purpose |
|---|---|
| `src/fast_llna/rules.py` | Partitions (exact integer cell index), rules, equivalence, enumeration |
| `src/fast_llna/graph.py` | CSR in-neighbour graphs, ring and Moore torus, disjoint union |
| `src/fast_llna/states.py` | Packed states, initial-condition helpers, `Trajectory` reducers |
| `src/fast_llna/reference.py` | Slow numpy reference: the definition the other backends must match |
| `src/fast_llna/kernels/llna.h` | The single kernel source shared by all three hosts (C / CUDA / Metal subset) |
| `src/fast_llna/kernels/{cpu,metal,cuda}.py` | Thin hosts: compilation, launch and recording |
| `benchmarks/` | `bench.py` scenarios, graph generators, `workstation_job.sh` for the NVIDIA machine |
| `docs/design.md` | Design rationale and plan |

## Tests

```bash
pytest -q
```

The test suite checks:
- every partition against the thesis interval definitions in exact rational arithmetic;
- a Game of Life glider and ECA rule 150;
- complementation symmetry, for odd and even r, with the legacy partition as a negative control;
- clamping, continuation and recording;
- bit-identity of every available backend against the reference.
