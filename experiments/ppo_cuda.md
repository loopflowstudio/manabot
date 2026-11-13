# PPO CUDA updates (MTG-338)

Exploratory throughput work: why does a PPO update spend most of its time in
backward on one NVIDIA L4, and can a behaviour-preserving change remove it?

**PPO updates are 2.17× faster.** On the Elemental w64l2 workload, alternating
the original and changed card-program lookup on one captured batch gives a
median **32.23 s** per update (32.09–32.41) against **69.97 s** (69.79–70.08),
both with 16 optimizer steps: **0.4964** against 0.2287 optimizer steps/s.
CPU results are bitwise identical; on CUDA the change reorders summation and
stays within the original method's own run-to-run drift.

## Cause

A profile of the original method (run `e9e5a6bd4649`) put 78.3% of first-step
GPU kernel time in 989 `indexing_backward_kernel` calls. Advanced indexing of
the 463 × 64 card-program table accumulates every gathered row's gradient
separately, and card and fact IDs repeat heavily within a batch:

| Program-table read | Incoming gradient shape | GPU kernel time |
| --- | --- | ---: |
| Fact sources | 6668 × 256 × 64 | 1.126 s |
| Continuation relation definitions | 6668 × 64 × 2 × 64 | 0.542 s |
| Visible cards | 6668 × 2 × 60 × 64 | 0.303 s |
| Permanents | 6668 × 2 × 32 × 64 | 0.181 s |
| Decklists | 6668 × 2 × 32 × 64 | 0.088 s |
| Known hand | 6668 × 2 × 8 × 64 | 0.046 s |

Total kernel time in that step was 2.96 s. Backward and gradient clipping were
46.0 s of a 59.6 s minibatch section; forward and loss 11.4 s; the optimizer
step 0.03 s.

## Change

Every read of the program table goes through `SemanticCardEncoder.lookup`
(`manabot/model/semantic_cards.py`): card and fact sources, continuation
relations, actions, visible cards, known hand, decklists, permanents
(`backends/torch/policy.py`) and recent-event references
(`model/recent_events.py`). When the table is on CUDA and requires gradients,
`lookup` uses `F.embedding`, whose backward groups repeated rows before
accumulating into the table. Row 0 still receives gradients (no padding index)
and no frequency scaling applies. CPU, MPS and inference keep advanced indexing.
`tests/model/test_semantic_cards.py` checks values and gradients against
indexing for repeated IDs, row zero, empty, scalar and noncontiguous IDs and a
non-leaf table, on CPU and, with `MANABOT_REQUIRE_CUDA=1`, on CUDA.

With the change, backward and clipping take 16.8 s and forward and loss 12.1 s
per update.

## Measurement

Etude run `0bf2b5e50d52`, revision `c9f823739`, one L4, Torch 2.15.0+cu132,
CUDA 13.2, TF32, one Torch CPU thread, deterministic cuDNN. One collection
(256 streams × 64 transitions, both seats, `elemental-uniform`, bucketed shapes,
live tokens; 53,341 learner rows, 26,671 retained) feeds every update. PPO uses
`Learning(reference="action_type_uniform")` at schedule progress 0.3, four
epochs and four minibatches. Both arms run in one process from the same native
build; the original arm patches `lookup` back to indexing at every site. Each
update restores the same initial weights, a fresh Adam and a copy of the batch.
One warmup per arm, then five alternating pairs:

| Pair | Original s/update | Changed s/update |
| ---: | ---: | ---: |
| 1 | 69.79 | 32.36 |
| 2 | 70.00 | 32.09 |
| 3 | 69.97 | 32.10 |
| 4 | 70.08 | 32.23 |
| 5 | 69.95 | 32.41 |

An earlier run (`34b6855d1d51`, revision `9268966e1`) changed only the fact
source, continuation relation and action reads: 36.88 s against 64.35 s
(1.74×) on another L4 allocation. The prediction committed before that run was
at least 20% less update time. Extrapolating the profile's per-step kernel time
over 16 steps predicted about 10 s more from the remaining reads; the measured
step is about 4.6 s (36.9 → 32.2 s, across allocations).

## Equivalence

- **CPU:** the MTG-325 harness compares 3,248 output, gradient and
  trained-weight tensors between the two methods; none differ. The CPU update
  traces are byte-identical.
- **CUDA:** two audit updates per arm retain raw first-step gradients,
  first/final weights and final probes. Across methods the largest gradient
  error is 27% of the bound `1e-6 + 1e-4 · max|reference gradient|` and the
  largest final-weight error 8% of 1e-4; between two runs of the original
  method they are 27% and 10%. Final values differ by up to 2.4e-4 across
  methods and 2.6e-4 between two original runs; first losses by up to 3e-8 in
  both. Exact first loss and final value error ≤1e-4 were frozen as bounds but
  sit below the original method's own CUDA nondeterminism, so they cannot
  separate the methods.
- 48 focused model tests (semantic cards, precision, bucketed shapes, live
  tokens, recent events) passed on the L4 with CUDA required.

Evidence: `experiments/data/ppo_cuda/paired-all/` (this run),
`experiments/data/ppo_cuda/paired/` (the two-site run) and
`experiments/data/ppo_cuda/{baseline,profile}/`.

## Limits

One batch, one seed, one L4 allocation per run; the spread covers repeated
updates, not hosts or seeds. It measures the update alone, not full training
throughput or strength per dollar; collection (about 28 s here) is unchanged.
CUDA equivalence is one update, not long-run training. Backward still takes
about twice the damped update's 7.2 s on the MTG-325 workload.

## Cost

About $1.14 at the launcher: $0.23 and $0.31 for the two paired runs, $0.04 for
an A6000 attempt (`bb2863fe318c`) that stopped when `test_active_training.py`
called `git rev-parse` in Etude's non-git source copy (that file is not in the
remote test list), and about $0.56 for the baseline, profile and two cancelled
attempts. Many launch attempts were refused for lack of L4 stock before any
machine started and cost nothing.

The launcher `experiments/ppo_cuda.py` runs its harness from the launch
revision's `scratch/` (`scratch/speed/`, `scratch/equivalence/`); launch it at
`c9f823739` to reproduce.
