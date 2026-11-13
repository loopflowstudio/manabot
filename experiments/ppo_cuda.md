# PPO CUDA updates (MTG-338)

Exploratory throughput work: why does a PPO update spend most of its time in
backward on one NVIDIA L4, and can a behaviour-preserving change remove it?

**PPO updates are 1.74× faster.** On the Elemental w64l2 workload, alternating
the kept and changed lookup on one captured batch gives a median **0.4338**
optimizer steps/s (0.3999–0.4611) against **0.2486** (0.2405–0.2608): 36.9 s
instead of 64.4 s per update, both with 16 optimizer steps. CPU results are
bitwise identical; on CUDA the change reorders summation and stays within the
kept method's own run-to-run drift.

## Cause

A profile of the kept method (run `e9e5a6bd4649`) put 78.3% of first-step GPU
kernel time in 989 `indexing_backward_kernel` calls. Advanced indexing of the
463 × 64 card-program table accumulates every gathered row's gradient
separately; the fact-source join (6668 × 256 × 64 gradient) alone took 1.13 s
and the continuation-relation join 0.54 s of 2.96 s. Backward and gradient
clipping were 46.0 s of a 59.6 s minibatch section; forward and loss 11.4 s;
the optimizer step 0.03 s.

## Change

`SemanticCardEncoder.embed_definitions` uses `F.embedding` when the table is on
CUDA and requires gradients. Embedding backward groups repeated IDs before
accumulating into the table. Row 0 still receives gradients (no padding index)
and no frequency scaling applies. CPU, MPS and inference keep advanced indexing.
`tests/model/test_semantic_cards.py` checks values and gradients against
indexing for repeated IDs, row zero, empty, scalar and noncontiguous IDs and a
non-leaf table, on CPU and, with `MANABOT_REQUIRE_CUDA=1`, on CUDA.

With the change, backward and clipping take 19.6 s and forward and loss 11.9 s
per update (candidate profile in the run's evidence).

## Measurement

Etude run `34b6855d1d51`, revision `9268966e1`, one L4, Torch 2.15.0+cu132,
CUDA 13.2, TF32, one Torch CPU thread, deterministic cuDNN. One collection
(256 streams × 64 transitions, both seats, `elemental-uniform`, bucketed shapes,
live tokens; 53,341 learner rows, 26,671 retained) feeds every update. PPO uses
`Learning(reference="action_type_uniform")` at schedule progress 0.3, four
epochs and four minibatches. Both methods compile in one process from the same
native build; each update restores the same initial weights, a fresh Adam and a
copy of the batch. One warmup per arm, then five alternating pairs:

| Pair | Kept s/update | Changed s/update |
| ---: | ---: | ---: |
| 1 | 64.35 | 36.88 |
| 2 | 66.52 | 40.01 |
| 3 | 62.29 | 38.69 |
| 4 | 65.58 | 36.80 |
| 5 | 61.35 | 34.70 |

The prediction committed before the run was at least 20% less update time
(≥0.330 steps/s); the result is 42.7% less.

## Equivalence

- **CPU:** the MTG-325 harness compares 3,248 output, gradient and
  trained-weight tensors between the two methods; none differ. The CPU update
  traces are byte-identical.
- **CUDA:** two audit updates per arm retain raw first-step gradients,
  first/final weights and final probes. Across methods the largest gradient
  error is 5.1% of the bound `1e-6 + 1e-4 · max|reference gradient|`, and the
  largest weight error 34% of 1e-4; between two runs of the kept method they are
  7.3% and 22%. The first loss differs by at most 3.0e-8 and final values by
  2.2e-4 across methods, against 3.0e-8 and 2.7e-4 between two kept runs.
  Exact first loss and final value error ≤1e-4 were frozen as bounds but sit
  below the kept method's own CUDA nondeterminism, so they cannot separate the
  methods.
- 39 focused model tests passed on the L4 with CUDA required.

Evidence: `experiments/data/ppo_cuda/paired/` (timings, differences, audit
tensors, tests, CPU comparison, candidate profile); the kept-method baseline
and profile are under `experiments/data/ppo_cuda/{baseline,profile}/`.

## Limits

One batch, one seed, one L4 allocation; the spread covers repeated updates, not
hosts or seeds. It measures the update alone, not full training throughput or
strength per dollar; collection (27.7 s here) is unchanged. CUDA equivalence is
one update, not long-run training. Backward still takes about twice the damped
update's 7.2 s on the MTG-325 workload, so further gains may remain.

## Cost

About $0.35 at the launcher: $0.31 for the paired run and $0.04 for an A6000
attempt (`bb2863fe318c`) that stopped when `test_active_training.py` called
`git rev-parse` in Etude's non-git source copy; that file is not in the remote
test list. Baseline and profile runs (`4a289dc30a82`, `e9e5a6bd4649` and two
cancelled attempts) cost about $0.56. Many earlier launch attempts were refused
for lack of L4 stock before any machine started and cost nothing.

The launcher `experiments/ppo_cuda.py` runs its harness from the launch
revision's `scratch/` (`scratch/speed/`, `scratch/equivalence/`); launch it at
`9268966e1` to reproduce.
