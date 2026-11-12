# Where an L4 learner step and a collection forward spend their time — MTG-301

Exploratory. The question was whether the host synchronizations and data-dependent
shapes found on Apple MPS also cost CUDA training. It sizes removing validation
syncs (MTG-299), catalog encoding (MTG-300), the earlier fixed-shape "option 3"
lookups and MTG-302's bucketed shapes. It is also the measured basis for an
accelerator-first redesign of the policy for larger models. One real w160l8
update was profiled on an L4, and its retained inputs were replayed per arm at
w160l8 and w384l8 (method in the appendix).

**Answer.** On an L4 the w160l8 learner step and large collection forwards are
GPU-bound (87–96% busy). Most of that time is fp32 attention blocks running at
full padded token capacity. Removing every validation sync could save at most
about 18 s of a ~180 s GPU-phase update. Catalog encoding costs at most about
5 s. Option 3 is slower on CUDA (1.4–1.9×). Bucketed shapes are 2.4–2.6× faster
in both phases, and the gain holds at w384l8. Padded work, not synchronization,
is the CUDA cost.

## Results

### (a) Synchronizations

Each window's wall time, GPU busy time (the union of kernel, memcpy and memset
intervals) and host time blocked in CUDA waits were read. With no waits the
host and GPU overlap fully, so the window cannot fall below max(GPU busy,
wall − waits). The difference from the wall is the upper bound on what removing
synchronizations alone could save (the *sync-removal bound* in the table under
(b)).

The replay counts are exact per call:
- **Collection forward, 118 syncs.** By file: `semantic_cards.py` 36,
  `facts.py` 27, `shared_effects.py` 16, `actor_critic.py` 9,
  `policy_distribution.py` 6, `name_candidates.py` 5, `continuation.py` 5,
  `records.py` 3, plus 5 in the selection's own `.cpu()` reads.
- **Learner step, 208 syncs.** By file: `semantic_cards.py` 65, `facts.py` 32,
  `health.py` 25, `policy_distribution.py` 20, `shared_effects.py` 15,
  `actor_critic.py` 14 and diagnostic reads 20.

The heaviest single sites are id-admission checks (`torch.equal(ids,
ids.floor())`, `(ids < 0).any()`, `(ids >= size).any()`), `require`'s
`bool(condition.all())`, fact-kind `torch.isin`, and nonzero-driven masked
lookups in facts and shared effects. The counts do not change with model size.

Upper bound for removing all of them, per w160l8 update: about 10.5 s of
learning (13.4% of 78.6 s) and 7.6 s of inference (5–24% by call size), about
18 s in total. At $0.59/hour that is at most about $0.30 per 100-update run.

### (b) GPU idle and launch-bound work

**Large calls and learner steps keep the GPU busy; small collection calls are
host-bound.**

| Window (w160l8 baseline) | Wall | GPU busy | Host waits | Sync-removal bound | Kernels | Syncs |
|---|---:|---:|---:|---:|---:|---:|
| Learner step, in situ | 345 ms | 290 ms (84%) | 256 ms | 57 ms | 4,885 | 195 |
| Learner step, replay | 400 ms | 349 ms | — | 51 ms (13%) | 4,935 | 208 |
| Collection call, ~200 rows, in situ | 319 ms | 296 ms (93%) | 285 ms | 26 ms | 1,195 | 118 |
| Collection call, 2–8 rows, replay | 22–27 ms | 5 ms | — | 1.4 ms | ~830 | 118 |

A small collection call launches about 1,000 kernels for only ~5 ms of GPU work
in a 22–27 ms call. Syncs bound only 1.4 ms of that; the rest is per-op host
dispatch. Those calls are most of the calls in an update:

| Rows per call | Calls | Seconds | Median per call |
|---|---:|---:|---:|
| 1 | 1,050 | 17.4 | 15.9 ms |
| 2–8 | 335 | 6.3 | 17.9 ms |
| 9–63 | 100 | 5.7 | 51.5 ms |
| 64–127 | 50 | 52.5 | 167.5 ms |
| 128–256 | 182 | 96.2 | 403.3 ms |

(In situ; seconds include preparation and recording.) The complete-game tail
produces 1,385 calls of eight rows or fewer (389,609 idle stream steps).

Median kernel duration is 2.4–2.6 µs, and 86% of learner kernels run under
10 µs, but in the busy windows a few large kernels dominate the time (see (d)).

The whole in-situ update (baseline, w160l8) also has a host-side phase larger
than any GPU phase:

| | option-3 job | bucketed job |
|---|---:|---:|
| Collection | 188.4 s | 291.2 s |
| … policy inference and action selection | 97.7 s | 99.7 s |
| … rollout recording | 76.3 s | 171.6 s |
| … environment step and encoding | 7.2 s | 6.9 s |
| Learning (196 optimizer steps) | 78.6 s | 79.0 s |
| Learner step, median | 0.387 s | 0.392 s |

The profiling hooks were active throughout, and this is update 1. MTG-267's
calibration read 194.5 s/update; its later updates read 86 s of collection with
71 s of inference.

*Rollout recording* took 76–172 s and varied 2.3× between two runs on one
machine. It is not explained by this profile. Whether it recurs past update 1
is unmeasured. In the slower bucketed job, the 16 profiled inference calls
(122–137) also carried 45–1,330 ms of host time that no recorded torch op
covers. The same calls in the option-3 job carried 17–22 ms
(`insitu/collection-trace.json.gz`, inside `mtg301/sample_learning_rows`). So
that job's host was slowed outside the GPU in both collection phases. The
window table above uses the option-3 job, and replays ran on retained inputs,
which this does not affect.

### (c) Catalog encoding

In the learner forward, `SemanticCardEncoder.definitions` costs 1.7 ms of GPU
and 9.6 ms of host span per step. Full-table encoding alone (361 definitions)
takes 8.5–9.2 ms forward and 24–26 ms forward plus backward at w160l8, and
12–14 ms / 34–40 ms at w384l8. That is at most about 5 s per update (≤ 6% of a
learner step), and under 0.5% of collection GPU time.

MTG-300 established that the catalog is encoded exactly once per weights
version, so this is the encoder's own cost: a GRU over referenced card
programs. It is not removable duplication.

### (d) Kernel compute and the candidate arms

**The busy time is fp32 attention at padded capacity.**

- **Collection forward at ~200 rows:** fp32 memory-efficient SDPA (37%),
  fp32 SGEMMs (27%), the attention-block FFN ReLU (11%) and copies/adds (17%).
- **Learner step:** SDPA backward (25%), `index_put_` accumulate (the backward
  of advanced-index gathers, 12.5%), SDPA forward (9%) and SGEMMs (~15%).

Attention blocks are 80% of replay inference wall. In the learner they are 20%
forward, plus most of the 60% that is backward on the autograd thread.

Replay pass totals (48 collection inputs; 20 learner minibatches with 29–103
rows):

| Arm | Size | Inference pass | Learner pass | Learner step | Kernels / learner step | Learner peak | Inference peak |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | w160l8 | 9.05 s | 7.88 s | 400 ms | 4,935 | 4.95 GB | 3.79 GB |
| option 3 | w160l8 | 13.42 s (1.48×) | 14.74 s (1.87×) | 875 ms | 25,664 | 5.41 GB | 4.40 GB |
| baseline | w384l8 | 19.81 s | 18.51 s | 927 ms | 5,018 | 12.0 GB | 4.43 GB |
| option 3 | w384l8 | 25.19 s (1.27×) | 32.66 s (1.76×) | 1,730 ms | 26,154 | 13.1 GB | 10.5 GB |
| baseline | w160l8 | 8.78 s | 7.96 s | 407 ms | 4,726 | 4.94 GB | 3.78 GB |
| bucketed | w160l8 | 3.72 s (0.42×) | 3.27 s (0.41×) | 185 ms | 7,090 | 2.03 GB | 0.83 GB |
| baseline | w384l8 | 19.78 s | 18.33 s | 915 ms | 4,814 | 11.98 GB | 4.42 GB |
| bucketed | w384l8 | 7.64 s (0.39×) | 7.70 s (0.42×) | 402 ms | 7,580 | 5.02 GB | 1.66 GB |

The first four rows come from the option-3 job and the last four from the
bucketed job. Ratios are to the same job's baseline: same machine, same retained
inputs, interleaved order and a baseline repeat at the end, which read within
2–3% of the first readings. The two jobs' baselines differ in torch version and
source, and they agree within 3% (inference pass 9.05 / 8.78 s, learner pass
7.88 / 7.96 s).

**Option 3 is not faster on CUDA.** It removes 44 inference and 87 learner
syncs, but launches 5× more kernels and processes facts and effect records at
full 512 / 4,096 capacity. Facts alone take 17% of its inference wall. It is
slower at every call size of nine rows or more (1.4–2.7×), and 0.79× only for
calls of eight rows or fewer.

**Bucketed shapes are 2.4–2.6× faster** on large collection calls and learner
steps at both sizes, and halve or better the memory. They cost 1.38× on calls
of eight rows or fewer (1,098 kernels against 989). Syncs barely change
(118 → 117; 208 → 192), and the sync-removal bound rises to 61 ms per learner
step as the GPU work shrinks.

Applying each band's replay ratio to the in-situ call mix projects, per w160l8
update:

| Arm | Inference | Learning |
|---|---:|---:|
| Bucketed | −38 s | −46 s |
| Option 3 | +50 s | +68 s |

This is a projection, not a whole-update measurement. Recording time was
subtracted per row and preparation per call. Against MTG-267's 194.5 s
calibration update, about 85 s less is roughly $1.40 per 100-update L4 run.

## Scale (w160l8 → w384l8, 5.5× parameters)

| Quantity | w160l8 → w384l8 |
|---|---|
| Large-batch collection call | 2.3× (502 → 1,150 ms) |
| Learner step | 2.3× (400 → 927 ms) |
| GPU busy | 88% → 95% (inference), 87% → 94% (learner) |
| Calls of 2–8 rows | flat (22–27 ms), host-bound |
| Sync count | unchanged |
| Sync-removal bound | flat (~50–65 ms/step), share 13% → 5–7% |
| Catalog fwd+bwd | 25 → 34–40 ms |
| Learner peak memory | 4.9 → 12.0 GB (bucketed 2.0 → 5.0 GB) |

At w384l8 the bucketed learner step (402 ms) matches the w160l8 baseline step
(400 ms).

## Decision

- **MTG-299 (validation syncs):** at most about 18 s per ~180 s GPU-phase update
  on CUDA, falling in share as models grow. Worth doing only as an enabler for
  CUDA graphs or compilation, not for its own CUDA saving.
- **MTG-300 (catalog):** closed negative. It is ≤ 5 s per update of real encoder
  work.
- **Option 3:** do not pursue. Fixed shapes at maximum capacity cost more GPU
  work than they save in syncs.
- **MTG-302 (bucketed shapes):** worth keeping. Task MTG-312 moves the switch
  into the typed regime: an Execution field next to `device`, serialized only
  when set so frozen regime hashes do not change, exposed on the CLI and
  recorded in run receipts. It then deletes `MANABOT_BUCKETED_SHAPES`. Making it
  the default needs separate learning evidence and should weigh the 1.38× cost on
  single-row calls.

## Implications for an accelerator-first redesign

Ranked by the measured share of CUDA time each direction addresses.

1. **Compute only live tokens.** Attention blocks (SDPA plus FFN) over the full
   ~450-slot object capacity are 80–87% of inference GPU time and dominate
   backward. Bucketing the variable axes alone gave 2.4–2.6×, and its share
   grows with width. Variable-length packed attention, or object capacity
   bucketed per batch as a first-class layout, is the largest lever. Fixed shapes
   help only when they are small.
2. **Precision.** Every attention kernel and GEMM ran fp32
   (`fmha_cutlassF_f32`, `ampere_sgemm`). The L4's tensor cores are unused.
   TF32/BF16 were not measured here, and each needs learning evidence; the earlier
   size ladder read w160l8 on L4 at 75.0 s (fp32) against 68.3 s (TF32).
3. **Host-bound tail of collection.** 1,385 of 1,717 inference calls carry eight
   rows or fewer. Each launches about 1,000 kernels for ~5 ms of GPU work in a
   22–27 ms call, about 18 s per update that grows with no model dimension.
   - Levers: fewer, larger calls (the complete-game tail idles 389,609 stream
     steps), or capturing the forward as a CUDA graph.
   - Graph capture needs static shapes and no host reads, which is where buckets
     and MTG-299 combine.
4. **Gather backward.** `index_put_` accumulation from advanced-index gathers is
   12.5% of learner kernel time at w160l8. Index-select or embedding-style joins
   with a cheaper backward are a measured target.
5. **Host syncs.** 118 per forward and 208 per learner step, mostly id-admission
   and masked-lookup reads in semantic cards, facts and shared effects. They are
   worth ≤ 13% of a step now and less at larger sizes; their main cost is that
   they block graphs and compilation (direction 3).
6. **Catalog program encoding and packed GRUs.** ≤ 6% of a learner step and
   under 0.5% of collection GPU time. Not a redesign priority at these sizes.
7. **Host-side rollout recording.** 76–172 s of update 1, unexplained and
   variable. It is not GPU work, but it was the largest collection phase
   observed, and needs its own reading over later updates before any
   accelerator design assumes collection is inference-bound.

**Cost:** about $0.80 for both jobs on one L4 window ($0.143 posted when
written), under the USD 10 cap.

## Limits

- One update, one host and one GPU type.
- The replay inference inputs are the first 576 calls, so they over-weight large
  batches; whole-update effects are projections.
- In situ, the learner sync window covered backward and step only; the replay
  gives whole-step counts.
- Sync-debug mode is a prototype and may miss some synchronizing operations.
- Kernels with no runtime launch record (4–17% of learner time, larger at
  w384l8) are unattributed.
- No CUDA baseline-to-baseline numerics reading (see the gates in the appendix).
- No learning or strength evidence for any arm.

<details>
<summary>Appendix: method, runs and cost, numerical gates, evidence</summary>

### Method

1. **In situ.** The frozen MTG-267 w160l8 regime (`tla_frontier.frozen()`)
   plays one real update on CUDA from its own initialization, unchanged except
   for `updates=1`. That regime is input v9, 512 facts, catalog vocabulary,
   256 streams × 64 transitions with complete games, both seats, four
   producers, float32, seed 21501.
   - Hooks profile inference calls 122–137 and learner steps 32–39 with
     `torch.profiler` (CPU and CUDA).
   - `torch.cuda.set_sync_debug_mode("warn")` attributes synchronizing calls to
     their innermost repository line over inference calls 200–215 and the
     backward/step of learner steps 50–59.
   - The hooks retain 48 collection observations (every 12th call up to call
     576) and 20 learner minibatches (steps 70–89).
2. **Replay.** Those inputs run through collection's preparation and *Policy
   inference and action selection* block, and through `update_damped_iteration`'s
   minibatch body, including its index round trip, `OptimizerHealth` and
   diagnostic reads. Each arm runs at w160l8 (the in-situ initial weights) and at
   w384l8, a random initialization with 31.4M parameters, which fit the L4.
   Labels on every top-level module attribute kernels to their launching code.
   Each window's bounds are read as defined under (a).

### Runs and cost

| Job | Etude run | Source | Torch | Arms |
|---|---|---|---|---|
| option 3 | `d8ad784530d7` | `4a002968e` = main `92560a1e0` + module | 2.10.0+cu128 | baseline, option 3 |
| bucketed | `1999b57daf19` | `e55157ac5` (tag `mtg301-bucketed-run`) = MTG-302 branch `54cd65420` (newer main) + module | 2.15.0+cu132 | baseline (switch off), bucketed (`MANABOT_BUCKETED_SHAPES=1`) |

Both jobs ran in sequence on one hand-held L4 window: machine
`l4-etude-6fc061afd2e6`, RunPod pod `jvkaqth5y6wyt9`, 11:27–12:40 UTC on
2026-11-12. The host was an AMD EPYC 9254 with 48 schedulable CPUs.
MTG-302 later landed on main as `bf4e280f7` with `manabot/model` identical to
`54cd65420`, so the bucketed arm measures the code on main.

Cost: RunPod had posted $0.143 when this was written ($0.032 for the first
attempt's pod `26qog3f0upfmuy`, $0.112 for `jvkaqth5y6wyt9`). The window at the
quoted $0.59/hour bounds the total at about $0.80 once billing settles, under
the USD 10 cap. The rental was terminated at 12:40; no pod remains.

Dated lines, 2026-11-12:
- Run `6441cc948c6a` failed in its first minutes because the bundle has no
  `.git` for `source_commit`. Run `42e16409d083` failed the same way. Run
  `ee82615b12e2` failed on a stale `/tmp` work directory from the previous
  attempt. Each fix is a later commit on this branch; the question is unchanged.
- At 10:24 UTC the first `etude launch` also admitted an already-queued
  MTG-267 run (`tla-frontier-c5fba4-cpu-split-v1-53dd18c20036`, cpu32). It was
  left running and noted on MTG-267; its cost is not this experiment's.

### Numerical gates

At fixed weights (`numerics`, `compare`):
- **Option 3, CPU, before timing** (four real learner minibatches and eight
  collection observations): logits and values bit-identical; losses within
  1e-10; gradients within 4e-7 (cosine ≥ 0.9999999999987).
- **Option 3, CUDA:** gradients within 8.3e-5, cosine ≥ 0.99999990.
- **Bucketed, CUDA:** losses within 9e-8, logits within 5e-9, gradients within
  2.3e-5, cosine ≥ 0.9999999941.

The CUDA differences are atomic-accumulation order. They exceed the strict
CPU-tuned gradient tolerance in `compare`, which therefore reports `ok: false`
for both; no CUDA baseline-to-baseline numerics reading was taken to set a
tolerance.

### Evidence and reproduction

Everything is under `experiments/data/l4_step_profile/{option3,bucketed}/`
(manifest entries in `experiments/data.manifest.json`):
- `summary.json`, `job.json` and `gpu-samples.csv`;
- `insitu/` (receipt, sync records, chrome traces);
- `replay/<arm>-<size>/` (receipts, sync records, traces);
- `numerics/`, `logs/`;
- `inputs/` (initial weights, regime, 4 learner and 8 collection inputs).

Reproduce one job with

```sh
etude rent l4-etude --hours 1.5 --max-usd 1.0
etude queue experiments/l4_step_profile.py --rent l4-etude --cohort NAME \
  --revision COMMIT --candidate option3|bucketed --hours 0.6
```

and test locally with `ETUDE_OUTPUT_DIR=… uv run python -m
experiments.l4_step_profile job --smoke`.

</details>
