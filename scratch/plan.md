# MTG-507: why L4 bf16 evaluation slows trained w160l8 and speeds up w384l8

Stacked on MTG-497 (`fb1580b472`); that Task's plan is in git history at this
path. This plan replaces it for this branch.

## Outcome

`experiments/reports/eval_precision_profile.md` says where the bf16 forward
time goes on an L4 for trained `w160l8` and seeded `w384l8`, names the cause
of the `w160l8` slowdown, and ends in one of two things: a loader fix with
before/after arena arms, or a stated width/batch threshold above which bf16
wins on an L4. `docs/evaluation.md#etude-agents` carries the resulting advice.
`float32` stays the evaluation default either way (nothing picks a precision
from width or device, `manabot/backends/torch/learner.py:149`).

## What the earlier numbers do and do not show

From `experiments/reports/eval_throughput.md` and the receipts in
`../manabot.mtg-497-support-gpu-batched/.runs/eval-throughput/l4-966e8ff296/`
(one arm each, one worker):

| Model | Calls | Rows/call | float32 ms/call | bf16 ms/call |
| --- | ---: | ---: | ---: | ---: |
| Trained `w160l8` | 603 | 18.6 | 39.7 | 67.2 |
| Seeded `w384l8` | 677 | 9.9 | 43.6 | 18.6 |

- **The arms differ in width, weights, batches and saved readings.** Their
  model settings differ only in `hidden_dim`. The trained checkpoint saves
  narrower column prefixes and two smaller relation categories than a fresh
  seeded model. The trained policy wins 113 of 128 over long games with 18.6
  rows per forward. The seeded one loses quickly with 9.9.
- **A fixed per-call cast cost cannot be the whole cause.** bf16 `w384l8`
  takes 18.6 ms per call in total, less than the 27.5 ms bf16 adds per call at
  `w160l8`. Whatever costs 27.5 ms is absent or much smaller in the seeded
  games.
- **Table attention shapes do not vary.** The loader calls
  `load_checkpoint(str(path))` (`manabot/eval/agent_loader.py:113`):
  `bucketed_shapes` and `live_tokens` are off, so object attention runs over
  the full padded table every call. What varies per call is the batch and the
  CSR inputs to the packed GRUs (facts, records, semantic cards), which run
  float32 with autocast disabled (`manabot/backends/torch/packed_gru.py:55`).
- **Autocast adds about 1,300 dispatched ops per forward at either width**
  (measured locally with `torch.profiler` on 66 captured arena batches:
  3,429 -> 4,754 ops per call at `w160l8`, 3,205 -> 4,492 at `w384l8`;
  `aten::copy_` 100 -> 305 per call, mostly float32 weights and biases cast
  again in every `linear`). This is host work, and the L4 host was slow: its
  CPU forward cost 121 ms per row against 31 ms on the M4 Max.
- **bf16 arms do not play the same games.** The trained bf16 arm matched 117
  of 128 traces and served 11,252 rows against 11,209, so the two precisions
  were timed on different batches.
- **MPS did not settle it.** Three local 8-deal arena pairs on the M4 Max gave
  bf16 forwards of 23.7, 23.8 and 11.2 s against float32 11.7, 14.9 and
  11.4 s, with the machine at load 30-40 from other work. Replaying captured
  batches outside the arena showed no bf16 penalty (23 ms per call either
  way). The L4 penalty is not one noisy draw: MTG-317 reproduced it (below).

## Leading hypothesis (concept review, 2026-11-27)

`w160l8` has `head_dim = 160 / 8 = 20`; `w256l8` and `w384l8` have 32 and 48.
On CUDA, float32 memory-efficient SDPA accepts `head_dim % 4 == 0`, but bf16
Flash and memory-efficient SDPA need `head_dim % 8 == 0`, so bf16 `w160l8`
falls back to math attention and materializes `[B, 8, N, N]` scores in every
layer (`nn.MultiheadAttention`, `manabot/backends/torch/policy.py:1387,1455`).
`experiments/reports/cuda_precision.md` (MTG-317, same L4 and Torch
2.15.0+cu132) already shows it: `w160l8` padded bf16 large calls 464 -> 719 ms
and learner peak 5.3 -> 10.7 GB, repeats within 2%, while `w256l8`/`w384l8`
sped up 1.95-2.7x and halved memory. The slowdown is therefore already
reproduced; the L4 run's job is to name the mechanism, and the 160/256/384
8-head grid alone cannot separate width from `head_dim % 8`.

## Design

One standalone experiment CLI, `experiments/eval_precision_profile.py`, in the
shape of `experiments/eval_throughput.py` (exploratory: no prediction, the
report ends it). Three subcommands:

1. `capture` plays the 32 held-out deals on the vector-64 runner under
   float32 and writes, per model, (a) every multi-row host `Prepared` batch in
   arena order, for exact same-checkpoint replay, and (b) the same decisions
   as single-row `Prepared` inputs, for the grid. Grid calls of 1, 4, 16 and
   64 rows are built with the existing `batching.concatenate`
   (`manabot/policy/batching.py:24`); there is no split and none is written.
   Batch sets: trained `w160l8`, seeded `w160l8` and seeded `w384l8`.
2. `replay` loads one checkpoint under one variant and runs one batch set
   through it, in order, with `torch.cuda.synchronize` between transfer,
   forward and readback. It writes one row per call (rows, packed GRU rows,
   first-touch or repeat, stage seconds), `torch.cuda.max_memory_allocated()`,
   the max and exact-equality logit difference against shipped bf16, then a
   `torch.profiler` pass. Buckets come from `record_function` scopes in the
   harness (transfer, attention, packed GRU, readback) plus op classes
   (autocast casts `aten::to`/`_to_copy`/`copy_`, linear/MLP, norm,
   gather/index, unattributed host), and the attention bucket lists the
   dispatched SDPA kernels (flash, mem-efficient, or math `bmm`+`_softmax`).
   Feeding both precisions the same float32-captured batches removes the
   diverging-trajectory confound.
3. `render_report` reads the receipts and writes the tables.

Cross-width replay: `admission.binding` dumps the whole `ModelConfig`
(`manabot/policy/admission.py:110`), so `forward_distribution`
(`policy.py:770`) rejects a `w160l8` batch on any other width although
`Prepared` holds no width-dependent arrays. Chosen for review: the harness
asserts equal `space.config`, readings and model settings apart from width
and head count, then re-tags `Prepared.binding` with `dataclasses.replace`.
The grid instantiates seeded controls from the trained checkpoint's settings
and readings with `random_checkpoint(..., template=trained)`. Use the
fresh-reading seeded `w384l8` checkpoint only on its own capture and in its
arena repeats.

Variants:

- `float32`, `tf32`, `bf16` as the loader ships them.
- `bf16-linear-once`: after `cache_programs()`, cast only `nn.Linear` and
  `nn.MultiheadAttention` projection weights and biases to bfloat16; leave
  `LayerNorm`, `Embedding`, `nn.GRU` and bare `nn.Parameter`s
  (`perspective`, `value_token`) float32, which autocast also keeps float32.
  Forward still under autocast. Candidate production fix if bit-identical.
- `bf16-cached-autocast` (diagnostic only): one `torch.autocast` context held
  open across replayed calls, so PyTorch's own weight-cast cache survives;
  the upper bound of removing per-call casts. Not a production shape: a
  thread-local autocast context would leak into other Torch work.

Models (seeded unless trained; `random_checkpoint` gains a head count):

| Model | head_dim | Separates |
| --- | ---: | --- |
| trained `w160l8` | 20 | the arena case |
| seeded `w160l8` | 20 | trained weights vs width |
| seeded `w160` 5 heads | 32 | alignment at narrow width |
| seeded `w256l8` | 32 | threshold point |
| seeded `w288l8` | 36 | misalignment at wide width |
| seeded `w384l8` | 48 | the arena case |

If `head_dim % 8` is the cause, `w160`/5-heads speeds up under bf16 and
`w288l8` slows down. Trained vs seeded `w160l8` on the trained and seeded
batch sets separates board density (packed GRU rows) from kernel dispatch.

Arena replicates come after the replay pass and are sized by it: alternating
float32/bf16 repeats of the existing `eval_throughput` vector-64 arm for
trained `w160l8` and seeded `w384l8` (up to three each, about 2.2 minutes per
arm), to state the spread and to verify any fix. `ForwardTiming` keeps
`(rows, seconds)` per call in memory and writes `forwards.jsonl` once after
the run, so no disk write enters a timed forward; `summary.json` is
unchanged.

Loader split, chosen for review: `agent_loader.py` gains
`load_torch_policy(path, precision, device)` (validated, eval-mode, programs
cached) and one per-call context helper (`apply_matmul_precision`,
`torch.inference_mode`, `forward_precision`) that both `torch_logits` and
`replay` use, with Torch and `ActorCritic` types under `TYPE_CHECKING` so
`manabot.eval.agents` stays Torch-free
(`tests/test_package_boundaries.py:260`). Passed over: leaving the loader
untouched until pass 2 (replay would duplicate the per-call contract).

Chosen for review: profile by replaying captured batches rather than
instrumenting the live arena. Passed over: `torch.profiler` around the live
arena (profiler overhead distorts the host stages that dominate there) and a
synthetic fixed-shape microbenchmark.

## Rental

One RunPod L4 through Etude under `--max-usd 5`. Use the replay/profile pass
first, then only the arena arms it calls for. Receipts go to
`.runs/eval-precision-profile/l4-<rev>/`. Record the launcher's cost once and
confirm the pod is gone.

2026-11-27 availability check. `l4-etude` quoted USD 0.69/hour against its
USD 0.60 ceiling. Existing `mtg324-l4-step` quoted the same price under its
USD 0.70 ceiling, with zero of one in use. The replay and arena used that
profile after rechecking availability. The report records the launcher cost
and confirmed deletion. Inject RunPod from Doppler `etude/prd` and R2 from
`manabot/prd` through nested `doppler run` for Etude launcher commands.

## Acceptance check

```bash
uv run --no-sync python -m experiments.eval_precision_profile replay \
  --checkpoint .runs/eval-throughput/policy/00002000-raw.safetensors \
  --batches .runs/eval-precision-profile/trained-w160.batches \
  --device cuda --variant bf16 --output-dir .runs/eval-precision-profile/l4-<rev>/w160-trained-bf16
```

prints per-call stage means, peak CUDA memory, the SDPA kernels dispatched,
and a bucket table whose buckets sum to the synced forward time within 5%;
the report's `w160l8` table accounts for the bf16-minus-float32 difference by
bucket.

A CPU smoke test (`tests/eval/test_eval_precision_profile.py`, one deal,
float32) captures, replays and checks that replayed logits equal the logits
the arena received, that a `concatenate`d grid call equals its single-row
parts, and that the buckets cover the forward.

## Passes

1. **Harness.** `f3863261e` added the capture, replay, report and arena receipts, with CPU and MPS smoke checks.
2. **L4 diagnosis.** `75ae6a1d4` corrected CUDA trace admission. `e86b798d1` records 84 replay profiles, the even-head alignment control and three arena repeats per arm.
3. **L4 guidance.** `e86b798d1` documents the head-width and batch-size crossovers after cast-once preserved the logits and left the narrow slowdown intact.

## Absent and error states

- `replay` refuses bf16 off CUDA/MPS through the existing
  `validate_precision`, and refuses a batch file whose checkpoint input
  binding differs from the model's.
- A profiler pass with no CUDA activity (CPU run) reports host buckets only
  and marks kernel time absent, not zero.

## Exclusions

Training-side bf16, MPS and MLX precision, host arena and replay cost
(MTG-497's deferred F1), `torch.compile`, bucketed or live-token evaluation
forwards, and any change to the default precision or to compute class ids.

## What goes

Nothing existing is removed. The local probe scripts under `.runs/`
(`probe_mps.sh`, `profile_forward.py`, `cold_warm.py`) are untracked
scaffolding the experiment file replaces. `scratch/qa.html` records MTG-497's
checks. `scratch/tour.yaml` gives the reading order for MTG-507.

## Done when

- The report shows, for trained `w160l8` and seeded `w384l8` on an L4, the
  float32 and bf16 forward time split by bucket on identical batches, and the
  spread of three arena repeats per arm.
- It names the cause of the `w160l8` result (or shows the slowdown does not
  reproduce), separating width from trained weights and batch composition.
- Either the loader fix is in with before/after arena arms and unchanged
  logits, or the docs give the `head_dim`, width and rows-per-call conditions
  under which bf16 is faster on an L4.

## Concept review (2026-11-27) dispositions

- `head_dim % 8` SDPA fallback missing from the grid: adopted; models table,
  SDPA kernel and peak-memory recording added.
- Cross-width binding rejection and no `split`: adopted; single-row capture
  plus `concatenate`, harness-only re-tag (for review).
- Cast-once rounding LayerNorm/Embedding: replaced by `bf16-linear-once`
  plus diagnostic `bf16-cached-autocast` (review Q2a).
- Rental order: replay first, arena arms sized by it (Q3a); `forwards.jsonl`
  written once after the run.
- Loader split vs Torch-free import: `load_torch_policy` plus shared per-call
  helper under `TYPE_CHECKING` (Q4a, for review).

## Deferred findings

- If the L4 profile confirms the `head_dim % 8` fallback, file a separate
  Task to revisit frontier width/head guidance and `BF16_MIN_HIDDEN_DIM = 256`
  (`manabot/infra/config.py:347`; MLX uses it too), since training (MTG-317)
  shows the same penalty. Padding `head_dim` inside `GameObjectAttention`
  changes bf16 rounding and needs a new compute class id, so it is not
  MTG-507's.
