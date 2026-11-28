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

- **Width is not the only thing that differs.** The two checkpoints have the
  same model settings apart from `hidden_dim` (checked with `read_container`
  against `tla_frontier.model("w160l8")`). But one is trained and one is
  random: the trained policy wins 113 of 128 over long games with 18.6 rows
  per forward, the seeded one loses quickly with 9.9. Width, trained weights
  and batch composition are confounded in the Task's headline.
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
  way). The single L4 arm may likewise be one draw from a wide distribution,
  so the first thing the rental does is repeat it.

## Design

One standalone experiment CLI, `experiments/eval_precision_profile.py`, in the
shape of `experiments/eval_throughput.py` (exploratory: no prediction, the
report ends it). Three subcommands:

1. `capture` plays the 32 held-out deals on the vector-64 runner under
   float32 and writes every host `Prepared` batch the checkpoint was asked
   for, in arena order, to one file per model. Three batch sets: trained
   `w160l8`, seeded `w160l8` (`random_checkpoint(path, 160)` already exists)
   and seeded `w384l8`.
2. `replay` loads one checkpoint under one variant and runs one batch set
   through it, in order, with `torch.cuda.synchronize` between transfer,
   forward and readback. It writes one row per call (rows, packed GRU rows,
   first-touch or repeat, stage seconds), then a `torch.profiler` pass with
   CUDA activity folded into named buckets: transfer, autocast casts
   (`aten::to`/`_to_copy`/`copy_` inside the forward), packed GRU, attention,
   linear/MLP, norm, gather/index, readback, and unattributed host time.
   Feeding both precisions the same float32-captured batches removes the
   diverging-trajectory confound.
3. `render_report` reads the receipts and writes the tables.

Variants:

- `float32`, `tf32`, `bf16` as the loader ships them.
- `bf16-cast-once`: weights cast to bfloat16 at load, GRU modules left
  float32, forward still under autocast. This is the candidate fix for the
  per-call weight casts; the replay records whether its logits equal
  shipped bf16 bit for bit.

Crosses that separate the confounds:

- **Trained or width?** trained `w160l8` against seeded `w160l8`, each on the
  trained batch set and on the seeded-`w160l8` set.
- **Batches or model?** seeded `w384l8` on its own set and on the trained
  set. `Prepared.binding` is checked in `forward_distribution`
  (`manabot/backends/torch/policy.py:770`); if it refuses a batch prepared
  for another width, each model replays only sets prepared for it and the
  report says the cross is by game population, not identical bytes.
- **Threshold.** Seeded widths 160, 256, 384 on the trained set, with the
  set re-chunked to 1, 4, 16 and 64 rows per call, gives the width-by-batch
  grid where bf16 forward time crosses float32.

Arena replicates: three alternating float32/bf16 repeats of the existing
`eval_throughput` vector-64 arm for trained `w160l8` and seeded `w384l8`
(12 arms of about 2.2 minutes), so the report states the spread the original
single arms came from. `eval_throughput` keeps its summary; its
`ForwardTiming` additionally writes per-call rows and seconds to
`forwards.jsonl` so in-arena and replayed calls can be laid side by side.

`torch_logits` is split so the profiler loads the model exactly as the loader
does (`load_torch_policy(path, precision, device)` returns the eval-mode model
with programs cached; `torch_logits` wraps it). No second loading path.

Chosen for review: profile by replaying captured batches rather than
instrumenting the live arena. The arena view comes from the replicates and
`forwards.jsonl`. Passed over: `torch.profiler` around the live arena
(profiler overhead distorts the host stages that dominate there) and a
synthetic fixed-shape microbenchmark (the local replay shows steady-state
shapes hide the effect).

## Rental

One RunPod L4 through Etude, the path the `l4-966e8ff296` arms used, under
`--max-usd 5`: about 30 minutes of arena replicates and 20 of replays at
roughly USD0.45/h. No CPU baseline arms (they took 65 minutes last time and
answer nothing here). Receipts go to `.runs/eval-precision-profile/l4-<rev>/`;
the launcher's cost is recorded once in the report. Confirm the pod is gone.

## Acceptance check

```bash
uv run --no-sync python -m experiments.eval_precision_profile replay \
  --checkpoint .runs/eval-throughput/policy/00002000-raw.safetensors \
  --batches .runs/eval-precision-profile/trained-w160.batches \
  --device cuda --variant bf16 --output-dir .runs/eval-precision-profile/l4-<rev>/w160-trained-bf16
```

prints per-call stage means and a bucket table whose buckets sum to the
synced forward time within 5%, and the report's `w160l8` table accounts for
the bf16-minus-float32 difference by bucket.

A CPU smoke test (`tests/eval/test_eval_precision_profile.py`, one deal,
float32) captures, replays and checks that replayed logits equal the logits
the arena received and that the buckets cover the forward.

## Passes

1. **Harness and the L4 run.** Split the loader, add `forwards.jsonl`, write
   `capture`/`replay`/`render_report` and the smoke test, validate locally on
   CPU and MPS, then run replicates, crosses and the grid on the L4. This is
   the pass that can prove the premise wrong: if three repeats do not
   reproduce a `w160l8` bf16 slowdown, the report says so with the spread and
   the Task ends there with the threshold grid.
2. **Fix or threshold.** If `bf16-cast-once` (or another cause the buckets
   name) removes the penalty with identical logits, change `torch_logits`,
   rerun the two bf16 arena arms on the same rental session and report
   before/after. If the fix changes logits, it is a different compute class
   and is not taken here; the report documents the threshold instead. Update
   `docs/evaluation.md#etude-agents` and add one line to
   `experiments/reports/eval_throughput.md` pointing at the new report.

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
scaffolding the experiment file replaces. `scratch/qa.html` and
`scratch/tour.yaml` are MTG-497's and are not this Task's review material.

## Done when

- The report shows, for trained `w160l8` and seeded `w384l8` on an L4, the
  float32 and bf16 forward time split by bucket on identical batches, and the
  spread of three arena repeats per arm.
- It names the cause of the `w160l8` result (or shows the slowdown does not
  reproduce), separating width from trained weights and batch composition.
- Either the loader fix is in with before/after arena arms and unchanged
  logits, or the docs give the width and rows-per-call above which bf16 is
  faster on an L4.
