# CUDA evaluation precision profile

The [earlier L4 comparison](eval_throughput.md#runpod-l4) measured trained
`w160l8` forwards at 24.0 s in float32 and 40.5 s in bf16. Seeded
`w384l8` ran in 29.5 s and 12.6 s respectively. Each arm played the same
held-out deals. Its policy controlled the actions, so each arm could produce
different batches.

The profiler runs both precisions on the same captured inputs. The exact
arena comparisons keep the original batches and checkpoint. The head-width
grid uses independently prepared single rows from those same calls. Seeded
controls in the grid inherit the trained checkpoint's saved observation
readings. The trained checkpoint predates additional columns and two larger
relation categories, so fresh seeded checkpoints require a different input
binding. The seeded `w384l8` arena comparison keeps its own capture and
checkpoint.

## Local verification

A CPU smoke test replayed every captured call of a four-game deal with
identical logits, joined single rows through the production batcher, and
ran a widened checkpoint with the same input contract. An MPS capture of
one actual held-out TLA deal with trained `w160l8` retained 315 calls across
four games. A four-call replay on MPS produced exact bf16 logits for
`bf16-linear-once`. That smoke does not establish CUDA parity or speed.

## Identical batches on the L4

Replay ran at `75ae6a1d4` on one NVIDIA L4 with Torch 2.15.0+cu132 and
driver 595.112.07. The host had an AMD EPYC 9254 and Torch used one CPU
thread. Replay and the arena used Etude SDK revision `d876a7b1`.
Each float32 capture passed Command replay for all 128 games. The table
gives the mean of two uninstrumented repeats over the entire capture.
The peak is PyTorch's peak live CUDA allocation.

| Checkpoint | Precision | Calls | Rows/call | Transfer (ms) | Forward (ms) | Readback (ms) | Peak (MiB) | Attention dispatch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Trained `w160l8` | float32 | 603 | 18.59 | 1.24 | 39.13 | 0.17 | 621.7 | Native MHA |
| | bf16 | 603 | 18.59 | 1.33 | 66.32 | 0.23 | 712.9 | SDPA math |
| Seeded `w384l8` | float32 | 691 | 9.76 | 0.95 | 44.63 | 0.15 | 1,120.1 | Native MHA |
| | bf16 | 691 | 9.76 | 0.96 | 18.12 | 0.12 | 639.9 | Memory-efficient SDPA |

Both float32 models dispatched `aten::_native_multi_head_attention`.
With bf16 autocast, `w160l8` used `aten::_scaled_dot_product_attention_math`
and its trace contained FP32 GEMMs, separate softmax, scaling and masking
kernels. The 48-wide heads of `w384l8` used
`aten::_scaled_dot_product_efficient_attention` and the trace linked its
`fmha_cutlassF_bf16_aligned_64x64_rf_sm80` kernel. Every recorded attention
layer took its arm's listed path.

Four deterministically sampled calls per checkpoint supplied the bucket
estimates below. The estimates sum to the synchronized, uninstrumented
mean of those four calls, which were larger than the full-capture average.
The traces' CPU forward durations were 1.03–1.11× their uninstrumented
means, and the correlated buckets covered the complete forward scopes.

| Checkpoint and precision | Attention | Casts and copies | Linear and MLP | Normalization | Gather and index | Packed GRU | Other | Total (ms/call) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trained `w160l8` float32 | 36.07 | 0.79 | 15.56 | 1.46 | 0.61 | 0.61 | 4.35 | 59.45 |
| Trained `w160l8` bf16 | 72.47 | 14.09 | 5.70 | 1.40 | 0.74 | 0.77 | 5.61 | 100.77 |
| Trained `w160l8` bf16 minus float32 | **+36.40** | +13.30 | −9.87 | −0.07 | +0.13 | +0.16 | +1.25 | **+41.31** |
| Seeded `w384l8` float32 | 33.05 | 1.45 | 41.39 | 1.41 | 0.92 | 0.51 | 7.75 | 86.48 |
| Seeded `w384l8` bf16 | 7.66 | 4.29 | 9.51 | 1.25 | 0.82 | 0.39 | 6.47 | 30.40 |

On trained `w160l8`, casting projection weights once preserved all legal
bf16 logits exactly and moved the full-capture mean from 66.32 to 65.23 ms.
Keeping the autocast weight cache across calls also preserved the logits
and measured 65.76 ms. Both variants kept the math backend. Cast and copy
time includes tensor conversions inside attention that persist when the
weights are cached.

| Diagnostic variant | Trained `w160l8` (ms/call) | Seeded `w384l8` (ms/call) |
| --- | ---: | ---: |
| TF32 | 34.94 | 30.36 |
| bf16, cast projections once | 65.23 | 15.48 |
| bf16, keep autocast cache | 65.76 | 17.71 |

## Head width and batch size

Each grid cell timed two repeats of 12 calls built from consecutive single
rows after skipping the first 2,048 rows of its capture. The seeded
checkpoints used seed 1 and the trained checkpoint's saved input contract.
The entries are bf16 forward time divided by float32 forward time.

| Checkpoint | Rows from | Head width | bf16 backend | 1 row | 4 rows | 16 rows | 64 rows |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Trained `w160l8` | Trained | 20 | Math | 1.61× | 1.47× | 1.72× | 1.70× |
| Seeded `w160l8` | Trained | 20 | Math | 1.25× | 1.37× | 1.69× | 1.72× |
| Seeded `w160`, 5 heads | Trained | 32 | Memory-efficient | 1.16× | 1.16× | 0.64× | 0.44× |
| Seeded `w160`, 4 heads | Trained | 40 | Memory-efficient | 1.20× | 1.08× | 0.49× | 0.35× |
| Seeded `w256l8` | Trained | 32 | Memory-efficient | 1.15× | 0.69× | 0.36× | 0.34× |
| Seeded `w288l8` | Trained | 36 | Math | 1.54× | 1.25× | 1.28× | 1.21× |
| Seeded `w384l8` | Trained | 48 | Memory-efficient | 0.95× | 0.63× | 0.35× | 0.34× |
| Trained `w160l8` | Seeded `w160l8` | 20 | Math | 1.49× | 1.50× | 1.70× | 1.70× |
| Seeded `w160l8` | Seeded `w160l8` | 20 | Math | 1.50× | 1.42× | 1.68× | 1.71× |

At a fixed width of 160, moving from 20-wide to 32- or 40-wide heads
switched bf16 from math to memory-efficient SDPA. The four-head control kept
native float32 MHA. The odd five-head control used memory-efficient SDPA
in both precisions. The wider `w288l8` returned to math SDPA with its
36-wide heads. These switches follow whether the tested head widths are
multiples of eight. Both `w160l8` checkpoints slowed down at every batch
size on both captures.

For this padded workload, the first clear bf16 win among the tested batch
sizes was 4 rows per call for `w256l8` and `w384l8`, and 16 rows per call for
the aligned 160-wide controls. Use float32 for 20- and 36-wide heads on this
L4 configuration. Casting projections once left the trained slowdown
essentially intact, so the guidance applies to the existing loader.

## Vector arena repeats

Each arm played 128 games across the same 32 held-out deals in 64 slots.
The arms ran sequentially on the same L4 with precision order reversed in
the second repeat. All 1,536 games passed Command replay. Each precision
followed its own policy choices.

| Checkpoint | Precision | Repeats | Forward seconds, min / mean / max | Wall seconds, min / mean / max | Mean rows/call |
| --- | --- | ---: | ---: | ---: | ---: |
| Trained `w160l8` | float32 | 3 | 24.34 / 24.41 / 24.52 | 122.01 / 124.07 / 127.53 | 18.59 |
| Trained `w160l8` | bf16 | 3 | 40.85 / 40.90 / 40.94 | 139.56 / 139.89 / 140.25 | 18.66 |
| Seeded `w384l8` | float32 | 3 | 29.98 / 30.06 / 30.15 | 124.45 / 126.60 / 128.80 | 9.76 |
| Seeded `w384l8` | bf16 | 3 | 13.35 / 13.37 / 13.39 | 110.33 / 111.07 / 111.65 | 9.76 |

Bfloat16 lowered mean games/hour by 11.3% for `w160l8` and raised it by
14.0% for `w384l8`.

The receipts, CUDA traces, per-call timings and SHA256 inventory are in
`.runs/eval-precision-profile/l4-a468879a7a/`. The first admission trace
exposed Kineto's mirrored GPU annotations. Commit `75ae6a1d4` counts only
the CPU forward scopes and records native MHA alongside the SDPA dispatches.
The corrected run verified forward coverage and linked CUDA attention
kernels in all 84 profile receipts.

Etude rental `mtg324-l4-step-3207a8817628` ran under `--max-usd 5`. Etude
recorded USD 0.0916 billed and approximately USD 0.6130 pending. Etude and
RunPod confirmed deletion at 02:53 UTC on November 28.

## Reproduce

Use the [throughput setup](eval_throughput.md#reproduce) to install the
editable Etude SDK and fetch the trained checkpoint and trophy corpus.
Use Etude SDK revision `d876a7b1` to reproduce this run. Use trusted local
batch files from this checkout. A capture includes the arena logits and
completes once all four games per deal pass Command replay.

```bash
profile_dir=.runs/eval-precision-profile
trained=.runs/eval-throughput/policy/00002000-raw.safetensors

uv run --no-sync python -m experiments.eval_precision_profile capture \
  --checkpoint "$trained" --device cuda --batches "$profile_dir/trained-w160.batches" \
  --output-dir "$profile_dir/capture-trained"
uv run --no-sync python -m experiments.eval_precision_profile capture \
  --random-width 384 --device cuda --batches "$profile_dir/seeded-w384.batches" \
  --output-dir "$profile_dir/capture-wide"

uv run --no-sync python -m experiments.eval_precision_profile replay \
  --checkpoint "$trained" --batches "$profile_dir/trained-w160.batches" \
  --device cuda --variant bf16 --output-dir "$profile_dir/l4/trained-w160-bf16"
```

Use a fresh output directory per arm. Replay writes `calls.jsonl`,
`summary.json` and a Chrome trace in `profile.json`. The call records keep
the first pass and subsequent repeats. The trace records operator dispatch
and linked CUDA kernels. The bucket table applies the profiled wall-time
shares to the uninstrumented mean of the same calls. CUDA kernel intervals
take precedence over CPU dispatch during overlap. The receipt retains the
raw bucket times and the ratio of profiled to uninstrumented time, since
profiling can change that mix.

Generate the seeded head-width controls from the trained checkpoint's
readings and model settings.

```bash
uv run --no-sync python - "$trained" "$profile_dir" <<'PY'
from pathlib import Path
import sys

import torch

from experiments.eval_throughput import random_checkpoint

trained, out = map(Path, sys.argv[1:])
torch.set_num_threads(1)
for width, heads in ((160, 8), (160, 5), (160, 4), (256, 8), (288, 8), (384, 8)):
    random_checkpoint(out / f"seeded-w{width}-h{heads}.safetensors", width, heads,
                      template=trained)
PY
```

Capture the seeded `w160` control with `--checkpoint` too. For the grid,
replay the trained capture on all seven checkpoints and the seeded
`w160l8` capture on the two `w160l8` checkpoints. Pass `--grid-rows 1`,
`4`, `16` or `64`, with `--limit 12 --skip-rows 2048 --profile-calls 3`
for each precision. The supported variants are `float32`, `tf32`, `bf16`,
`bf16-linear-once` and `bf16-cached-autocast`. Run arena repeats with
`experiments.eval_throughput` and put their outputs alongside the replay
receipts before rendering tables.

```bash
uv run --no-sync python -m experiments.eval_precision_profile render_report \
  --receipts "$profile_dir/l4" --output "$profile_dir/l4-tables.md"
```
