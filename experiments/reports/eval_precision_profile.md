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

The L4 run is pending. The operator dispatch and crossover remain open.

## Reproduce

Use the [throughput setup](eval_throughput.md#reproduce) to install the
editable Etude SDK and fetch the trained checkpoint and trophy corpus. Use
trusted local batch files from this checkout. A capture includes the arena
logits and completes once all four games per deal pass Command replay.

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
for width, heads in ((160, 8), (160, 5), (256, 8), (288, 8), (384, 8)):
    random_checkpoint(out / f"seeded-w{width}-h{heads}.safetensors", width, heads,
                      template=trained)
PY
```

Capture the seeded `w160` control with `--checkpoint` too. For grid arms
pass `--grid-rows 1`, `4`, `16` or `64` to replay. Reuse the same
`--limit` and `--skip-rows` for each precision in a comparison. The
supported variants are `float32`, `tf32`, `bf16`, `bf16-linear-once`
and `bf16-cached-autocast`. Run arena repeats with
`experiments.eval_throughput` and put their outputs alongside the replay
receipts before rendering tables.

```bash
uv run --no-sync python -m experiments.eval_precision_profile render_report \
  --receipts "$profile_dir/l4" --output "$profile_dir/l4-tables.md"
```
