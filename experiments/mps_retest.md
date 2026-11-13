# Torch MPS and MLX in a whole training update — MTG-335

Does torch MPS close the gap with MLX when both use bucketed shapes, MPS also
uses half precision, and training runs larger batches and models? Torch is
the incumbent. Keeping one backend is simpler; the additional MLX backend
needs a clear advantage where the current training recipe spends its time.
This is an exploratory measurement, with a USD 25 total rental cap. There is
no pre-registered adoption threshold. The recommendation will explain the
results and their limits after the measurements exist.

## Answer

**Not measured yet.** The harness and controlled correctness checks are in
place. The rented-M2 comparison, overflow measurements, evidence publication
and final recommendation remain outstanding. No speed or backend-adoption
conclusion follows from the implementation checks.

Per-call charts retain failed, unsupported, capacity-limited and unmeasured
cells. Generated `comparisons.md` tables expose first-call/setup time, warmed
rates, memory, gradient/logit/value/loss differences and sampled choices.
Each host gets separate charts and tables; replay receipts preserve the
capture host and its thread count. These outputs have been checked with
synthetic receipts, not performance evidence. Receipt completeness is not a
correctness verdict; the numerical differences must be interpreted separately.

The installed Etude 0.2.16 launcher exposes RunPod, Vast and Lium rental
providers, not an AWS dedicated-Mac provider. A capped AWS launch/release path
is still needed for the requested fresh M2 rental. AWS access and the prior
host's release were verified. The access notes' local release loop has no
budget admission or durable release guarantee; registering SSH in Etude only
records already-committed cost. No host was allocated by this implementation
step. This is a launch limitation, not missing credentials.

## Named deliverable: full context

`experiments/data/mps_retest/full_context.png` is the shareable counterpart to
a forward-only policy-call benchmark. It shows **seconds per whole update**,
stacked into engine/host work, collection policy calls and learning, with
**collected learner decisions per whole-update wall-second** alongside.

There is one panel per model shape and a group per 256, 512 and 1,024 streams.
Every group includes CPU8 float32, MPS float32, each supported MPS half dtype,
and MLX float32. Bars show the median-total replay of three repetitions and
its component times, so their stacks sum to the median total. Each group must
be measured on the same host within one hour. Does-not-fit, unsupported and
not-measured cells stay visible. Captions identify hardware, framework
versions, half dtype and parity evidence. The chart uses an opaque light
background for sharing in either light or dark interfaces.

Publish the actual PNG with `manabot.data`, embed it here and link it first
in the PR body after the run. M4 Max results get their own visibly labelled
chart, never bars mixed with M2 results. An L4 reference is optional only if
the same capture can be replayed through Etude for under USD 1 within the
whole-Task cap; the Apple harness does not implement that optional arm.

## Arms and inputs

| Arm | Precision |
|---|---|
| PyTorch CPU, 8 threads | float32; correctness and timing reference |
| PyTorch MPS | float32 |
| PyTorch MPS half | bfloat16 autocast |
| PyTorch MPS half | float16 autocast with a gradient scaler |
| MLX | float32 |

Both MPS half dtypes are attempted at every shape. An unsupported dtype is
recorded with its error; the other half arm remains required. Autocast warnings
that would silently run float32 are errors. Nonfinite gradients are errors,
not fast optimizer steps. MLX's current backend explicitly owns float32
arithmetic; adding its own mixed-precision implementation is outside this
retest. Regime defaults, live-token refusal off CUDA and reduced-precision
refusal off CUDA are unchanged. Autocast exists only in the harness.

| Axis | Points |
|---|---|
| Model | w64l2, w96l4, w160l8, w384l8, w512l8, w768l8, w1024l8 where memory permits |
| Whole-update streams | 256, 512, 1,024; both seats learning |
| Learner minibatch rows | 64, 256, 512, 1,024, 2,048 where memory permits |
| Collection call rows | 64, 256, 512 |

Three real w384l8 CPU captures use the same seed and recipe, changing only the
stream count. Every width replays each capture with identical initial weights
across arms; widths other than 384 use seeded fresh weights. w64l2 uses four
attention heads, matching batched-baseline; the other shapes use eight.
Per-call comparisons use the 256-stream capture. A larger batch consists of
consecutive retained rows with their own targets, never repeated rows from an
undersized capture. Combining rows with different loss settings is refused.

PyTorch uses `bucketed_shapes=True`. MLX applies the same bucket measurement
and reference-remapping code before its ordinary host preparation. Its object
joins and ownership follow the trimmed batch. GPU collection arms attend the
bucketed object table; CPU retains its production inference column compaction.
All arms encode the whole catalog in learning. Per-call times include their
host preparation, device transfer where needed and synchronized completion.

## What is measured

- **Learner step:** forward, production damped-policy loss, backward, clipping
  and Adam. The first call is separate from the median of three warmed calls.
  Each call resets weights, optimizer and scaler outside the measured interval,
  so timings use the same initial state. Gradients are unscaled before clipping.
- **Collection:** production inference scope with its catalog cached once;
  logits return to the host. Catalog setup and first-call time are separate.
- **Memory:** process lifetime peak RSS; MLX allocator peak; MPS driver peak
  sampled every 50 ms. The sampled peak is a lower bound, not an exact allocator
  high-water mark. Each arm/shape/phase runs in a fresh process.
- **Correctness:** legal-logit and value differences, full-gradient cosine,
  minimum tensor cosine and worst tensor against CPU float32. MLX comparisons
  use its independent parameter layout (including folded GRU biases). Tiny
  tensors use the existing parity check's global-norm floor. Common-uniform
  sampling comparisons report disagreements separately; near-identical logits
  do not establish identical decisions.
- **Whole update:** all original collection calls and learner minibatches,
  with evolving weights during learning. Engine time comes from the real
  capture, excluding policy and recording I/O time. It includes other
  non-policy collection host work at the capture's four CPU threads; policy
  and learner replays use eight. Three replays per arm and shape produce
  the full-context chart. Rate counts collected learner decisions before
  filtering, rather than optimizer exposures or bootstrap calls.

## Limits

This is one capture per stream count on one main host, not a population estimate.
The whole update is a replay with a shared engine component, not on-policy
training with each backend in the loop. Changed weights at other widths do not
change recorded actions or targets. Gradient agreement at initialization does
not establish equivalent learning trajectories. Larger batches, shapes that do
not fit, host load, first-call overhead and phase shares must remain visible
when interpreting ratios. The harness reports numbers and completeness; it
never chooses a backend automatically.

<details>
<summary>Method, process and evidence</summary>

**Main host:** fresh AWS mac2-m2, Apple M2, 24 GiB, Python 3.12, pinned torch
2.15 and the locked MLX version. All main bars use this host. Rental uses the
launcher's USD 25 cap and must include release after the 24-hour minimum.

**Overflow host:** Apple M4 Max, 128 GiB, at most eight CPU threads. Only
wider shapes or largest minibatches the M2 cannot hold belong here. Every
measured block checks the one-minute load is below 8. Otherwise the command
stops without timing; the operator must quiet the host before retrying. There
is no indefinite wait and no artificial load generation.
The replay path reuses the M2 capture's engine component even on the overflow
host. Receipts, captions and tables identify its M2 origin and capture threads;
the chart title labels a cross-host phase sum. This is not an independently
measured M4-only whole update. Missing capture provenance leaves coverage
incomplete; conflicting provenance is refused.

**Commands** inside the frozen source environment on the M2:

```sh
uv run --extra mlx python experiments/mps_retest.py capture capture-256 --streams 256
uv run --extra mlx python experiments/mps_retest.py matrix bench --capture capture-256
uv run --extra mlx python experiments/mps_retest.py replays bench --capture capture-256
uv run --extra mlx python experiments/mps_retest.py capture capture-512 --streams 512
uv run --extra mlx python experiments/mps_retest.py replays bench --capture capture-512
uv run --extra mlx python experiments/mps_retest.py capture capture-1024 --streams 1024
uv run --extra mlx python experiments/mps_retest.py replays bench --capture capture-1024
uv run --extra reports python experiments/mps_retest.py report experiments/data/mps_retest --bench bench
```

One cell can be rerun with `bench bench --capture capture-256 --phase learner
--width 384 --rows 256 --arm mps-bf16`; its same-source, same-host parity receipt
must exist first. `--phase parity` writes that receipt. The full matrix tries
both half dtypes. `replays` checks parity for each width/arm before its replay
blocks. `--widths` selects shapes for a partial run. Laptop overflow commands
must include `--laptop`; matrices and whole-update replays there also require
explicit wide `--widths`. No M2 measurement is claimed for a laptop run.

`bench.jsonl` and `updates.jsonl` retain failures, timeout/OOM distinctions,
source, capture identity, hardware, versions, dtype and before/after load.
CPU references retain capture and source provenance. Captures and their index
files must be retained alongside these ledgers, `summary.json`, the charts and
`comparisons.md` (hostname-suffixed for supplemental hosts). The tables retain
each whole-update repetition and errors, rather than only successful medians.
Report input refuses mixed source revisions or multiple captures at one stream
count on one host. New attempts supersede the corresponding cell in summaries;
raw receipts remain intact. Parity labels use the latest attempts for that
stream count. M2 report artifacts use `full_context.png`; other hosts have a
hostname suffix.

**2026-11-12:** The decision threshold was removed before any experiment run.
The steered matrix adds small/wide models, all three stream counts, both half
arms, whole-update phase stacks and a separate quiet-laptop overflow path.
Controlled checks exercised MLX bucket trimming and the harness on tiny fixed
inputs; they are implementation validation, not performance evidence.

**Cost:** no rental was launched in this step; a launcher's actual cost receipt
is still required. No cost is reconstructed from wall time.

</details>
