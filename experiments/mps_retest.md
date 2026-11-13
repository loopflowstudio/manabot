# Torch MPS and MLX in a whole training update — MTG-335 partial results

Does torch MPS close the gap with MLX when both use bucketed shapes and half
precision, and training runs larger batches and models? Torch is the incumbent.
Keeping one backend is simpler; MLX needs an advantage where the training recipe
spends its time. This is exploratory measurement under a USD 25 rental cap,
with no pre-registered adoption threshold.

## Answer

**At w384l8 / 256 streams, float32 MPS is close to float32 MLX, but MLX half
is faster than MPS half.** On the shared M4 Max with a four-thread CPU reference,
three CPU-paired replays at `140eec28d` give medians of 277 s for MPS float32,
263 s for MLX float32, 261–267 s for MPS half and 203–208 s for MLX half.
MLX uses about 20–24% less update time than the matching MPS half dtype here.
MPS float32 uses about 5% more time than MLX float32; shared-host variation
limits interpretation of that smaller difference.

| Arm | Whole update, s | Collection policy, s | Learning, s | Learner decisions/s |
|---|---:|---:|---:|---:|
| PyTorch CPU · float32 | 674.76 | 208.36 | 410.42 | 71.23 |
| PyTorch MPS · float32 | 277.32 | 102.16 | 119.18 | 173.31 |
| MLX · float32 | 263.20 | 72.97 | 134.25 | 182.61 |
| PyTorch MPS · bfloat16 | 261.04 | 86.33 | 118.73 | 184.13 |
| MLX · bfloat16 | 207.58 | 55.75 | 95.85 | 231.54 |
| PyTorch MPS · float16 | 266.53 | 86.68 | 123.87 | 180.33 |
| MLX · float16 | 203.41 | 54.76 | 92.66 | 236.29 |

The shared engine/host component is 55.98 s, retained from the same laptop's
`99210bf8d` capture. The executable replay uses the current shared update and
independent MLX GRU biases. These are reconstructed phase sums, not end-to-end
on-policy training wall times. Collection includes catalog setup and first
calls; loading captured arrays and common-uniform choice verification are
outside the replay timers. Non-policy collection host work is retained in
the captured engine/host component.

Only CPU and one accelerator are resident in each pair. Every repetition runs
all six pairs. Each accelerator bar is its own three-replay median; the CPU bar
is the median of three means across six paired CPU references. Its phase stack
is an aggregate, not one observed replay. Each accelerator's contemporaneous
CPU reference, loads and memory scopes remain in the detailed table. CPU ratios
help describe the comparisons but do not correct every shared-host effect.

All 48,064 sampled collection choices matched CPU in every selected replay.
There were four load retries: every MPS bfloat16 repetition and the final MLX
float16 repetition. Both attempts remain in the ledger; lower mean endpoint
load selects the entire pair. Endpoint load includes the benchmark itself and
does not measure background activity throughout the block.

**Recommendation:** keep PyTorch in production and retain MLX as an experimental
Apple backend. Float32 MPS has narrowed the performance case for a second
float32 backend at this shape. MLX's half paths retain a larger measured speed
advantage, which supports further accuracy work rather than removing MLX.
Every accelerator, including float32, exceeded the 1e-6 one-step parameter
check on these real inputs. Matching sampled choices does not establish
matching optimization or playing strength. Defaults and off-CUDA precision
refusals remain unchanged.

This result does not answer whether larger batches close the gap: the renewed
512-stream capture again stopped at the unchanged 20 GiB process-tree guard
before learning. No replayable 512/1024-stream capture exists. The w768/64-row
correctness probe executed all seven arms, but provides no whole-update timing;
its accelerator steps also exceeded the float32 bound. The w64 rerun stopped
on the wrap-up direction after one complete six-pair repetition: no median
is reported. No w160, new 512-stream block or 1024-stream block started.

The renewed M2 w64 and w160 attempts stopped during isolated CPU parity on
small swap-in counter increases. They are literal paging-guard stops, not
evidence of sustained pressure or standalone backend capacity failures. No
new M2 whole-update result exists. The operator says the instance is terminated and the host releases tomorrow
(2026-11-14); its exact release timestamp remains pending. Historical results below belong to
`fa23fc364`, before the shared update and independent MLX GRU biases, and are
not pooled with the current measurements.

No independent warmed per-call matrix completed. Whole-replay phase totals
include first-call costs and do not establish a batch-size crossover. Torch
2.15.0 and MLX 0.32.4 were used on both hosts; the laptop uses Python 3.13 and
four CPU threads, while M2 uses Python 3.12 and eight replay threads.

## Enlarged-arm correctness at the current replay source

These checks used source `140eec28d` on the M4 Max, 64 learner rows from the
original 256-stream capture. They check one initial minibatch; completed timing does not establish
equivalent optimization trajectories. Both MLX half dtypes executed. All
accelerators exceeded the shared 1e-6 absolute one-step parameter tolerance,
including float32 arms whose logits and gradients otherwise nearly matched.
The supported MLX API now defaults to float32 and offers both half dtypes
explicitly. Its dtype-specific gates cover logits, values, relative loss and
full-gradient cosine; finite clipped steps are required, while step deltas and
tensor minima remain diagnostics. The retained measurements were made before
that API promotion and are not rewritten as a new-source run.
A separate bounded check of the supported API on two real learner rows passes
float32 and float16, but bfloat16's relative loss difference is **1.07%**, above
its unchanged **1%** gate; its other gates pass. The
[validation receipts](data/mps_retest/laptop/paired-update/validation/README.md)
retain that counterexample. Use the parity command on the intended checkpoint
and capture before relying on a half-precision configuration. After syncing
main and rebuilding the native extension, a fresh bounded capture passes all
three MLX dtypes; [those receipts](data/mps_retest/laptop/paired-update/validation/after-sync/README.md)
stay separate from the earlier counterexample and the timed evidence.

| Arm | Legal logit Δ | Value Δ | Relative loss Δ | Global gradient cosine | Minimum tensor cosine | One-step parameter Δ | Choice mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch CPU · float32 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| PyTorch MPS · float32 | 7.217749953e-09 | 4.321336746e-07 | 1.162997701e-07 | 0.9999999974 | 0.9999999561 | 9.932555258e-06 | 0 |
| MLX · float32 | 8.381903172e-09 | 4.172325134e-07 | 2.809461134e-07 | 0.9999999981 | 0.9999999711 | 5.504116416e-06 | 0 |
| PyTorch MPS · bfloat16 | 5.927821621e-05 | 0.003815792501 | 0.00206529002 | 0.9990542327 | 0.9630805262 | 0.0001747384667 | 0 |
| MLX · bfloat16 | 7.632025518e-05 | 0.003880620003 | 0.0003955238141 | 0.9975223392 | -1 | 0.0001899059862 | 0 |
| PyTorch MPS · float16 | 6.433983799e-06 | 0.0005453228951 | 0.0001111444506 | 0.9993675361 | -1 | 0.0001827236265 | 0 |
| MLX · float16 | 9.342562407e-06 | 0.0007421076298 | 0.0001016489573 | 0.9998640781 | -1 | 0.0001506153494 | 0 |

Tensor minima include tiny gradients; the exact score-bias gradient is zero.
That does not erase observed one-step differences. Timing remains diagnostic,
not evidence of equivalent learning trajectories.

## Full context: completed seven-arm comparison

![M4 Max w384l8, 256 streams: measured phase stacks and decisions per second](data/mps_retest/full_context.png)

[Shareable PNG](data/mps_retest/full_context.png) ·
[all paired references, loads and memory](data/mps_retest/laptop/paired-update/report/comparisons.md) ·
[machine-readable summary](data/mps_retest/laptop/paired-update/report/summary.json).

Data bytes live in the team's authenticated artifact store, not GitHub. Restore
the PNG and the detailed evidence from the committed manifest:

```sh
uv run --extra artifacts python -m manabot.data fetch experiments/data/mps_retest/full_context.png experiments/data/mps_retest/laptop/paired-update
```


The primary chart includes only the completed current-source panel. Each bar
stacks engine/host work, collection policy and learning; the strip divides
48,064 learner decisions by the whole-update phase sum. The CPU aggregation
is labelled. Half precision has its own numerical tolerances and is not
trajectory-equivalence evidence. Hosts never share an axis. The earlier
[source-fa23fc364 chart](data/mps_retest/full_context-fa23fc364.png) is historical;
its older update implementation and different host conditions preclude pooling.

### Whole-update coverage after the final stop

Every requested host/width/stream/arm is listed. **Measured** means three
selected complete replays at `140eec28d`, not that float32 parity passed.
**Failed: capture memory** identifies the shared 512-stream prerequisite at
`7924086ad`: a 20 GiB process-tree guard, not a backend allocation failure.
Peak sampled process-tree RSS was 21,598,879,744 bytes; 169 collection calls,
no learner minibatches. The later 96 GiB laptop allowance was never exercised:
the wrap-up direction arrived before a new larger-stream block started.
**Failed: paging guard** identifies isolated CPU parity at `7b07ee0ab`, with
small swap-ins and no new swap-outs; it does not prove sustained pressure.
**Not reached** means no completed three-replay result. M4 w64/256 has one
repetition; w768/256 has correctness only. Neither becomes a timing bar.
Older results and failures at `fa23fc364` stay in the historical sections.

<!-- coverage:start -->
| Host | Model | Streams | CPU f32 | MPS f32 | MLX f32 | MPS bf16 | MLX bf16 | MPS f16 | MLX f16 |
|---|---|---:|---|---|---|---|---|---|---|
| M4 Max | w64l2 | 256 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M4 Max | w64l2 | 512 | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory |
| M4 Max | w64l2 | 1024 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M4 Max | w96l4 | 256 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M4 Max | w96l4 | 512 | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory |
| M4 Max | w96l4 | 1024 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M4 Max | w160l8 | 256 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M4 Max | w160l8 | 512 | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory |
| M4 Max | w160l8 | 1024 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M4 Max | w384l8 | 256 | Measured | Measured | Measured | Measured | Measured | Measured | Measured |
| M4 Max | w384l8 | 512 | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory |
| M4 Max | w384l8 | 1024 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M4 Max | w512l8 | 256 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M4 Max | w512l8 | 512 | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory |
| M4 Max | w512l8 | 1024 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M4 Max | w768l8 | 256 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M4 Max | w768l8 | 512 | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory |
| M4 Max | w768l8 | 1024 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M4 Max | w1024l8 | 256 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M4 Max | w1024l8 | 512 | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory | Failed: capture memory |
| M4 Max | w1024l8 | 1024 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w64l2 | 256 | Failed: paging guard | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w64l2 | 512 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w64l2 | 1024 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w96l4 | 256 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w96l4 | 512 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w96l4 | 1024 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w160l8 | 256 | Failed: paging guard | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w160l8 | 512 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w160l8 | 1024 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w384l8 | 256 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w384l8 | 512 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w384l8 | 1024 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w512l8 | 256 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w512l8 | 512 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w512l8 | 1024 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w768l8 | 256 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w768l8 | 512 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w768l8 | 1024 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w1024l8 | 256 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w1024l8 | 512 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
| M2 | w1024l8 | 1024 | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached | Not reached |
<!-- coverage:end -->

## Historical w384 comparison at `fa23fc364`

The earlier five-resident-arm M4 w384/256 group measured 703.20 s on CPU4,
521.22 s on MPS float32, 478.42 s on MPS bfloat16, 442.03 s on MPS float16
and 402.22 s on MLX float32. The three blocks spanned 6,870.95 s; endpoint
loads were 119.12→57.56, 60.47→30.57 and 30.98→47.43, with no retries.
All sampled choices matched CPU. These older source and residency conditions
prevent attributing the difference from the renewed timings to any one code
change. The [historical M2 table](data/mps_retest/comparisons.md) and
[historical M4 table](data/mps_retest/comparisons-Jacks-MacBook-Pro-2.local.md)
retain their receipts and numerical checks.

## Historical small models change the ordering

At w64l2 and 256 streams, CPU beat every MPS precision on both hosts. Half
precision did not provide the large-shape advantage seen at w384l8. These
points change both width and attention depth; they do not locate a precise
width-only crossover.

| Host | CPU float32, s | MPS float32, s | MPS bfloat16, s | MPS float16, s | MLX float32, s |
|---|---:|---:|---:|---:|---:|
| M2, CPU8 | 107.75 | 141.98 | 144.15 | 142.93 | 84.00 |
| M4 Max, CPU4 | 110.18 | 136.38 | 135.82 | 136.08 | 95.94 |

Each row is a separate host's median of three selected repetitions. On M2,
MLX's median-total replay spent 23.31 s in collection policy calls and 50.89 s
in learning, against CPU's 28.55 s and 69.42 s, plus 9.79 s of shared
engine/host work. On M4 Max, those MLX phases were 8.48 s and 31.49 s,
against CPU's 19.51 s and 34.69 s, plus the much larger 55.98 s shared
engine component. That reused engine cost makes cross-host totals misleading.

M2 loads were 2.29→3.60, 3.35→3.35 and 2.91→3.80; no retry was needed.
M4 selected loads were 39.08→86.73, 91.23→61.18 and 55.81→161.78.
The first and third M4 repetitions each ran one retry; the original attempt
had the lower mean endpoint load in both cases and was selected. Both retries
remain in the ledger. All sampled collection choices matched CPU.

The small M2 setup completed, but it was not free of paging: swap-outs rose
by 517,550 16-KiB pages (7.90 GiB) between 05:46:05 and 06:19:03 UTC,
while used swap ended near 257 MiB. These host counters include the setup
and cleanup interval; they cannot assign paging cost to an arm. Treat the
M2 values as this shared-residency comparison, not isolated-backend timing.

## Historical numerical agreement and memory

The initial 64-row gradient check used the w384l8 model and each host's own
256-stream capture. Both MPS half dtypes executed; neither silently fell back
to float32. These checks and matching collection choices do not establish
equivalent optimization trajectories.

| M4 Max arm | Global gradient cosine | Minimum tensor cosine | Max legal-logit difference | Max value difference |
|---|---:|---:|---:|---:|
| MPS float32 | 0.999999996 | 0.999999952 | 8.15e-9 | 3.58e-7 |
| MPS bfloat16 | 0.999054233 | 0.963080526 | 5.93e-5 | 0.003816 |
| MPS float16 | 0.999367535 | -1.000000000 | 6.43e-6 | 0.000545 |
| MLX float32 | 0.999999977 | 0.999998970 | 8.38e-9 | 3.87e-7 |

The worst bfloat16 tensor was `policy_head.0.bias`; the float16 minimum was
`policy_head.2.bias`. The latter is a softmax-shift bias whose exact gradient
should be zero; its cosine alone cannot establish a meaningful update error.
The existing global-norm floor did not classify both compared tensors as zero.
The half arms' value and full-gradient differences still need interpretation;
the receipt's `passes_gradient` field checks only global cosine above 0.999,
not every tensor or every numerical tolerance.

M2 float32 MPS and MLX global/minimum cosines were above 0.9999996. M2
bfloat16 measured global cosine **0.996963478**, minimum -1 at
`policy_head.2.bias`; float16 measured **0.999100406**, minimum 0.996795989
at `continuation.projection.2.bias`. All initial common-uniform sample
choices matched CPU. This is parity evidence only; the M2 has no complete
w384 whole-update timing.

| M4 Max arm | Peak process RSS, GiB | Sampled MPS driver peak, GiB | MLX allocator peak, GiB |
|---|---:|---:|---:|
| CPU4 float32 | 7.971 | — | — |
| MPS float32 | 1.515 | 12.927 | — |
| MPS bfloat16 | 1.532 | 9.008 | — |
| MPS float16 | 1.406 | 8.993 | — |
| MLX float32 | 0.998 | — | 11.069 |

These are the largest observed peaks across the three completed replays.
RSS, driver allocation and MLX allocation measure different scopes; do not
sum or compare them as interchangeable process totals. MPS driver sampling
every 50 ms is a lower bound. All five arm processes were resident together.

## Planned matrix and inputs

| Arm | Precision |
|---|---|
| PyTorch CPU, 4 laptop / 8 M2 threads | float32; correctness and timing reference |
| PyTorch MPS | float32 |
| PyTorch MPS half | bfloat16 autocast |
| PyTorch MPS half | float16 autocast with a gradient scaler |
| MLX | float32 |
| MLX half (new runs) | bfloat16; float16 with fixed loss scale 128 |

The harness attempts both MPS half dtypes at every selected shape. An unsupported dtype is
recorded with its error; the other half arm remains required. Autocast warnings
that would silently run float32 are errors. Nonfinite gradients are errors,
not fast optimizer steps. The new required MLX bfloat16 arm and executable float16 arm use half
weights and activations in each forward, with float32 master weights, loss
and Adam state. Float16 uses a fixed loss scale of 128; overflow refuses the
step. This has the same state-precision split as MPS autocast, not an
identical operator-by-operator autocast policy. Regime defaults stay float32. Live-token and reduced-precision refusals off
CUDA are unchanged. MPS autocast is selected only in the harness.

| Axis | Points |
|---|---|
| Model | w64l2, w96l4, w160l8, w384l8, w512l8, w768l8, w1024l8 where memory permits |
| Whole-update streams | 256, 512, 1,024; both seats learning |
| Learner minibatch rows | 64, 256, 512, 1,024, 2,048 where memory permits |
| Collection call rows | 64, 256, 512 |

The planned w384l8 CPU captures use the same seed and recipe, changing only
the stream count. Only the 256-stream captures completed. Every width replays each capture with identical initial weights
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

## Measurement definitions

- **Learner step:** forward, production damped-policy loss, backward, clipping
  and Adam. The first call is separate from the median of three warmed calls.
  Each call resets weights, optimizer and scaler outside the measured interval,
  so timings use the same initial state. Gradients are unscaled before clipping.
- **Collection:** production inference scope with its catalog cached once;
  logits return to the host. Catalog setup and first-call time are separate.
- **Memory:** process lifetime peak RSS; MLX allocator peak; MPS driver peak
  sampled every 50 ms. The sampled peak is a lower bound, not an exact allocator
  high-water mark. Each arm/shape/phase runs in a fresh process. Historical comparisons kept all
  arm models resident. The new paired comparison keeps CPU and one accelerator
  resident; only one process computes at a time. Allocation or paging limits
  describe that comparison setup, not an isolated arm.
- **Correctness at the measured revision:** legal-logit and value differences, full-gradient cosine,
  minimum tensor cosine and worst tensor against CPU float32. Historical MLX comparisons
  used folded GRU biases; the current source preserves independent GRU biases
  and also compares loss and the clipped one-step parameter update. Tiny
  tensors use the existing parity check's global-norm floor. Common-uniform
  sampling comparisons report disagreements separately; near-identical logits
  do not establish identical decisions.
- **Whole update:** all original collection calls and learner minibatches,
  with evolving weights during learning. Engine time comes from the real
  capture, excluding policy and recording I/O time. It includes other
  non-policy collection host work at the capture's four CPU threads; policy
  and learner replays use four threads on the laptop, eight on M2. Three replays per arm and shape produce
  the full-context chart. Rate counts collected learner decisions before
  filtering, rather than optimizer exposures or bootstrap calls.

## Limits

Only one 256-stream capture completed on each of two hosts; there is no
replayable 512/1024-stream capture and no population estimate.
The whole update is a replay with a shared engine component, not on-policy
training with each backend in the loop. Changed weights at other widths do not
change recorded actions or targets. Gradient agreement at initialization does
not establish equivalent learning trajectories. Larger batches, shapes that do
not fit, host load, first-call overhead and phase shares must remain visible
when interpreting ratios. The harness reports numbers and completeness; it
never chooses a backend automatically.
The delivery sync brought newer native observation shapes from main. Retained
captures remain bound to their original world and source; reproduce the timed
results from `140eec28d`, not by rebinding those arrays to the delivery tree.


The M2 comparison has already shown active paging with all five models
resident: between 04:24:04 and 04:25:37 UTC, swap-ins increased by 361.98 MiB
and swap-outs by 5.52 GiB. Used swap reached 9,033.75 MiB at 04:41:43 UTC.
These observations qualify the shared-memory comparison setup; they do not
establish whether any backend fits or performs well by itself. Do not equate
paging with an allocation failure or label an unattempted arm “does not fit.”
At 05:29 UTC used swap was 10,518.25 MiB, with no complete replay after about
105 minutes. The series was cancelled and workers exited. Pairwise
interleaving would change the all-arm comparison and its single CPU reference;
it was not substituted in that series. Later direction authorized pairwise
replay; the new source and its explicit CPU aggregation preserve that change
in method rather than pooling both series.

<details>
<summary>Method, process and evidence</summary>

**M2 host:** fresh AWS mac2-m2, Apple M2, 24 GiB, Python 3.12, pinned torch
2.15 and the locked MLX version. The operator allocates it through AWS, writes
`~/.manabot-mps-bench/access.md`, and releases it using the 24-hour guard in
`release.sh`. Allocation is 2026-11-13T02:56:06Z; exact release remains pending
as recorded in Cost below (24-hour minimum, about USD 21; USD 25 cap).

**Laptop host:** Apple M4 Max, 128 GiB, four CPU threads for capture and the
CPU reference. Start captures and comparison blocks regardless of load,
recording the one-minute load at both endpoints. A historical comparison block is all arms at one width and per-call size/phase,
or one whole-update repetition. Under `--pairwise`, each CPU/accelerator pair
is one whole-update block, with its own retry and retained reference. Resident isolated processes advance one collection
call or learner step per arm in order. The start and end loads belong to the
entire comparison; there is no mid-block load refusal that would selectively
truncate slower arms.

On either host, if end load exceeds twice start load, repeat the entire replay
once,
without a load threshold. Keep the attempt with the lower arithmetic mean
of endpoint loads; a tie keeps the first. Retain both attempts and select all
arms together. An unavailable repeat leaves coverage incomplete. Per-call blocks on both hosts and laptop captures also retain and repeat
load increases over 2×, keeping unused captures beside the selected one.
Tables carry the block and capture endpoint loads; the measured-only chart
keeps a short source caption. Repeated
blocks are identified. This reduces shared-host drift; it does not make a busy
laptop equivalent to a dedicated host. No artificial load is generated.

Keep each host's captures and ledgers in separate directories. Capture locally
for a same-host whole-update comparison. A replay of another host's capture
retains that capture's engine component: receipts, captions and tables identify
its origin and threads, and the chart title labels a cross-host phase sum.
Missing capture provenance leaves coverage incomplete; conflicting provenance
is refused.

**Reproduction commands** inside the measured `fa23fc364` source environment
with a real Etude launcher run identity. The ported checkout uses the same
commands but its seven-arm results belong to a new source group. These are
measurement commands, not rental commands. Prepare the M2 environment with
`uv sync --locked --python 3.12 --extra play --extra mlx --extra reports
--extra artifacts` and build its matching native extension. Explicit interpreter
selection avoids the checkout's Python 3.13 development pin. Start with a selected width to
establish replay durations before scheduling the full matrix; the default
worker timeout is 900 seconds and a timeout is a failed attempt, not OOM.
Three interleaved replays run back to back on the same host, without a
one-hour admission or reporting limit. Each repetition
interleaves all executable arms call by call and step by step, or runs all six
CPU/accelerator pairs in order when `--pairwise` is selected. Their total
elapsed span, including the gaps spent running the other arms, determines the
recorded window; phase sums exclude those gaps. The timeout applies to an
individual worker call, not a complete update.

```sh
uv run --python 3.12 --no-sync python experiments/mps_retest.py capture capture-256 --streams 256
uv run --python 3.12 --no-sync python experiments/mps_retest.py matrix bench --capture capture-256
uv run --python 3.12 --no-sync python experiments/mps_retest.py replays bench --capture capture-256
uv run --python 3.12 --no-sync python experiments/mps_retest.py capture capture-512 --streams 512
uv run --python 3.12 --no-sync python experiments/mps_retest.py replays bench --capture capture-512
uv run --python 3.12 --no-sync python experiments/mps_retest.py capture capture-1024 --streams 1024
uv run --python 3.12 --no-sync python experiments/mps_retest.py replays bench --capture capture-1024
uv run --python 3.12 --no-sync python experiments/mps_retest.py report experiments/data/mps_retest --bench bench
```

Run a correctness cell with `bench bench --capture capture-256 --phase parity
--width 384 --rows 256 --arm mps-bf16`. Timed work uses `matrix` or `replays`
with `--widths 384` to select a width; single-arm serial timing is refused.
`matrix` tries both half dtypes at every shape. `replays` checks parity for
each width/arm before its replay blocks. Admission requires an executed parity
cell with matching host, software, source and capture, plus the row count for
per-call timing. It does not require numerical agreement; gradient and decision
differences remain evidence to interpret. If parity cannot execute, the timed
cell retains that failure's status and identifies parity as its source; an OOM
there is not a measured timed-arm allocation peak. Laptop commands select `--python 3.13` and must include `--laptop`, including
`capture`; every width is allowed. CPU receipts use the
arm name `cpu` and retain their actual thread count. No M2 measurement is
claimed for a laptop run.

`bench.jsonl` and `updates.jsonl` retain failures, timeout/OOM distinctions,
source, capture identity, hardware, versions, dtype and before/after load.
CPU references retain capture and source provenance. Captures and their index
files must be retained alongside these ledgers, `summary.json`, the charts and
`comparisons.md` (hostname-suffixed for supplemental hosts). The tables retain
each whole-update repetition and errors, rather than only successful medians.
Report input refuses mixed source revisions or multiple captures at one stream
count on one host. New attempts supersede the corresponding cell in summaries;
raw receipts remain intact. Load retries retain both attempts and append an
explicit selection of the quieter whole block. Captures retain `selection.json`
and each attempt’s `provenance.json`; publish the unused attempt too. Parity labels use the latest attempts for that
stream count. The combined measured-only chart uses `full_context.png`; supplemental
host renderings keep their hostname suffix. Host identity remains
in every chart and table regardless of filename.

**2026-11-12:** The decision threshold was removed before any experiment run.
The steered matrix adds small/wide models, all three stream counts, both half
arms, whole-update phase stacks and a separate quiet-laptop measurement path.
Controlled checks exercised MLX bucket trimming and the harness on tiny fixed
inputs; they are implementation validation, not performance evidence.

**Preparation:** the laptop used its locked Python 3.13 environment with
Torch 2.15.0 and MLX 0.32.4. Obsolete load-under-8 and load-under-16 rules
prevented measurement before 02:16 UTC. Their detailed preflight notes remain
in `fa23fc364:experiments/mps_retest.md`. The default Etude database was
incompatible with the installed launcher; a separate `.runs/mps-etude` home
kept it untouched. No rental tooling was added.

**2026-11-13 02:16 UTC direction:** remove the laptop load threshold entirely;
retain endpoint load labels, four threads, interleaving and the drift retry.
M2 access is optional for delivery; laptop evidence is the deliverable if it
does not arrive.

**2026-11-13 02:21 UTC first execution:** source `99210bf8d`, local Etude
run `c91b93dcc210`, M4 Max, load 69.68→69.68. Capture failed before any games
with `SemanticProjectionError: catalog_tokens.keys: expected array`; the
local native extension predates the checked-out catalog API. Retain its receipt
under `laptop/capture-256-failed-c91b93dcc210`. Rebuild the extension from this
source before retrying.

**2026-11-13 02:37 UTC capture:** the CPython 3.13 extension was rebuilt
from `99210bf8d` and installed atomically. Etude `f800789ff669` completed
the 256-stream w384l8 capture: 48,064 learner decisions, 792 collection calls,
188 learner minibatches, 624.57 s capture wall time. Shared engine/host work
is 55.98 s after subtracting policy and recording I/O; capture load
77.81→85.09, four threads, no drift retry.
This capture is input evidence, not a backend comparison.

**2026-11-13 laptop replay feasibility:** Etude `86e16cc560fc` completed
one five-arm w384/256 block in 2,425.13 s. Load 54.52→21.48 required a
drift retry. The continuation was cancelled during that retry when M2 became
the primary host: three blocks at this observed rate take about two hours,
not one. Retain the completed block as provisional evidence; no selected
three-replay median exists and the retry is incomplete.

**2026-11-13 03:38 UTC direction:** remove the one-hour window. Run three
interleaved replays back to back on each host, recording wall-clock endpoints
and load. Retry only when end load exceeds twice start load, once at most;
retain both attempts and select the lower mean endpoint load. Load falling
by more than half does not trigger a retry. The laptop window ends 06:17 UTC;
priorities are w384 at 256/512/1024 streams, w768 and w1024 at 256, then
smaller widths. Report cells not reached. Restart both current replay series
from one new freeze; retain earlier receipts separately instead of combining
repetitions separated by cancellations or source changes.

**2026-11-13 back-to-back execution:** source `fa23fc364`, laptop Etude
`5bd37fb3abaa`, M2 Etude `877fa3cf766d`. The M2's independent capture
`bfb4d0e22dd7` at `99210bf8d` retained 48,064 learner decisions, 792 collection
calls and 188 learner minibatches in 1,223.33 s; its shared engine/host
component is 9.79 s. The M2's earlier `5884f474a199` series was cancelled
during its first replay to apply the revised method before restarting all
three repetitions. Current receipts live in each host's `bench-back-to-back`;
earlier `bench` receipts remain separate.

**2026-11-13 05:29 UTC design counterexample:** cancelled M2 `877fa3cf766d`
for sustained paging in the five-resident-arm setup. No OOM or whole-update
timing is claimed. Retain the exact M2 capture, both parity series and references,
and launcher receipts in `m2-evidence-fa23fc364.tar.gz`; retain raw memory
observations separately in `m2/host-observations.jsonl`. The smaller w64l2
comparison subsequently completed, with its more modest paging recorded above.
Further M2 scaling awaits a memory-bounded comparison design.

**Completed replay series at `fa23fc364`:** laptop w384 `5bd37fb3abaa`
completed three blocks without retries; laptop w64 `7fb121dc08a6` completed
three selected blocks and two retained retries, ending at 06:19:45 UTC;
M2 w64 `133dacc3287a` completed three blocks without retries, ending at
06:16:29 UTC. Their full spans were 6,870.95 s, 1,828.49 s and 1,814.93 s,
respectively. Source, run identity, loads, endpoints and the original attempts
remain in the per-host `bench-back-to-back` ledgers.

**Larger captures at `fa23fc364`:** laptop 512-stream `dcf93e86b142` and
1,024-stream `88dd067a2d9b` failed the 20 GiB process-tree memory guard before
learning. Their exact partial captures are in
`laptop/capture-failures-fa23fc364.tar.gz`, with launcher logs in
`laptop/run-receipts`. No budget was raised and no missing replay arm was
labelled as an OOM.

**Retained artifacts:** `bench.jsonl` and `updates.jsonl` at the report root
are byte-concatenated report inputs from the two host ledgers. `summary.json`,
six PNGs and the two `comparisons*.md` tables are generated from those inputs.
`laptop/captures-99210bf8d.tar.gz` holds the original successful and failed
256-stream laptop captures; `m2-evidence-fa23fc364.tar.gz` holds the M2 capture,
old and current ledgers, references, observations and launcher receipts.
The archives preserve the original array bytes. Use `manabot.data` to fetch
the named paths, then unpack the tar archives at the repository root to replay
from the measured source. These are partial results: the missing matrix
cells and the M2 memory-design problem remain open.

**2026-11-13 shared-update port and MLX half:** source `7924086ad` repaired
capture and replay against the shared update contract. The bounded both-seat
capture replay test passed; source `7b07ee0ab` added measurement-only MLX half
forwards with float32 master weights, loss and Adam state. Both bfloat16 and
float16 executed gradients and a clipped step in the controlled fixture; the
float16 path uses fixed loss scaling at 128. This is not numerical-equivalence
evidence for the real training shapes. The seven-resident-arm w384/256 laptop attempt `d85c036cb749` retained the
original `99210bf8d` capture inputs and engine component, but was cancelled
before a complete replay when laptop swap grew substantially. At 11:11:45 UTC
used swap was 14,474.12 MiB; at 11:15:23 it was 35,306.00 MiB, with
20.8 GiB of new swap-outs. This is a residency limit of the shared host
comparison, not a standalone backend capacity result. Its parity receipts
and cancellation evidence remain separate. CPU-paired replay is authorized
and now implemented at `140eec28d`: only CPU and one accelerator are resident,
with calls and steps alternating. Each of three repetitions contains all six
pairs. The CPU bar is the median of three means across its six paired CPU
references; every individual CPU timing and load is retained in the table.
Accelerator bars remain medians of their own three replays.

**New 512-stream capture failure:** `05d01891989a` at `7924086ad` again hit
the unchanged 20 GiB process-tree guard before learning: 169 retained collection
calls, 331.50 s, load 340.41→300.78. Chunked learner admission does not solve
the collector's full-capacity observation storage. Its partial arrays and
receipt are retained under `laptop/shared-update/`. The subsequent direction
authorized 75% of laptop RAM, but the final stop arrived before that work began.
No later capture or 512-stream replay ran; this is not a backend OOM.

**New M2 stops:** `d475968d4499` (w64) and `ec717373d298` (w160), both at
`7b07ee0ab`, stopped during isolated CPU parity. Swap-ins rose by 27 and 4
pages respectively; swap-outs remained 20,676,478. These are literal
paging-guard stops, not evidence of sustained pressure or seven-arm allocation
failure. Pairwise replay would not reduce below the single resident arm at
that boundary. No w384 retry started under the no-paging condition. The raw
counters and failures are published as `m2-shared-update-7b07ee0ab.tar.gz`.
The operator was notified that M2 blocks are stopped and workers have exited.


**Current paired completion and final stop:** `6b7892940c26` completed the
w384/256 seven-arm comparison at source `140eec28d`, with all 18 selected
CPU/accelerator blocks and four retained retries. The selected receipt span
is 17,202.69 s; actual execution includes the unselected attempts and lasted
about five hours. `f11d5fa70f69` completed seven-arm w768/64-row correctness
only. `4c3b4b17a851` was cancelled at 16:44 UTC after one full w64 repetition
on the explicit wrap-up direction. Neither an uncompleted repetition nor the
correctness-only probe is used in a median. Raw ledgers, reference arrays,
launcher sources and terminal Etude receipts remain under
`laptop/paired-update/`. No larger-stream or CUDA block started.

**Delivery checks:** after syncing main and rebuilding the CPython 3.13
native extension, 81 affected backend, capture, inference, bucket-shape,
experiment-discovery and artifact tests pass. The focused package-boundary,
capture, discovery, CI-selection and release checks also pass (111 tests).
Ruff and diff checks pass. The public parity path passes all three MLX dtypes
on a new two-row real capture; the earlier bfloat16 loss-gate failure is
retained above. Injected reporting checks confirm incomplete repetitions and
pending retries never become bars. Native rules changes came from main; this
Task makes no Rust changes, and the native suite remains CI's responsibility.

**Supported MLX precision:** the final implementation replaces the temporary
mixed-precision subclass with one policy-owned precision option shared by
inference and the learner, defaulting to float32. The
[backend guide](../manabot/backends/mlx.md#compute-precision) states loss scaling,
float32 probability normalization and the half-specific numerical gates.
The gates were calibrated from the retained w384/w768 diagnostics; this is
not an independent validation population. Training quality and trajectories
remain unmeasured. Torch defaults and precision refusals are unchanged.

**Cost:** the operator reports allocation at 2026-11-13T02:56:06Z, with a
24-hour minimum of about USD 21 under the USD 25 cap. At 16:43 UTC on
2026-11-13 the operator said, “the instance is terminated; the host releases
tomorrow.” Exact host release time remains operator-owned and unverified. No
new M2 work or rental was launched after that direction.
Etude local-job USD 0.00 receipts exclude this
operator-managed rental; they are not the whole experiment cost.

</details>
