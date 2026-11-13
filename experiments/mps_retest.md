# Torch MPS and MLX in a whole training update — MTG-335 partial results

Does torch MPS close the gap with MLX when both use bucketed shapes, MPS also
uses half precision, and training runs larger batches and models? Torch is
the incumbent. Keeping one backend is simpler; the additional MLX backend
needs a clear advantage where the current training recipe spends its time.
This is an exploratory measurement, with a USD 25 total rental cap. There is
no pre-registered adoption threshold. The recommendation below interprets the completed measurements and their limits.

## Answer

**The answer depends on model shape. MPS beat CPU at w384l8 on the M4 Max,
but lost to CPU at w64l2 on both measured hosts. MLX was fastest in all three
completed host/shape groups.** At w384l8 / 256 streams on the M4 Max,
median whole-update time across three interleaved replays was 703 s on CPU4, 521 s on MPS float32, 478 s on MPS
bfloat16, 442 s on MPS float16 and 402 s on MLX float32. MPS float16 was
about 10% slower than MLX and used 37% less time than CPU; float32 MPS was about
30% slower than MLX. These are one busy host's replay measurements, not a
training-quality result or a crossover curve across model sizes.

| Arm | Whole update, s | Collection policy, s | Learning, s | Learner decisions/s |
|---|---:|---:|---:|---:|
| PyTorch CPU4 float32 | 703.20 | 218.12 | 429.10 | 68.35 |
| MPS float32 | 521.22 | 107.83 | 357.40 | 92.21 |
| MPS bfloat16 | 478.42 | 99.35 | 323.09 | 100.46 |
| MPS float16 + scaler | 442.03 | 99.08 | 286.97 | 108.73 |
| MLX float32 | 402.22 | 84.68 | 261.56 | 119.50 |

Each phase row belongs to its arm's median-total replay. The shared
engine/host component is 55.98 s. All 48,064 sampled collection choices
matched CPU in every replay. The three blocks ran back to back over
6,870.95 s, with endpoint loads 119.12→57.56, 60.47→30.57 and 30.98→47.43;
none required a retry. Individual totals ranged from 691–705 s for CPU,
507–528 s for MPS float32, 451–482 s for bfloat16, 438–473 s for float16,
and 401–413 s for MLX.

**Recommendation:** keep PyTorch as the production backend and retain MLX
as an experimental Apple backend for now. At this measured shape, MLX's
float32 path saves about 23% of whole-update time versus float32 MPS and
about 9% versus float16 MPS. The half path narrows the performance case for
an extra backend, but its gradient differences remain unresolved and the
wider/batched comparison is incomplete. This evidence supports neither
removing MLX nor adopting MPS half precision in production. Defaults and
off-CUDA precision refusals are unchanged.

The M2 w384 series was stopped because five resident arms sustained paging on
its 24 GiB host. Its capture and parity checks remain evidence, but no
complete M2 w384 replay exists. Further scaling needs a comparison design
that fits the host. The full matrix is not complete. The laptop window ended
at 06:17 UTC;
its already-running final retry finished at 06:19:45 UTC. No further
measurements are running on either host.

No independent warmed per-call matrix completed. The collection and learning
numbers above are whole-replay phase totals, including their first-call costs;
they do not establish a batch-size crossover. Generated per-call charts mark
those cells unmeasured. The [M2 table](data/mps_retest/comparisons.md) and
[M4 Max table](data/mps_retest/comparisons-Jacks-MacBook-Pro-2.local.md) retain
all selected repetitions, loads, timing/memory scopes and parity metrics.
Coverage is separate from correctness.

The operator owns AWS allocation, access notes, release and the rental cost
receipt. The M2 became available on 2026-11-13 at 02:56 UTC; its Python 3.12
capture and the laptop's Python 3.13 capture are separate evidence. Both use
Torch 2.15.0 and MLX 0.32.4. The laptop has no load threshold and uses four
CPU threads; M2 uses eight for replay and four for capture. Comparisons
interleave all arms call by call and step by step. Three replays run back to
back on each host, with wall-clock and load endpoints retained and no one-hour
cap. The laptop window prioritised w384 at 256/512/1024 streams, then
w768/w1024 at 256, then smaller widths. Wider three-replay series could not
fit the remaining window. w64l2 completed; w96l4, w160l8, w512l8, w768l8 and
w1024l8 remain unmeasured, as do all 512/1024-stream replay cells. Further
M2 scaling needs review of resident memory and the capture budget.

Larger-stream capture attempts hit the configured 20 GiB process-tree
budget: 512 streams stopped after 143.94 s (load 30.95→37.03), and 1,024
streams after 164.44 s (load 22.51→19.88). Neither reached learning or
produced a replayable capture. These are explicit capture-budget failures,
not backend OOMs or the laptop's hardware memory limit. Their whole-update
replay cells remain unmeasured.

## Named deliverable: full context

`experiments/data/mps_retest/full_context.png` is the shareable counterpart to
a forward-only policy-call benchmark. It shows **seconds per whole update**,
stacked into engine/host work, collection policy calls and learning, with
**collected learner decisions per whole-update wall-second** alongside.

There is one panel per model shape and a group per 256, 512 and 1,024 streams.
Every group includes CPU float32 (four threads on the laptop, eight on M2), MPS float32, each supported MPS half dtype,
and MLX float32. Bars show the median-total replay of three repetitions and
its component times, so their stacks sum to the median total. Each group must
run back to back on the same host, with no elapsed-time cap. Does-not-fit, unsupported and
not-measured cells stay visible. Captions identify hardware, framework
versions, half dtype and parity evidence. The chart uses an opaque light
background for sharing in either light or dark interfaces.

The named chart below retains all requested groups, including unmeasured
cells. M2 and M4 results are separate; their engine components come from
independent captures at different host loads and must not be compared as
hardware speedups. No optional L4 arm ran.

![Full context — Apple M2, partial matrix](data/mps_retest/full_context.png)

![Full context — Apple M4 Max, partial matrix](data/mps_retest/full_context-Jacks-MacBook-Pro-2.local.png)

## Small models change the ordering

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

## Numerical agreement and memory at the measured shape

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

The harness attempts both MPS half dtypes at every selected shape. An unsupported dtype is
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
  high-water mark. Each arm/shape/phase runs in a fresh process. During a comparison all
  arm models remain resident, although only one process computes at a time.
  An allocation failure is a limit of this interleaved setup, not proof that
  the arm cannot fit alone.
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
  and learner replays use four threads on the laptop, eight on M2. Three replays per arm and shape produce
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

The M2 comparison has already shown active paging with all five models
resident: between 04:24:04 and 04:25:37 UTC, swap-ins increased by 361.98 MiB
and swap-outs by 5.52 GiB. Used swap reached 9,033.75 MiB at 04:41:43 UTC.
These observations qualify the shared-memory comparison setup; they do not
establish whether any backend fits or performs well by itself. Do not equate
paging with an allocation failure or label an unattempted arm “does not fit.”
At 05:29 UTC used swap was 10,518.25 MiB, with no complete replay after about
105 minutes. The series was cancelled and workers exited. Pairwise
interleaving would change the all-arm comparison and its single CPU reference;
it was not substituted without reviewing that design change.

<details>
<summary>Method, process and evidence</summary>

**M2 host:** fresh AWS mac2-m2, Apple M2, 24 GiB, Python 3.12, pinned torch
2.15 and the locked MLX version. The operator allocates it through AWS, writes
`~/.manabot-mps-bench/access.md`, and releases it using the 24-hour guard in
`release.sh`. Cite the operator's allocation/release timestamps and cost receipt
(24-hour minimum, about USD 21) against the USD 25 Task cap.

**Laptop host:** Apple M4 Max, 128 GiB, four CPU threads for capture and the
CPU reference. Start captures and comparison blocks regardless of load,
recording the one-minute load at both endpoints. A comparison block is all arms at one width and per-call size/phase, or one
whole-update repetition. Resident isolated processes advance one collection
call or learner step per arm in order. The start and end loads belong to the
entire comparison; there is no mid-block load refusal that would selectively
truncate slower arms.

On either host, if end load exceeds twice start load, repeat the entire replay
once,
without a load threshold. Keep the attempt with the lower arithmetic mean
of endpoint loads; a tie keeps the first. Retain both attempts and select all
arms together. An unavailable repeat leaves coverage incomplete. Per-call blocks on both hosts and laptop captures also retain and repeat
load increases over 2×, keeping unused captures beside the selected one.
Every table row and chart point/bar carries the block's endpoint loads;
whole-update captions also give the reused engine capture's loads. Repeated
blocks are identified. This reduces shared-host drift; it does not make a busy
laptop equivalent to a dedicated host. No artificial load is generated.

Keep each host's captures and ledgers in separate directories. Capture locally
for a same-host whole-update comparison. A replay of another host's capture
retains that capture's engine component: receipts, captions and tables identify
its origin and threads, and the chart title labels a cross-host phase sum.
Missing capture provenance leaves coverage incomplete; conflicting provenance
is refused.

**Commands** inside the frozen source environment with a real Etude launcher
run identity. The operator supplies M2 access when available. These are
measurement commands, not rental commands. Prepare the M2 environment with
`uv sync --locked --python 3.12 --extra play --extra mlx --extra reports
--extra artifacts` and build its matching native extension. Explicit interpreter
selection avoids the checkout's Python 3.13 development pin. Start with a selected width to
establish replay durations before scheduling the full matrix; the default
worker timeout is 900 seconds and a timeout is a failed attempt, not OOM.
Three interleaved replays run back to back on the same host, without a
one-hour admission or reporting limit. Each repetition
interleaves all executable arms call by call and step by step. Their total
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
stream count. A sole host uses `full_context.png`; in a combined report the M2
uses that name and other hosts have a hostname suffix. Host identity remains
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

**Cost:** the operator reports allocation at 2026-11-13T02:56:06Z, with a
24-hour minimum of about USD 21 under the USD 25 cap. Release time is pending
and remains operator-owned. Etude local-job USD 0.00 receipts exclude this
operator-managed rental; they are not the whole experiment cost.

</details>
