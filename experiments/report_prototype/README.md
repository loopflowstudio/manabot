# Evaluation performance viewer

This directory retains the reviewed evaluation activity overview. Training
reports use Etude experiment files; the former training-data builder and
experiment definitions are removed.

```sh
uv run python -m experiments.report_prototype.build \
  --eval-receipt .runs/eval/panel --out .runs/eval/report
```

Repeat `--eval-receipt` to compare retained attempts. The builder executes no
games and copies linked evidence into the output directory. Open `report.html`.
Inference, Game engine, Evidence and replay, and Other and unmeasured are the
default activity groups; click for measured phases. Each resource keeps its own
clock, and incomplete coverage never becomes zero.

Etude 0.2.5's report panels read training metrics and evaluation score tables;
they do not yet accept this resource-specific evaluation performance contract.
The viewer remains until that integration preserves clocks, coverage, activity
drill-down, shared allocation accounting, and diagnostic evidence links.

Check: `./scripts/verify-eval-performance-report`.

Retained LLM games use the same Perf view, without synthetic evaluation scores:

```sh
uv run python -m experiments.report_prototype.build \
  --llm-receipt .runs/llm/l4-result --out .runs/inference-baseline/report
```

Combine `--llm-receipt` and `--eval-receipt` for a mixed report. Each LLM attempt
keeps its own successful/failed selection and generation-attempt distributions.
Floor-index p50/p95 are labeled separately from an original historical median.
Preparation, worker readiness and game wall retain their original boundaries;
selection time overlaps game wall. Unrecorded warmup, queue, tokenization,
prefill/decode, inclusive billing and whole-attempt time remain unavailable.
Memory distinguishes the process allocator from sampled whole-device occupancy.

`--baseline-manifest experiments/inference_baselines/mtg185-l4.json` verifies the
pinned historical corpus and includes its available receipts. Paths resolve
relative to the manifest. Absent receipt directories and unattempted cells stay
visible; changed bytes, missing files within a present pinned directory, malformed
records and duplicate imports fail validation. The historical manifest binds
1,708 files and all 160 selections in ordinal order; it contains no raw prompts,
Commands or model weights. Raw historical evidence remains outside source control.
A manifest validates retained evidence; it does not admit a current-world run.

Build with `--baseline-manifest` to inspect every declared attempt and its linked
measurements. Each attempt identifies its cohort and repeat, declared outcome,
evidence availability and observed outcome where known. Missing evidence keeps
the declaration visible without proving success. Restore the exact pinned files
and rebuild to recover the same attempt's measurements; rebuilding never executes
or retries inference. Contradictory terminal declarations and receipts fail with
the attempt ID and source. Correct the declaration against authoritative receipts;
do not edit raw evidence to fit a declaration. Direct receipt inputs still work
without a plan. Game completion does not certify replay coverage; a policy panel
with failed games has an observed failure even when other games completed.

The builder writes `data.json` for headless inspection, alongside `data.js` and
the existing page. Linked evidence is copied byte-for-byte with SHA-256 and byte
counts in each corpus/inventory file. Keep the output outside input directories,
including declared evidence directories that are currently absent.
Report generation runs no model, native replay, download or experiment. The
LLM-only path imports no training, simulation, Torch or Transformers runtime.
The report cannot establish missing CPU/GPU or many-game baseline coverage.

A baseline binds each attempt's receipts and its shared allocation evidence.
The optional v1 `allocations` list contains one record per allocation:
`identity`, `root`, and `files` entries with `path`, `sha256`, and `bytes`.
The root resolves relative to the manifest; file paths resolve relative to that
root. Include `serving.json`, `execution.json`, `allocation.json`, and every
referenced diagnostic trace (inventory paths may be absolute or relative to the
allocation root; paths inside historical serving receipts retain their original
working-directory interpretation). The attempt's recorded `serving_path` links it to that
allocation; the serving identity must agree with the declaration.

Shared charges appear once across cells. Missing or unpinned shared files leave
local observations visible and only the dependent measurements unavailable,
with reasons. Restore exact pinned bytes and rebuild to recover the same
attempt/allocation identities. Changed pinned bytes fail validation, including
at copy time. Legacy policy manifests cannot certify external measurements
without these pins; historical v1 LLM manifests remain readable unchanged.
Direct inputs acquire digest inventories during reading. The report copies and
links serving, execution, allocation, and diagnostic evidence so its cost remains
explainable after original paths disappear. Rebuilding never settles a bill.

New policy performance receipts distinguish submission-to-service transport from
receipt-to-dispatch queue wait. Dispatch ends before checkpoint loading. Historical
`batch_queue` receipts retain their original submission-through-readiness meaning.
Queue and request totals overlap across games and never partition serving wall
time. Service startup includes spawned imports and IPC, but excludes model
admission; admission round trips can overlap startup. Both remain inside the
allocation elapsed denominator, with cleanup retained separately.

Latency populations distinguish service reply send, supervisor reply validation,
and full game-worker round trip. Each has separate completed/failed samples and
floor-index quantiles; worker distributions remain per game. The first observed
completion or failure remains in its population, not an excluded warmup. Counters
and maxima cover all observed calls even after the sample cap; disabled timing
retains counts without reporting measured latencies. RSS peaks are sampled after
admission and batches; CUDA allocated and reserved peaks describe the serving process allocator,
not whole-device occupancy. These instruments do not establish empirical CPU/GPU
comparisons or a complete baseline protocol.

Bounded new measurements use the [inference baseline runner](../inference_baselines/README.md).
Its manifest links the frozen protocol, whole-attempt boundaries and child cleanup,
including failures before a game exists. Retained-input diagnostics keep their
byte-pinned witnesses without creating evaluation rows or game throughput.

Reasoning-enabled Qwen versus checkpoint games use this same reader and Perf
page. A `qwen-v-checkpoint` summary adds an outcome card, exclusive game-host
time shares and model-specific latency boundaries. Policy encoding and inference,
native commands, rendering, replay, writes and unattributed time remain separate.
Nested Qwen worker phases are shown below their host wait, never added to it.
A sequence's `games.json` includes the entire plan so absent games stay visible;
only games with final timing summaries contribute to the exclusive breakdown.
Raw receipts, latency definitions and evidence inventories are under disclosures.

The MTG-188 campaign protocol is `experiments/protocols/qwen-mtg183-188.md`.
Download its output metadata and per-game archives, extract each archive beside
`games.json`, then pass that directory as `--llm-receipt`. Keep `results.json`
and the original archives as evidence. The builder does not extract archives,
fetch model bytes, run a game or infer settled billing from elapsed time.
