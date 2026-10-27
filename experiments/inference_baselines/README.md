# Inference baselines

The experimental runner separates an executable protocol from the report's
retained-evidence manifest. It never rents hardware, downloads missing weights,
evicts a cache, retries a failed cell, or treats fixture output as a trained
baseline. Run commands from the repository root. Relative recipe paths use that
working directory; frozen pins record resolved local paths.

Prepare an admitted `EvaluationRequest` JSON with the existing
`manabot.eval.local.checkpoint_request` API and local checkpoint bytes. A policy
workload references that request, explicit legs (for example `[0]` for one game,
`[0, 1]` for both seats of one deck pair), and optionally a retained Command trace.
The request owns decks, world/ABI, checkpoints, seeds, opponent and game limits.
A cell references an existing `EvalCompute` JSON, or omits `execution` for scalar
CPU play. For example:

```json
{
  "workloads": [{
    "id": "policy-single", "kind": "policy",
    "request": ".runs/baseline/request.json", "legs": [0], "roots": 64
  }],
  "cells": [
    {"id": "scalar-cpu", "workload": "policy-single"},
    {"id": "resident-cpu", "workload": "policy-single", "execution": ".runs/baseline/cpu.json"}
  ],
  "repetitions": 3, "warmup_passes": 2,
  "cell_seconds": 300, "total_seconds": 1800,
  "process_tree_bytes": 8589934592, "sample_device": false,
  "absolute_tolerance": 0.00001, "relative_tolerance": 0.0001
}
```

The compute file declares `max_delay_seconds: 0`, at most four active games and
four batch rows. The runner fixes one Torch thread and disables TF32. Ordinary
and diagnostic cells remain separate, as do instrumentation-on/off cells.
Repeat order rotates the declared cell order; the entire ledger is reserved
before any child starts. Deadline/RSS failure retains progress and stops owned
children, with no retry. A total deadline leaves later cells unattempted.
Collection and retained-output comparison run in a supervised child with the
same absolute cell/invocation deadline and RSS ceiling. Received file digests
survive collection failure. An interrupted inventory is marked incomplete;
the report retains its failed attempt and pinned evidence without deriving game
metrics from unpinned files. All raw progress remains in the attempt directory.

```sh
uv run python -m experiments.inference_baseline freeze --plan recipe.json --out protocol.json
uv run python -m experiments.inference_baseline validate --plan protocol.json
uv run python -m experiments.inference_baseline run --plan protocol.json --out .runs/baseline/games
uv run python -m experiments.report_prototype.build --baseline-manifest .runs/baseline/games/manifest.json --out .runs/baseline/report
```

`freeze` checks local source/native/content bytes, runtime/device facts and every
input dependency. Changing any of these requires a newly frozen protocol. The
report copies the pinned protocol alongside raw evidence. Execution verifies it
again inside the supervised attempt; preparation failures therefore retain their
own elapsed time and failure receipt.

To measure identical policy inputs, add the produced trace's `artifact_path` to
the workload's `trace`, freeze the new recipe, then use `replay` instead of `run`.
The first declared candidate roots are replayed with the admitted observation
configuration. Exact observation arrays, masks and row seeds are retained in
`inputs.npz`; every warmup/measured pass retains logits and selected actions.
Comparison uses legal logits and the recipe tolerances, reports action agreement,
and preserves differing outputs. It makes no cross-device bitwise promise.
When a completed cell differs only in instrumentation, comparison uses the latest
such cell to check identical outputs and report measured-pass overhead. Otherwise
it uses the first completed cell in the same workload. Diagnostic settings must
also match for an instrumentation comparison; rotated repeats and scalar-first
ordering do not change this rule. The receipt names the selected reference.
Diagnostics appear as attempts with evidence, without creating game rates.

An LLM workload uses `kind: "llm"`, an `InferenceConfig` path as `config`, a
`CacheManifest` path as `cache_manifest`, an ordered list of `GameConfig` paths as `games`, and an
ordered `corpus` of retained selection paths for replay. Model/tokenizer snapshots
must already exist locally. Use the `llm` uv extra. A workload is one attempt
with one resident session shared by its sequential games (up to five). The
report retains each game population and charges readiness once for the attempt. A separately authorized LLM protocol can declare a cell ceiling up
to 1,800 seconds, still within its total invocation ceiling. Policy cells remain
capped at 300 seconds. Replay re-encodes the original messages and checks exact
prompt text/token IDs before generation; it never regenerates native text.

Parent whole-attempt time includes verification, imports/readiness, games or
retained-input work, received output collection and child cleanup. It excludes
the final self-describing ledger write. Worker sums, nested admission, host
forward/generation, and optional device intervals stay separate. Warmup is
retained and charged; only the named measured pass excludes it. `generation_host`
includes transfers and decode, and does not establish prefill or decode time.
No local timer establishes provider billing. UUID-tagged 1 Hz device sampling is
optional; unavailable utilities remain explicitly unavailable.

The MTG-188 empirical baseline is still pending pinned trained bytes, CPU/GPU and
LLM single/many-game observations, and artifact publication. New paid execution
has not been authorized. `mtg185-l4.json` remains the unchanged historical corpus
inventory; neither the runner nor this example converts it into new measurements.
