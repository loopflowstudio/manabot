# Batched local-update search

`LocalUpdateTeacher` advances independently seeded branches together and evaluates
each live frontier in one policy/value forward. `Rollout` owns branch execution;
`RolloutReceipt` owns its immutable replay evidence. Finished branches leave the
frontier, and unfinished leaves retain actor-relative value signs. Native
Commands, legal support, canonical hashes and private branch audits remain intact.
See [usage and limits](../../manabot/README.md#local-update-search).

## Measured results — 2026-11-09

Two distinct comparisons support the retained implementation. Both use initialized
policies at openings 42 and 43, depth 20, one CPU thread, and five samples per
side on an Apple M4 Max. They do not measure playing strength.

| Comparison | Opening 42 | Opening 43 |
| --- | --- | --- |
| Before the writer change: sequential → batched, 16 worlds | 3.33854 → 0.97911 s (3.41×) | 4.21951 → 1.22335 s (3.45×) |
| Separate writer A/B: immediate batched baseline → borrowed writer, 16 worlds | 0.97506 → 0.73197 s (24.93% lower) | 1.20364 → 0.91563 s (23.93% lower) |

These are median whole-search latencies. Observed ranges are disjoint within
each comparison. The first comparison loads sequential Python owners from
`841b1c50c` while sharing the current native runtime; it includes the model and
encoding changes, not batching alone. The writer comparison uses alternating
ordinary binaries against the immediate batched baseline `0bbb05512`. Do not
multiply the gains or pool their samples.

The [complete measurement record](local-update.html) preserves all historical
tables, ranges, samples, source/runtime identities, rejected trials, profiler
attribution and artifact locations. Its final post-sync series was interrupted:
only two of five grid invocations completed, one overlapping unrelated tests;
all five eight-world invocations are missing. It supports no new final scaling
estimate. The writer remains kept on its original five-pair result.

Discrete receipt comparisons preserve Commands, hashes, audits, terminal flags,
support and sampling identity. Float predictions use `atol=1e-6, rtol=1e-5`;
the retained complete pre-writer grid's maximum base/value/target differences
are approximately `1.03e-10 / 4.10e-8 / 3.14e-9`. Eleven independent baseline
fixtures preserve exact canonical JSON bytes, SHA-256 and BLAKE3. These checks
do not promise identical trajectories for every policy: a rounding difference
at a sampling threshold can change an action.

The original goal of 8–16 worlds near the old singleton cost is **not
established**. Representative trained and terminal-heavy positions and
MTG-183/w13 admission remain unresolved. Profiles identify Command/audit and
network work as similar remaining costs; snapshots are small. No fixed latency
or strength improvement follows from these opening-position measurements.

## Validation and reproduction

The [local-update validation guide](../development.md#validation-by-change-area)
names the search, receipt and Command suites included in CI core. Model coverage
also checks sparse facts, padding restoration, shared effects, selected input
capacities and gradients after inference. The complete debug native suite and
Engine Budget run in CI; a passing local subset does not replace them.

`uv run --extra dev python scripts/bench_local_update.py --reference 841b1c50c`
runs the bounded 1/4/16-world fixture with separate latency and profiler regions.
Use an ordinary release extension for latency. This creates a new comparison
under the installed world/runtime, not a reproduction of a historical binary.
Retained native comparisons additionally require the original receipt directory
described in the complete record. Games and evaluations for this work run on
the Mac mini or CI.
