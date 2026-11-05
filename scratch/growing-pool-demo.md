# Reference-league demo — 2026-11-05

MTG-245, demo decision pass 1. Reviewed checkout
`a1927892def0b6654b727e7c40f969937886c9cf` and the existing uncommitted shipping
direction in [the accepted design](explore-design.md#shipping-direction--2026-11-05).
Decision: **advance** to the existing Graph's remaining delivery steps.

This was an autonomous headless review. Jack Heart did not observe these new
demonstrations or give new feedback during this run. The standing-in decision
uses the recorded selection of A, required exposure capture, accepted workflow
boundaries, and the saved “lg2m ship it” direction. It does not invent scientific
approval or treat absence as consent. The report-headline refinement remains a
proposal, explicitly not a new shipping requirement.

## Experience and observed results

The experience is authoring a frozen reference-league binding and reading its
retained training/evaluation evidence through the actual offline report CLI.
The [protocol](../experiments/protocols/growing-pool-245-v1.md) owns its usage;
[work boundaries](../docs/workflows.md) own the associated workflow.

The preceding review requested complete-positive, complete-inconclusive and
missing-receipt demonstrations. This pass supplies new evidence for all three.
The [demo harness](test_reference_league_demo.py) adapts the existing sealed-league
fixture, changing only the synthetic endpoint improvement between positive and
flat cases. It creates synthetic Etude declarations, metric logs, exports, sealed
evaluation blobs and digest-named exposure receipts. Each report runs
`experiments.growing_pool_analysis` in a fresh subprocess using the pinned venv
interpreter; the reader receives no compatibility monkeypatch. The controlled
world fixture applies only to synthetic store construction, never to native
execution. Product code and frozen manifests were not changed.

| Retained synthetic input | `complete` | `modest_check` | `training_exposure_complete` | Observation |
| --- | --- | --- | --- | --- |
| Positive scores, all receipts | true | true | true | All 800 evaluation rows and 256 terminal captures per seed reconcile |
| Flat scores, all receipts | true | false | true | Complete evidence can report no observed improvement; no missing items |
| Positive scores, exposure omitted | false | true | false | Favorable evaluation does not substitute for training exposure |
| Flat scores, exposure omitted | false | false | false | Both the missing evidence and unmet learning criterion remain visible |

All cases retain the full 800-row/200-match denominator and healthy synthetic
optimizer measurements. Positive anchor mean gains are 0.0625 and 0.125; both
bootstrap lower bounds are zero. Even that positive modest criterion is not
strong evidence of general learning. Flat gains are exactly zero. In both
missing-exposure cases, each seed says “Expected terminal training captures are
missing.” Missing capture has no per-card rows; it is not reported as observed
zero. Captured opportunities, choices and completed frames remain separate.

These are authored fixture outcomes, **not 800 played games or 512 trained games**.
Etude's fixture-sealing progress uses the word “played”; its callback here returns
synthetic rows. No training, inference, complete native game, campaign, resource
allocation or spending occurred. Training receipt replay status remains
“inputs retained; not replay-verified.” No measured learning or throughput claim
follows from this demonstration.

## Supporting checks

- `uv run --extra dev --extra artifacts --extra evaluation pytest scratch/test_reference_league_demo.py -q -s`
  passed: 2 cases in 27.16 seconds, covering four fresh CLI reports.
- `uv run --extra dev --extra artifacts --extra evaluation pytest tests/training/test_growing_pool_protocol.py tests/training/test_growing_pool_analysis.py tests/training/test_training_exposure.py -q`
  passed: 56 checks in 57.96 seconds. This includes actual Etude discovery and
  two-binding routing, crossed identity rejection, absent schedule preservation,
  zero versus missing exposure, interrupted retention, and the bounded native
  one-land-decision capture round trip. It runs no scientific campaign.
- `uv run --extra artifacts --extra evaluation python -m experiments.growing_pool_protocol check`
  exited 1 with `runtime world or engine source differs from frozen protocol`.
  This is the expected protection of v1's frozen source identity, not successful
  runtime admission. The integrated first pilot still needs Task 04's new freeze.
- `uv run python scripts/check_loopflow_workflows.py` passed all seven graphs:
  all skills resolve, three Coverage loops remain independently revisable, and
  optional experiment software preserves the scientific completion boundary.
- The fresh `lf wave list --json` read retained Experiments identity
  `54bc9162-c883-4574-bab1-af589526c1ad`, Coverage, Engine performance,
  Infrastructure and separate Academy. Study remains abandoned; no replacement
  Training or Eval is active. No migration was replayed. The supported migration
  provenance remains in [the dated receipt](../docs/workflows/2026-11-04-migration.json).
- The Task read retained MTG-245, its checkout, code Graph and PR #428. No other
  Task was executed or moved. Prior full debug/Clippy and wider Python evidence
  remains in [the implementation handoff](questions.md#publication-verification--2026-11-04);
  this demo does not claim to repeat those suites.

## Evidence and reproduction

[The compact result record](growing-pool-demo-results.json) preserves report
digests, exact counts, gains and issues. Full synthetic stores, receipts, reports
and subprocess logs are under:

`/Users/jack/src/manabot.mtg-245-design-training-and/.runs/growing-pool-demo-20261105/`

Each of `positive/` and `flat/` has `store/`, `exposure/`, `complete.json`,
`missing-exposure.json`, corresponding `.log` files and `summary.json`. These
local fixture artifacts are outside `experiments/data/` and production stores.
The harness refuses to overwrite an existing fixture root. Re-read the positive
case without constructing any data:

```sh
ETUDE_HOME="$PWD/.runs/growing-pool-demo-20261105/positive/reader-home" uv run --extra artifacts --extra evaluation python -m experiments.growing_pool_analysis --source .runs/growing-pool-demo-20261105/positive/store --training-evidence .runs/growing-pool-demo-20261105/positive/exposure --out .runs/growing-pool-demo-20261105/positive/reread.json
```

Omit `--training-evidence` to reproduce the missing-receipt result; use `flat`
paths for the no-improvement case. No browser/app, private Machine, background
server or watcher was started. All test and report subprocesses exited.

## Decision and remaining work

Advance because the demonstrated behavior satisfies the accepted distinction
between software readiness, evidence completeness and measured learning, while
the accepted reference league and workflow structure remain intact. The new
proof closes the preceding review's evidence gap. There is no concrete repair
gap requiring another implementation loop and no unresolved product question
for a person at this boundary.

No design change was agreed during this headless pass. Leading a future
human-facing report with evidence completeness remains the earlier presentation
proposal. The existing CLI writes the independent JSON fields correctly; its
output is not an implemented prose headline. This distinction is preserved,
rather than claiming the proposal shipped.

Continue the current Graph through gate, delivery and post-merge verification.
Do not treat this demo as merge or Task completion. Task 04 still owns the
integrated prospective binding, actual decks/probes, accepted numeric settings,
execution authorization and measured pilot. Task 08 retains Validation B;
later PR #421 integration and deferred backlog mappings retain their existing
owners and limits. None requires changing historical evidence or launching work
as a demo substitute.
