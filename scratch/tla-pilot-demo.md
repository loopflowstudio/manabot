# TLA pilot capacity repair — headless demo

Reviewed 2026-11-03 America/Los_Angeles (2026-11-04 00:41 UTC), MTG-213,
code Graph demo, decision pass 1. Reviewed source:
`feab628225e6225292f3a9ad3782ab5ca3800438`,
[PR #416](https://github.com/loopflowstudio/manabot/pull/416).

## Decision and scope

**Advance to gate and delivery of the enabling repair. The training Task is
not complete.** The preceding pass intended to repair the reachable 140-fact
overflow prospectively, preserve frozen v5 and prior evidence, select v6/256,
and prepare the same trophy population and evaluator on the merged runtime.
This demo directly confirms that progress. No concrete implementation defect
in that repair was found that warrants replaying the implementation loop.

This is an autonomous decision under the caller's explicit instruction to
stand in at demo. Jack Heart's recorded direction authorizes the immediate
capacity hotfix and shipping verified enabling changes; it also requires
actual GPU training and evaluation. No person observed this demo or supplied
new feedback, and no human acceptance of the unfinished pilot is implied.
Advancing here lets gate/landing proceed; the remaining paid execution cannot
be omitted at final verification or Task completion.

Current design: [pilot account](mtg-213.md),
[v3 protocol](../experiments/protocols/tla-trophy-213-v3.md),
[frozen v3 inputs](../experiments/protocols/tla-trophy-213-v3.json),
[active runner](../experiments/tla_pilot_v3.py).
The [architecture walkthrough](../docs/observation-architecture.html) retains
its historical measurement snapshot; this demo makes no new visual claim.

## Actions and observed results

1. Loaded the real runner's `frozen()` configuration, then replayed the retained
   failed stream using its actual decks, seed 31482, all 87 actions and native
   trivial-decision skipping. Used the installed merged-runtime extension,
   explicitly accepting changed source/runtime for this repair proof. Both
   scalar and vector paths matched the original pre-command frame.
2. Encoded the resulting state through the configured v6/256 native scalar,
   vector-buffer and Python paths. Every observation field agreed exactly;
   all **140 facts** remained valid. Native and Python v5 still raised
   `observation capacity exceeded: facts 140 > 128`. Nothing was truncated,
   deduplicated or relabeled as successful historical training.
3. Ran ten focused contract checks: capacity/provenance/overflow at v5/128 and
   v6/140, 256 and 512; native vector buffers at 128, 140, 256 and 512; the
   frozen 134-entry independent population and learning recipe; and Etude's
   actual runpy discovery of the importable evaluation callback. **10 passed
   in 1.54 seconds.** The 256 boundary fixture accepts 256 distinct rows and
   rejects 258. These controlled checks are not training or performance runs.
4. Tried the authenticated configured launcher inventory through nested Doppler
   `manabot/prd` then `etude/prd`, using `uv run etude runs`. It exited 1:
   `RunAttempt.cause` rejected `input_undelivered`, expecting one of
   `capacity_refused`, `setup_failed`, `machine_lost_before_start`,
   `machine_lost`, or `create_unconfirmed`. This is a fresh client/record
   compatibility finding, not proof of missing credentials or no live jobs.
   The empty redirected stdout is not an empty inventory. No rental or job
   was started, and no cost reconciliation was claimed.

Fresh proof:
[replay receipt](../.runs/mtg213/demo-capacity-replay.json),
[bounded replay script](../.runs/mtg213/demo_capacity.py),
[contract-check output](../.runs/mtg213/demo-contract-checks.log).
The script was run under a 45-second subprocess deadline and completed
successfully. It hashes the single-stream input as
`6a35316ae7c1e38a3b305410be62595d538073e1ff77168ccc98ab1da65a1fd0`;
that is the input-file hash, not the larger failure-bundle hash in the earlier
v3 receipt. Its runtime digest is
`dcb9d47c674a657176c2a9d56e016288d1e175729aa09cd60f0dc333633330ed`.
These local receipts supplement the
[retained empirical report](../experiments/tla-trophy-pilot.md) and
[original stream](../experiments/data/mtg213/gpu-corrected-20261103/capacity-stream.jsonl).

## Remaining work and interpretation

No new design change is proposed. The accepted pilot stays v6/256, with
overflow fatal and the original 134 entries/weights, independent seat draws,
eight evaluation pairs/four legs and missing-result accounting preserved.
Generic v6 remains 512. Broader representation/history/memory design remains
separate. Success at 140 or 256 facts does not establish universal capacity.

Gate must evaluate the published revision and required checks before landing.
Then continue the existing Task's paid pilot: resolve launcher inventory
compatibility through the existing Etude/MTG-239 owner or supported compatible
client, obtain authoritative live-job and billing state, reconcile both prior
$0.7375 reservations, and freeze the exact verified v3 source. Use a fresh
initialization/baseline and the existing bounded allocation under the cumulative
$10 cap; a new protocol does not replenish it. Do not create a competing job
while inventory is unknown.

Observe GPU learner updates before publishing the requested training milestone;
retain an admitted trained checkpoint and run the frozen evaluation/analysis.
Report denominators, uncertainty and Elemental-relative pace honestly. The
historical 32-game initialization evaluation and separate CPU update remain
evidence of their own runs only. There is still no demonstrated GPU learning
gain in the evidence reviewed here. The unavailable fresh inventory leaves
current external execution status unknown, not a new empty-plan conclusion.

No missing human choice prevents this repair from advancing. The launcher
compatibility issue and paid execution are concrete continuation work, not
grounds to rewrite the accepted capacity design. This demo started no servers,
watchers or paid work; all subprocesses it started exited. No HTML review page
was generated because this was a headless non-UI demonstration.
