# TLA resource-recovery demo — 2026-11-04

MTG-213, headless demo decision pass 1, reviewed commit
`4fb8109628a7597cedebe35a4296d5f1aea7cdbf` (PR #426).

**Decision: advance this repair to delivery. MTG-213 remains incomplete.**
The configured v10 admission and bounded replay/collector demonstration passed.
No concrete implementation gap emerged that warrants returning to implement.
The current `code` Graph places delivery and verification after demo; advancing
this step does not establish a trained endpoint or approve Task completion.

## Accepted design and review basis

The current design is the [v10 protocol](../experiments/protocols/tla-trophy-213-v10.md),
with [accepted resource-limit direction](resource-limit-direction.md) and
[implementation/validation record](tla-pilot-postmerge-verify.md).
The [pilot report](../experiments/tla-trophy-pilot.md) retains the failed v9
execution, cost accounting and incomplete evaluation.

Jack Heart's recorded direction was “can we just drop the game and not
everything else” and a permissive error rate “like 25%.” His later direction
keeps permanent64 and game-local recovery; larger capacities and representation
changes remain exploratory. These are prior human statements, not feedback
from this demo. No person attended or accepted behavior during this headless run.
Acting on the supplied proxy-review instruction, the agent judges the observed
behavior consistent with those explicit choices. No new design change is selected.

## Demonstrated behavior

Real `experiments.tla_pilot_v10.frozen()` admission passed against the installed
native runtime: w17, 134 decks, input v7, card128, permanent64, first64 legal
actions and matched heuristic prefix support. The configured resource policy is
`recover_capacity=true`, threshold .25, window/warmup 100/100 and
`max_game_actions=40000`. Milestones remain 25/50/75; allocation and cumulative
ceilings remain USD1.25 and USD10. This checks the actual protocol/regime hashes,
runtime identity and tensor shapes; it is not a training launch.

The bounded demonstration ran:

```sh
uv run --extra dev --extra artifacts pytest tests/verify/test_native_buffer_capacities.py -k retained_airship tests/training/test_game_errors.py -k 'retained_airship or resource_limit_receipt or strict_threshold_window or quarantine_refills or later_crash_preserves' -q
```

Observed result: **16 passed, 78 deselected in 3.05s**, exit 0.
The session log is `/tmp/mtg213-demo-proof.log`; the durable reproducible evidence
is the command above and these repository fixtures:

- [Retained Airship replay](../tests/fixtures/tla-pilot-permanent-capacity.jsonl),
  stream43/seed32624/371 actions: the old40 allocation rejects 41 permanents;
  exact41 and pilot64 preserve every permanent with Python/native scalar/vector
  equality. This is an actual retained gameplay prefix, not a random stress run.
- [Native buffer demonstration](../tests/verify/test_native_buffer_capacities.py):
  the retained overflow through single-worker and sharded paths identifies only
  the affected stream as `resource_limit`. The healthy stream matches its
  independently stepped control; no terminal flag or reward is invented.
- [Collector and ledger demonstration](../tests/training/test_game_errors.py):
  controlled resource failures remove the affected game's samples from both
  seats, preserve prior complete games and healthy-stream data, and leave a
  batch that completes an optimizer update. Replay receipts reproduce the
  retained encoding boundary. Restart preserves the ledger; 25/100 continues
  and 26/100 stops after the configured warmup. Controlled collector fixtures
  complement the native replay; they are not a full GPU pilot.

Prior publication evidence records 320 affected Python passes, 969 debug-native
passes/two ignored, formatting and Clippy success. This demo did not repeat
those full suites. It added a fresh, bounded observation of the behavior through
the configured admission, native replay and real collector/ledger APIs.

## Progress and remaining proof

The preceding implementation pass intended to replace v9's whole-run capacity
failure with a prospective capacity allocation and isolated game recovery.
Observed results support that repair. V9 itself remains failed after five GPU
updates, with no trained export and 256 missing scheduled games; none of this
demo fills or rewrites those historical slots. V10 remains unexecuted.

No unresolved product judgment blocks this slice. The remaining uncertainty is
whether the bounded GPU pilot exports a usable trained checkpoint and completes
its paired schedule. Finite tables can still overflow, and discarding affected
games introduces selection bias. The .25 policy is an accepted engineering
tolerance, not evidence of a low actual failure rate or improved playing strength.

Next action: deliver PR #426 through the existing Graph, then execute verification
autonomously under standing authorization. Freeze the final v10 source once;
reconcile fresh inventory, bills and reservations; retain CUDA admission;
launch fresh initialization within USD0.97/hour, USD1.25 whole-rental and USD10
cumulative ceilings. Retain actual learner progress, trained exports, paired
evaluation including failed/missing slots, overflow frequency/decks/positions,
whole-rental accounting and confirmed teardown. Do not retry frozen v9, reuse
its initialization, replay completed implementation, or mark MTG-213 complete
on merge alone. No provider allocation or background process was started here.

This headless review has no interactive page or absent-user checkpoint.
`scratch/tla-resource-recovery-demo.md` is the review record.
