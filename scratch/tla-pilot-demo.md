# TLA pilot repair demonstration — 2026-11-04

Task: MTG-213. Step: headless demo, decision pass 1. Reviewed source:
`972d5567343e0226df60e25e6a313815ee02cdeb`.

Decision: **advance to delivery**. The implemented legality, configured input
capacity and explicit first-64 policy support hold in the demonstrated paths.
No concrete implementation gap was exposed. This accepts the enabling change
at this Graph boundary, not the completed pilot or a learning improvement.
`lf help code` places delivery and verification after this demo. Paid execution,
trained checkpoint admission and paired evaluation remain verification work.

## Direction and review basis

The accepted design is [the v8 protocol](../experiments/protocols/tla-trophy-213-v8.md),
its [machine-readable contract](../experiments/protocols/tla-trophy-213-v8.json)
and [regime](../experiments/regimes/tla-trophy-213-v8.json).
[Continuation and validation](tla-pilot-verify.md) records the implementation;
[the pilot report](../experiments/tla-trophy-pilot.md) retains paid attempts,
failures, missing comparisons and costs.

Jack Heart's recorded direction requires native rejection of unavailable
choices (comment `f61beeb7`) and explicitly selects deterministic first 64 legal
actions with logging (comment `3b0f8826`). The complete native/Academy action
set, shared effects, 128-card capacity, 134-deck population and USD1/USD10
ceilings stay intact. Those are prior human decisions. No person attended this
headless review; none of the observations below is new human feedback or approval.
The advance decision is the reviewing agent's judgment under that direction.

## Demonstrated behavior

Fresh focused checks passed **11 tests in 2.77 seconds**. The retained seed-32172,
stream-103 capture replays its 76-command prefix through the repaired scalar
and vector runtime. The historical illegal X0..255 offers no longer appear;
submission 255 rejects without advancing the decision or mutating buffers.
Native scalar, vector and Python encodings agree, including at 64 action slots.
Insufficient permanent capacity still rejects rather than silently truncating.

A controlled legal 71-action position retains native offers 0–63, leaves all
71 offers at the authority, and executes retained row 63 through its original
Command. The checks cover unchanged small action sets, one masked policy
distribution, scalar/sharded collection and replay-linked diagnostic counts.
The separate debug publication receipt verifies payable X0..5 and unchanged-state
rejection for ordinary, airbend and any-type-mana casting; this demo did not
rerun the full debug workspace suite.

A fresh bounded CPU demonstration called `tla_pilot_v8.frozen()` without a
compatibility fixture, admitted **all 134 decks**, instantiated the exact v8
model and observation configuration, and used `SeatRoutedCollector` with the
frozen independent per-seat deck source. Two streams returned **717 decisions**;
the normal complete-game collector drained the games after its small decision
floor. Every selected action was valid, all padded slots had zero probability,
each distribution summed to one, and every configured observation block reached
the batch. Legal support ranged from 2 to 11 actions. The close receipt records
exactly 717 consumed decisions and zero overflows across 11 decision kinds.
This small sample establishes neither population-wide capacity nor rare-cutoff frequency.

The scalar evaluation adapter used frozen pair 104/81 and deal 21391000 with
the same observation configuration. Four deliberate decisions exposed 4, 2, 5
and 3 legal actions, preserved their masks, and advanced without truncation.
Its close receipt records four decisions and zero overflows. This is adapter
proof, not a completed paired evaluator result. No optimizer update, checkpoint,
scored evaluation, provider query, rental or expense was created in this demo.

Local supporting evidence:

- [Bounded demonstration script](../.runs/mtg213/demo-20261104/configured_path.py)
  and [configured-path receipt](../.runs/mtg213/demo-20261104/configured-path.json).
- [Focused behavior checks](../.runs/mtg213/demo-20261104/behavior-checks.log).
- [Retained debug validation](../.runs/mtg213/publish-cargo-test.log), including
  the direct X rejection and native vector prefix witnesses.
- [Historical capture](../tests/fixtures/tla-pilot-action-capacity.jsonl) and
  [historical observation](../tests/fixtures/tla-pilot-action-observation.json).

## Progress, remaining evidence and next action

The preceding pass intended to remove the observed illegal-action failure,
honor supplied tensor capacities, and implement the chosen cutoff consistently.
The fresh replay and configured collection add execution evidence for that
intent. Earlier failed GPU attempts are preserved, not reclassified as successes.

V5 remains the last confirmed GPU learner execution: three updates, 585 optimizer
steps and 149,260 learner transitions, followed by failure before trained export.
Its initialization-only evaluation has 32 valid games and 256 missing scheduled
games. V8 Run `f47be2d3c0d8`, bound to older source `900c20750`, was never submitted.
The retained 19:11–19:12 UTC quotes refused the RTX5090 price ceiling and L4/RTX3090
availability. Those dated receipts are not current supply claims.

No redesign or further product judgment is required. Continue delivery checks
and shipping, then freeze a successor Run from the delivered source, reconcile
inventory/bills and obtain a fresh compatible quote within existing limits.
Verification must retain actual learner progress, an admitted trained checkpoint,
teardown/whole-rental costs, complete paired comparisons and qualified
Elemental-relative analysis. Do not launch the old prepared Run as evidence for
the later collector changes or fill historical missing slots with successor results.
An empirical pilot success claim remains unavailable until that proof exists.

All demo-owned processes exited. No interactive HTML was generated: this was
a headless runtime review with no UI change or rendering surface.
