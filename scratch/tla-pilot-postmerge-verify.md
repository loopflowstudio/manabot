# TLA pilot permanent-capacity repair — 2026-11-04

## Accepted outcome and current slice

MTG-213 still requires a trained GPU checkpoint, complete paired evaluation and
a qualified report. V9 Run `2171156493ca` failed after five updates / 251,040
transitions at 41 permanents against capacity 40; initialization scored 7–25,
with 256 scheduled games missing. Evidence and confirmed rental teardown are
retained in `6dff2e919` and `experiments/tla-trophy-pilot.md`.

The verification feedback selects a prospective repair and fresh initialization.
Use the existing `max_permanents_per_player=64` configuration; do not widen a
frozen input version, truncate permanents or alter global defaults. Preserve all
134 decks, v7 effects, card128, matched first64 support, recipe and schedule.
This capacity fits the retained failure, not every reachable battlefield.
Jack Heart subsequently accepted isolated resource-limit recovery at .25
(comment ca1660e8); `scratch/resource-limit-direction.md` owns that direction.

## Delete — do not maintain

- Removed `experiments/tla_pilot_v9.py`; the v10 runner, analysis import and
  discovery inventory now own the live path. Historical execution uses
  source `6da60684c`; keep v9 protocol/regime/evidence bytes unchanged.
- Replaced obsolete scratch launch instructions with this current plan. Full
  prior notes remain at `6dff2e919:scratch/tla-pilot-postmerge-verify.md`.
No deletion targets remain; frozen protocols, regimes and evidence stay retained.

## Implementation complete; remaining acceptance

1. Pin v10 regime/protocol with permanent64 and its shapes/hash; link predecessor
   v9. Carry the accepted USD1.25 allocation ceiling and USD10 cumulative cap.
   V9 checkpoints cannot become v10 initialization or fill its evaluation slots.
2. Promote the retained stream43 / seed32624 / 371-action prefix to an offline
   regression fixture. Verify old40 rejection and exact41/pilot64 parity through
   Python, native scalar and vector buffers, preserving all configured keys.
3. Preserve typed native encoder capacity errors through isolated vector writes,
   opt in prospectively and settle them as resource_limit in the existing ledger.
   Keep setup/buffer/provenance failures fatal. Verify both-seat sample isolation,
   replay with the saved input contract, .25/.26 threshold and sharded transport.
4. Verify prospective discovery/admission and reject mutated permanent capacity,
   error policy and old initialization. Rebuild the native extension atomically.
5. Gate owns affected suites; delivery owns publication. Execution owns a final
   source freeze, fresh inventory/budget reconciliation, bounded GPU training,
   checkpoints/evaluation, confirmed teardown and report. These remain acceptance,
   not a reason to mark the Task complete after implementation.

Latest settled estimate USD0.475343889; conservative reservations
USD5.101645926 of USD10. Use USD0.97/hour / USD1.25 whole-rental ceilings after
fresh reconciliation; do not touch another Task's rentals. Paid execution must
retain CUDA admission and initialize afresh, never retry frozen v9 unchanged.

Replay reuses one fixed buffer allocation; the preservation test compares v10
directly with frozen v9. Native code is unchanged by this simplification, so its
prior debug isolation tests and atomic release-extension rebuild remain applicable.

Check: `uv run --extra dev --extra artifacts --extra evaluation pytest tests/training/test_game_errors.py tests/verify/test_native_buffer_capacities.py tests/eval/test_tla_pilot.py tests/eval/test_experiment_files.py -q` — 146 passed; Ruff lint/format and diff checks passed. Gate owns remaining affected suites per docs/development.md; delivery and final-source GPU execution remain unfinished.

## Retained Airship investigation

Exact replay only: player1 cast one Airship at action94, turn20. A qualifying
end step doubled 1→2 at106; turns22/24 doubled 2→4 and4→8; turn28 resolved eight
existing triggers to16. Turn30 sacrificed one, leaving15; ten of its15 triggers
resolved before the capacity failure, leaving25 Airships and five pending
frames. Each copying phase had that controller's sacrifice history and matching
Airship source frames. No same-end-step recursive trigger growth was observed.
The sequence is consistent with the card definition and its existing directed
TLA tests; it does not certify every rules interaction.

Across the retained prefix, observed peaks: 41 permanents/40 old slots (64 new),
75 card rows/128, 25 facts/128, 27 effects/128, 27 members/2048, zero details/512,
18 relations/64 and 26 legal actions/64. All prefix observations encode at64.
More qualifying turns can exceed these finite limits. Capacity recovery, not
indefinite allocation escalation, is the accepted bounded pilot treatment.
The 256/512-row, grouping and GPU minibatch-memory ideas in comments52d6354e
and173f02a6 remain design discussion; no recipe or attention changes are selected.
