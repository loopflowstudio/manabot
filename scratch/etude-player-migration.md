# Etude evaluation names — Task 9da33d

## Accepted direction

Jack Heart, 2026-10-27: migrate all Etude consumers to Player, Model, Code,
Inference and seat, preserving stored results, match keys, frozen records and
existing dependency changes. The initial SDK selection was 0.2.7. The task
operator, under Jack Heart's autonomous-execution instruction, selected **0.2.8**
with the upstream historical-reader fix on 2026-10-27. No Manabot player/seat reader shim.
Jack Heart's subsequent Task direction on 2026-10-27 confirms continuation
against published 0.2.8 through validation, publication and autonomous landing.
Publication, gate and authored reviews remain separate steps.

The original baseline was `d7b98a487`; its OpenRouter dependency additions are
preserved. The current Task base is `e742e8bff`, merged at `90e5640e6`.
No Wave is selected. This is one complete migration, not staged API support.

That upstream merge adds controller-selected trigger ordering and world w14,
including changed observation and inference sources. Historical SDK fixtures
remain valid schema/identity evidence: their checkpoint references are synthetic
and resume executes no native games. They do not admit historical checkpoints
to w14 or establish unchanged inference behavior across the world change.
The existing world/checkpoint admission boundary remains intact; frozen results
and source provenance retain their original meaning.

## Historical counterexample and resolution

Elemental and MTG-186 tables originally contain `player` (Etude participant)
and `seat` (native game seat). SDK 0.2.7 only renamed `player` when `seat` was
absent, silently reversing challenger/heuristic attribution when they occupied
opposite native seats. The two-row reproduction remains in
`scratch/etude-seat-collision.py`; the incomplete prior edit remains in
`scratch/etude-migration-draft.patch`. Neither is the final implementation.

The operator selected the upstream fix: historical `player` becomes `seat`,
and historical native `seat` becomes `game_seat`. New tables cannot emit
`player`. Upstream fix `276ced9` was merged as
<https://github.com/loopflowstudio/etude/pull/61> on 2026-10-27.

## Implemented contract

`Player = Model | Code` describes entrants; `Inference` describes how a
checkpoint becomes a model. `Match.players` owns participant order. Table
`seat` joins that order to a player identity; `game_seat` describes native
position. Neither display names nor native position replace identity.

The working diff migrates:

- Etude Agent/AgentKey and match.agent helpers → Model/Code and string keys;
  Player for the union. Experiment helpers are checkpoint_model/heuristic_code.
- Placeholder Model(code=...) and Checkpoint.agent → Inference(source=...)
  and Checkpoint.model; real models retain checkpoint references and inference.
- Match.agents and emitted `player` → Match.players and participant `seat`.
- SQL agents/agent, seats.player and groups.player → players/player and seat.
- Native-seat overloads in Elemental/MTG-186 producers, collectors and ratings →
  game_seat; card/benchmark tables also emit explicit native seats.

Card audits reconstruct absent
`game_seat` from the retained `on_play` fact, including nulls in mixed runs,
while checking explicit new values. This is optional derived card metadata;
the SDK still owns all historical participant-column conversion. Audit rows
sort by their per-game keys (seat, plus card for counts), then compare complete
rows, preserving duplicate-row rejection independently of dictionary insertion
order. Cached Inference and the scheduled Code own source versions; entrants
use direct Model/Code branches and query fixtures use the reducer's fields.

Required behavior preserved: checkpoint artifact validation, local simulation
import isolation, source/config identity, card metrics and evidence publication,
heuristic source filtering including SDK 0.2 digest identities, native seats,
and all frozen experiment records. The archived 0.2.5 fixture writer deliberately
retains its old dependency/API; it is never imported by production or tests.

## Evidence and remaining verification — 2026-10-27

The report recognizes old digest-keyed heuristic players by code kind and exact
identity/source, because the SDK retains their identity as the code name.

`tests/fixtures/etude_sdk_025/` was written by actual PyPI SDK 0.2.5. It contains
synthetic scalar outcomes for production Elemental full/subset/pair schedules,
one complete MTG-186 match, and two benchmark deals. No native games or training
were used to generate it. Current adapter schedules have the same six match
keys; resume plays zero new matches, retains match/table bytes, attributes wins
to the correct participant in both native assignments, and feeds the real
Elemental rating and MTG-186 summary readers. Additional coverage checks the
older 0.2 heuristic fallback and union of historical/new table schemas.

`uv.lock` pins the published PyPI 0.2.8 wheel
and source archive. Locked sync built Manabot and replaced the temporary
`276ced9` source installation with published `etude-sdk==0.2.8`; installed
package metadata confirms that version. All four dependency declarations agree.
Production consumers and focused tests are migrated; current source has no
old Etude API imports. No production implementation remains unfinished.

The prior CI-equivalent collection of 165 tests includes the migration test
module; subsequent simplification added no test modules. No configured Python
type checker was found.
Gate owns affected-suite acceptance, tracking/report checks and isolated remote
requirements validation; publication and landing remain with the authored
delivery steps. The published-wheel proof also passed the isolated local CLI
boundary check. Lock checking and locked sync passed before simplification;
dependency declarations and the lock are unchanged by this reconciliation.
No further product decision is unresolved.

Prior focused validation recorded 50 passing cases and a native-resume failure
after an edit changed the source fingerprint during execution. With source
held stable, the native-resume test and historical-card-position audit passed
all four cases. This is a successful focused rerun, not an uninterrupted green
suite. Changed-file Ruff, formatting and compilation passed. These results are
reused; subsequent edits only clarify guidance in `AGENTS.md`,
`docs/card-evaluation.md` and this plan.

Check: `git diff --check` — passed; behavioral checks reused as described above, broader acceptance remains with gate.

## Concept review — 2026-10-27

Outcome: **nothing further to change; no open product choice**. The earlier
guidance correction is implemented: ordinary SDK 0.2.8 delivery carries card
evidence and results together, and `AGENTS.md` now agrees with the usage docs.
This follows Jack Heart's accepted migration direction; it is not a new decision
attributed to him. The current review confirms that correction against the
working diff and installed published SDK. No next implement direction remains.

The concrete interaction is unchanged: select a checkpoint and opponent,
evaluate a run, resume it without replaying completed matches, inspect results
by participant and play/draw, then sync and audit the retained run. `Inference`
is the reusable checkpoint execution recipe; binding a checkpoint produces a
`Model`, while `Code` is an entrant without a checkpoint. `Player` is their
union, not another persisted lifecycle. Manabot owns admission and game
evidence; Etude owns match identity, historical schema reads and delivery.
The user needs no stored-data migration or separate card-publication step.

The opposing-seat historical fixture remains the decisive counterexample:
participant 0 may play from native seat 1 and still owns its score. Collapsing
these concepts would reintroduce incorrect attribution. The unrelated proposed
log-import experience remains unapproved.

### Delivery guidance correction

`AGENTS.md` previously said built-in sync omitted arbitrary evidence and required a card
publication operation. Current `docs/card-evaluation.md`, `retain_evidence`, and
the delivery/audit test instead use ordinary sync. Installed published 0.2.8's
`etude/carriage.py` lists run files recursively and treats `artifacts/` as
immutable; `etude/run/delivery.py` owns transfer markers and retries. There is
no separate card-publication API in the current adapter. Adding one would
duplicate SDK delivery ownership solely to satisfy outdated guidance.

### Delete — do not maintain

- Removed the **Etude card evaluation** paragraph's obsolete sync
  limitation and separate card-publication requirement from `AGENTS.md`.
- Preserved immutable evidence with its run, historical paths and bytes, the
  prohibition on publishing indexes alone or using the checkpoint namespace,
  and the existing seat guidance. No exclusive tests or fixtures were removed.
  No deletion targets remain.

Applied the reviewed replacement: SDK 0.2.8 ordinary delivery carries
content-addressed `artifacts/evidence/` bytes with tables and match records.
The documented sync command uses the repository's `uv run` environment.
No additional publication API or user-facing step was introduced.

### Failure, recovery and proof boundary

On an interrupted transfer, keep the source run and retry ordinary SDK sync;
its destination markers track successful copies. Audit at the destination
checks retained bytes and reconstructed rows. Missing or corrupt evidence
must remain an explicit audit failure, never an empty success or a reason to
rerun models. Since immutable objects already marked shipped are not rechecked,
ordinary sync alone is not a promise to heal later destination deletion; exact
retained bytes are needed for repair. No new automatic repair behavior is proposed.

Next proof: gate retains the existing historical
resume/attribution checks and delivery/audit acceptance test. An interrupted
copy, if additionally exercised, should use a controlled failing Storage fixture,
not native games or remote pressure. This review inspected source and existing
evidence; it did not rerun behavioral tests or certify remote delivery. The
historical fixture checks completed-match resume, unchanged match/table bytes
and both native assignments; it does not prove historical checkpoint admission
to w14. The delivery test checks destination audit and explicit failure after
evidence deletion. Those proof limits do not introduce a product choice.

## Publication check — 2026-10-27

`lf sync` completed cleanly; actual base remains `e742e8bff`. Publication is
blocked by unresolved behavioral validation failures. No PR was published,
auto-merge was not enabled, and Task state was not changed.

The broader evaluation run exposed one missed consumer in
`tests/eval/test_arena_log_conformance.py`: its reducer assertion still used
`CardGame.player`. Changed it to `seat`; all four conformance cases then passed.
The same broad run also failed both parameters of
`test_two_policies_never_share_a_forward_and_hung_service_is_bounded` during
ordinary policy admission: the service exceeded its 10-second request timeout,
before the deliberate stopped-service fixture. No timeout was relaxed.
Interrupted the remaining broad run to investigate its failures: 4 failed
(including the two subsequently fixed conformance cases), 23 passed, 1 skipped
in 696.84 seconds. This is not a complete suite pass.

The tracking/report run failed
`test_tiny_comparison_evaluates_live_and_notebook_refresh_preserves_edits`:
`experiment incomplete: None; training attempt 1 failed: RuntimeError: learner
exited -9: None`. Interrupted its remaining work after that failure: 1 failed,
20 passed in 622.94 seconds. The cause of signal 9 is not established; neither
host pressure nor a product regression has been proven. These failures need
resolution or successful evidence before publication. Both interrupted test
processes and their observed children exited.

`uv lock --check`, changed-file Ruff and formatting, and `git diff --check`
passed. The documented isolated `tests/remote/` command could not run because
that directory no longer exists. Used the current credential/storage owner
instead: `uv run --isolated --extra dev --extra artifacts pytest
tests/training/test_artifact_storage.py -q` — 8 passed. Initial report collection
without the notebook extra failed for missing `nbclient`; adding the configured
notebook/reports extras resolved collection before the behavioral run above.

The final focused run passed all 21 cases in 496.56 seconds:
`uv run --no-sync pytest tests/eval/test_etude_player_migration.py
tests/eval/test_elemental_long_eval.py tests/eval/test_elemental_long_report.py
tests/eval/test_mtg186_cards.py tests/eval/test_experiment_files.py
tests/eval/test_local_evaluation.py::test_cli_completes_without_training_cloud_or_callback
-q -x`. This covers historical byte/key retention, zero-match resume, both native
positions, report attribution, experiment adapters and the isolated local CLI
boundary. It does not resolve the separate broad-run failures above.
