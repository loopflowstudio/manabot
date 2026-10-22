# Game crash recovery — MTG-176

Status: reconciled, 2026-10-22. Replay now separates binding, retained boundary
and error agreement across training, automatic capture and arena. Remote
accounting integration is implemented and reconciled; broader validation and
acceptance remain with gate. The supplied index-only/frame-classification feedback
is resolved by the current implementation and focused regressions; it no longer
identifies remaining implementation work. The accepted outcome is
durable, reproducible engine-failure evidence for training/evaluation, bounded
training quarantine/regather, and controlled continuation from committed learning
state after compatible repair. Capture, replay, repaired restart and honest
reporting must work together. No paid execution or campaign restart is authorized.

## Accepted direction and implementation choices

Jack Heart, 2026-10-22: isolated training game failures are discarded and replaced;
a configurable rolling settled-attempt error rate strictly above 10% stops
training. Count completed plus crashed games once per attempt, retain lifetime
and deck/pair/seat breakdowns, and preserve allowance, costs, deadlines and attempt
limits across restart. Quarantine both seats' uncommitted samples and pending
returns. Preserve unrelated streams, unchanged collection policy, update-boundary
freshness, immutable evidence and strict evaluation denominators. Regather does
not demonstrate a rules repair or unbiased strength. Frozen protocols stay frozen.

Engineering choices, 2026-10-22 (not additional decisions attributed to Jack):
opt-in `game_errors`, window/warmup 100/100, bounded per-game actions, complete-game
batches for both-seat and opted-in one-seat collection. Complete-game buffering
means no segment of a later-crashed game has reached the optimizer; one-seat batch
sizes and freshness differ from its historical quota path and must be declared in
comparisons. Native seed succession and declared matchups remain unchanged.
Setup/reset, encoding, policy/numerical and infrastructure failures fail closed.
External deadlines end collection rather than grant replacement time.

## What exists

- Native per-stream gameplay AgentError/panic outcomes and stream replacement;
  successful rows advance once even around a failed row. The old strict API
  remains for frozen training and evaluation.
- Both-seat quarantine and complete-game refill in `SeatRoutedCollector`.
  Settlement owns suffix removal, durable receipts and prefix clearing. Earlier
  completed games stay valid. Threshold stops are sticky across restart.
- Private fsynced `game-attempts.jsonl` records with setup, seed, explicit prefix,
  rejected action/frame, error, identities and update/cost coordinates; bounded
  local replay CLI. Catch-site native backtraces are retained, not an assertion
  that the original panic stack was captured. Identity hashes are not a source
  or binary archive.
- Append-only ancestor ledgers own failure allowance independently of optimizer
  snapshots. `CollectorSnapshot.game_errors` is unnecessary and absent;
  replacement debt derives from discarded/replaced totals and ledger reads stream
  receipts. No second launcher, scheduler, learner or payment solver is needed.
- Existing VerifyStore/update snapshots support CPU continuation when the recipe
  also declares current-game recovery. Explicit source-repair admission permits
  source/native byte changes while retaining world/content/input/runtime checks.
  Setup failures carry the inherited snapshot's admission forward. Game-level
  regather alone does not enable process recovery, especially on CUDA.
- Run summaries expose counts, samples, measured window and stop reason using
  the retained threshold reason or typed run status, never raw `run.error`;
  exception text stays private.

## Related work and evidence limits

The current base `95ca2059e` includes MTG-165's both-seat collector and the
MTG-174 shared payment repair plus native Elemental uniform sampler (PR #353).
MTG-172's planned policy identity, drift and KL-stop work remains separate.
The sync failure retained the entire prior implementation in stash
`d815bbfc94e03881681ca30283be0a4b3ccfea8d`. Code, tests and design were restored
onto this base; the stash remains intact. The local `origin/main` reference still
points to this base at reconciliation; no newer upstream delivery was observed.
Intelligence's objective remains reproducible training and full-game improvement
at declared cost. Its memory now records recovery lessons, without treating the
local fixture as strength evidence. No child Wave memories exist in this checkout.

Integration resolves both conflicts: native isolated completion/replacement uses
the same roster/seed reset authority as strict stepping; Python capture and
settlement retain actual sampled setups, not the recipe's representative pair.
Discarded rows leave valid transition exposure, and replacement roots enter the
existing exposure counts. No sampler, seed blacklist or payment solver is added.

Prior implementation evidence records collector quarantine/return alignment,
all-failure stopping, rolling threshold persistence, native rejected-action replay,
repaired-source restart after an uncommitted failure, preserved prior diagnostics
and cumulative time, strict incomplete evaluation, native debug vector tests and
the release extension build. These results are reused, not rerun for reconciliation.
MTG-175's offered Gnarled Professor rejection (seed 2165384) still lacks its
original 34-action prefix. The new deterministic fixture uses native Mountain /
Gray Ogre games and a narrow faulty payment validator (spends all red on one red
symbol); its repaired arithmetic spends one. Collection, fresh-process bounded
replay and repaired restart use that same fixture path. This is not a claim of an
MTG-175 reproduction or MTG-174 native rules repair.

## Implemented capture and combined fixture

- Arena sends the selected offer and Command before native execution. The parent
  fsyncs a private attempt journal before acknowledging; worker crash/hang retains
  the pending decision. Successful decisions stay separate. Journals own attempted
  command evidence; canonical cell traces remain their final projection and now
  publish atomically with private permissions. Replaying an interrupted journal
  reports unknown execution outcome honestly.
- Added bounded arena failure replay for both trace and journal inputs, checking
  prefix and pending bindings without policies, updates or rentals. Runtime source,
  native, content/input identities, registrations, limits, elapsed time and failure
  traceback are retained. Complete cohort/score admission is unchanged.
- The combined payment fixture proves a legal offered cast, durable training
  capture and fresh-process bounded replay, corrected fixture payment, source
  admission/world rejection, continuation through the same fixture collector,
  three optimizer updates, unchanged earlier diagnostics/snapshot/ledger,
  inherited failure allowance, cumulative elapsed/watchdog time, a nonempty
  original absolute deadline and honest exported counts. The same fixture now
  binds its allocation to the existing remote owners as described below. Arena
  uses the same fixture and covers
  death/hang after acknowledgement, repaired advancement and unchanged evidence.
- Live arena events and journal recovery share `apply_game_event`, so pending and
  completed decisions have one projection. Failure replay validates the prefix
  once in `_replay_game` and attempts the pending Command in that same environment;
  callers close it even when replay raises. Receipt fields own counter defaults.
  `tests/training/game_error_fixtures.py` owns the shared payment adapter and
  bounded subprocess replay; it imports no remote test owners. The combined
  proof lives in `test_game_error_recovery.py`, with one typed job-bound training
  entry point and one evaluator configuration across owner restarts.

## Automatic strict capture

- `manabot.env.failures` owns strict scalar/vector evidence independently of the
  opt-in tolerance ledger. Current-game prefixes, exact setups/seeds, masked rows,
  reset strides, pre-call offer frames, source/native/world/input/content identity,
  error/trace and available run/update/cost/deadline coordinates survive failure.
  Regime failure artifacts and ordinary Trainer paths retain private files.
- Scalar consumers cover distillation and pooled/value live games. Native pool and
  flat-MC search capture additionally retains search root, seed/parameters and
  policy-sampled slot actions. No model/optimizer or rental is needed for replay.
- Compound/selection retain pending Commands before native execution; their
  complete-game receipt admission remains strict. All newly opened command logs
  and failure evidence use private permissions. Unknown worker/batch outcomes do
  not create success/failure counts. Capture write errors annotate the original
  exception rather than replacing it.
- Replay uses the existing bounded CLI and never mutates retained files. New
  automatic records bind source/native bytes (explicit changed-runtime admission),
  reject world changes, and validate the retained pre-call frame. Native aborts,
  hard kills and hangs still require external supervision for positional/pool
  capture; source hashes do not archive binaries or code.
- Capture boundaries share one exception guard, including duplicate suppression
  when semantic/policy guards nest. `GameInputs.restart` owns clearing a stream's
  previous prefix and frame while keeping its setup. Replay compares an explicit
  retained phase/index instead of nested conditional expressions.

Replay CLI timeout/runtime checks and automatic-input reconstruction now live in
`manabot.env.replay`; `env.failures` owns capture and evidence decoding. `_retain`
owns create/fsync/register ordering for scalar journals and failures; shared
capture coordinates bind both pending Commands and failures to the live owner. Automatic
records check source and native bytes; historical arena/tolerance replay checks
native bytes and world, not their differently named source digests. This is
distinct from restart's stricter world/content/input/runtime admission.

## Remote accounting integration

The repaired-restart fixture now passes a persisted `Job.allocation` into every
training attempt. Existing `remote.cohort` / `remote.job_client` run against the
shared SQLite control-store and fake provider. A lost create response reconciles
one permanent claim; owner restarts retain the full unsettled reservation, source
intent, resource receipt and absolute deadline. Completion and failed-finalization
cases settle provider/storage cost exactly once, including prior/controller dollars,
and neither starts another rental. VerifyStore rejects a second recovery child;
training rejects renewal of the original allocation deadline. These preserve the
existing attempt boundaries: one permanent provider claim and one recovery child
per stopped attempt, not a new numeric process-retry policy.

The same runs export update checkpoints to the existing `CheckpointQueue`. Losing
the fake evaluator/owner charges its full reservation. Parent, setup-failed and
repaired run exports (including duplicate source paths) retain one identity per
checkpoint/protocol. Interrupted and failed evaluation attempts are not retried;
exhausting the cumulative allowance leaves the final checkpoint pending across
another owner restart. Original evaluation job bytes, training evidence, snapshots
and provider receipts remain unchanged. No evaluation score is fabricated.

Fixture boundary: only provider effects, evaluator processes and CPU placement
are fake. The plan freezes the actual recovery recipe, but the fixture substitutes
CPU placement for RunPod's CUDA resolver, whose process-recovery rejection remains
unchanged. This connects existing accounting owners to local repaired continuation;
it does not establish remote worker recovery, CUDA portability, provider service
behavior or an Etude SDK recovery path. No production linkage repair or new
orchestration was necessary. The previous feedback's remote integration gap is
covered by `tests/training/test_game_error_recovery.py`; the proof moved out of
`test_game_errors.py` when the shared payment fixture was extracted.

## Remaining acceptance work

Concept-review outcome, 2026-10-22: nothing to change and no open product
choice. The current model expresses Jack Heart's accepted experience without
another recovery lifecycle. The operator interaction is:

| Operator action / event | Concept and state-changing authority |
| --- | --- |
| Start an opted-in training recipe; games finish normally | `GameErrorPolicy` declares tolerance; `GameErrorLedger.record` settles attempts before the existing learner updates on complete valid games. |
| An isolated game fails during collection | The collector quarantines that game's two seats, records the attempt, and regathers under the unchanged policy; the ledger owns the rolling allowance. |
| Inspect the private receipt with the bounded replay command | Replay reconstructs retained inputs and returns `ReplayAgreement`; it changes no attempt, learning or evaluation state. |
| After compatible repair, explicitly resume the stopped run | `--resume-from` and `--source-repair-reason` enter existing restart admission; the child restores committed learning while ancestor ledgers and allocation/evaluation owners retain their independent accounting. |

The consequential counterexample remains advancement after repair: it cannot
turn the original failure or unknown execution outcome into a completed game,
reset the threshold, or produce a score. Likewise, matching error text at an
earlier action cannot establish reproduction, and a strict vector batch failure
cannot establish each stream's original outcome. The current diagnostic model
preserves these distinctions. Merging replay with attempt settlement or treating
restart as a fresh allowance would violate the accepted contract; no such
simplification is proposed. Existing usage docs already make these interactions
explicit, so no usage or skill rewrite is needed.

This review records a design conclusion, not additional execution evidence.
The next proof remains the bounded gate validation below; no implementation
iteration or new product decision follows from this review.

Gate owns broader affected-suite validation and acceptance review, including
full native debug checks and scientific evaluation accounting. No production
rules repair, original MTG-175 reproduction, source archive or portable CUDA
restart is claimed. The missing original prefix remains a historical limit.
Wave memory reflects the bounded remote integration proof and its limits.
No new product decision is required for the local recovery contract. Jack Heart's
2026-10-22 steer authorizes reviewable publication after bounded validation,
without merge arming; publication remains separate from this reconciliation.

Prior implementation checks and full design detail are retained at
`7d34af117212e733c5d73048531d47b9a5c5bc43:scratch/game-crash-recovery.md`.
The native extension was rebuilt and atomically installed there. The subsequent
replay diagnostics and focused regressions are reconciled below. This
reconciliation changes only the plan, diagnostic documentation and shared memory;
it reuses the recorded focused test result without rerunning unchanged code.

Check: `uv run --extra dev --extra artifacts pytest tests/training/test_game_errors.py tests/training/test_automatic_game_capture.py -q --tb=short` — 55 passed (recorded before reconciliation, reused); broader validation and acceptance: gate.

## What replay establishes

Jack Heart's accepted requirement is reproducible original failure evidence with
honest reporting after repair. Concept review's engineering clarification is
implemented; it introduces no new product choice. Three owners remain independent:

- `GameErrorLedger` settles attempts and owns quarantine/regather allowance.
- Bounded replay reconstructs immutable inputs and emits diagnostics only.
- Explicit TrainingRun restart restores committed learning while ancestor ledgers
  and existing allocation/evaluation owners preserve costs, deadlines and limits.

`ReplayAgreement` in `manabot.env.replay` defines binding agreement, execution at
the retained phase/index, error agreement and original outcome. Their operator
contract is documented in `docs/game-crash-recovery.md`. Missing metadata or
unreached binding/error checks stay null; the boundary flag stays false until
execution there is attempted. A wrong frame is binding divergence; the same error
before the retained boundary is a prefix failure; another error at the same index
is a new failure. Repaired advancement cannot revise the original outcome,
restore allowance or supply an evaluation score. Vector streams keep unknown
original outcomes even when the batch's failure is known.

Error comparison strips the vector diagnostic suffix and recognizes Rust/PyO3
exception aliases without changing evidence or observed error text. Automatic
replay reconstructs each stream in `_replay_game_inputs` and assembles agreement
fields once. Arena uses the boundary flag to distinguish prefix from pending
native errors in one handler, with environment cleanup in `finally`. This removes
repeated result assembly and exception handling without merging evidence formats.

Training's index-only interpretation and automatic replay's classification of
frame divergence as a native failure are gone. Arena exposes the shared
`matches_error` field, with one comparator across replay paths. Capture formats,
evidence payloads, runtime admission, allowance and accounting owners are unchanged;
no migration or frozen protocol change is needed.

Bounded regressions cover wrong frames, different errors/classes at the same
index, matching errors before the retained boundary, vector diagnostic suffixes,
older missing metadata, interrupted journals and repaired advancement. They compare
retained bytes and inherited allowance, including the existing combined restart
fixture. Broader validation and acceptance remain with gate.

## Publication verification — 2026-10-22

Reviewed the complete range from `95ca2059ea443351cabd4e8c322991c36c0d8212`,
including the uncommitted replay simplification and combined accounting fixture.
The remote main reference still matches that base. No additional implementation
gap was found at this publication boundary.

Fresh checks:

- `uv run --extra dev --extra artifacts --extra notebook --extra reports pytest tests/training/test_game_errors.py tests/training/test_automatic_game_capture.py tests/training/test_game_error_recovery.py tests/training/test_elemental_uniform.py tests/training/test_recovery.py tests/training/test_experiment_report.py tests/arena/ tests/remote/ tests/test_package_boundaries.py -q --tb=short` — 339 passed, 2 skipped.
- `uv run --extra dev --extra artifacts --extra notebook --extra reports pytest tests/env/ tests/sim/test_net_opponent.py tests/training/test_two_seat_learning.py tests/training/test_monitoring.py tests/training/test_checkpoint_queue.py -q --tb=short` — 146 passed.
- `cargo test --manifest-path managym/Cargo.toml` — full debug suite passed.
- Ruff checks for every changed/new Python file, Cargo formatting and
  `git diff --check` passed.

The first combined Python invocation stopped during collection because arena
and env both contain an unqualified `test_match.py`; the separate invocations
above cover both suites successfully. No code change was needed for that collision.

Publication remains reviewable delivery only. These checks do not establish
the unavailable original Gnarled Professor reproduction, a production rules
repair, CUDA/Etude recovery, provider service behavior or learning strength.
No paid execution, merge arming or Task completion is selected.

## Conversation review — Jack Heart, 2026-10-22

Jack Heart reviewed `scratch/pr-review.html` and replied “seems ok to me”.
This accepts the walkthrough's presented behavior and implementation choices;
the reviewed snapshot was base `95ca2059e`, HEAD `7d34af117` plus unpublished
working edits captured at 2026-10-22T19:04:33Z. It is not a claim that later
changes or broader acceptance checks have passed. Continue the already authorized
bounded validation and reviewable PR publication. The existing no-paid-execution
and no-merge-arming boundaries remain unchanged. At this read, the live pursue
Flow had reached `pr-publish`; leave that process in control.

## CI repair — ordinary allocation continuation

PR #355 head `280147fc8` failed main integration at
`tests/training/test_allocation.py::test_pause_export_and_resume_preserve_target_and_learning`.
Reproduced locally. The new original-deadline guard incorrectly covered ordinary
paused CPU recipes, whose documented contract permits a new explicit allocation.
Commit `90c551388` scopes that guard to opted-in game-error recovery, preserves
the ordinary continuation contract, strengthens its deadline/parent assertions,
and adds allocation coverage to the recovery gate. No auto-merge was enabled.

Finalized-source verification: allocation, recovery and combined game-error
recovery suites passed 29 tests; Ruff and whitespace checks passed. An earlier
verification pass was invalidated by formatting source during identity-sensitive
recovery tests; the clean rerun above is authoritative. Published head verified
on GitHub as `90c551388a1aea3421b976cccc8937f29e00d53e`; CI is running.

## Delivery direction — Jack Heart, 2026-10-22

Jack Heart now requests “automerge and ship”, superseding the earlier
no-merge-arming boundary. Run the ship gate, land PR #355 with waited auto-merge,
and complete MTG-176 after authoritative merge and outcome verification.
No paid execution or campaign restart is authorized. The latest repair is
`90c551388`; its 29 focused tests and checks are recorded above. Preserve the
actual evidence limits in PR copy before scratch cleanup.

## Ship gate — 2026-10-22

Gate reviewed `95ca2059e..90c551388` and the existing uncommitted delivery notes.
No further implementation gap was found. The final allocation guard preserves
ordinary paused continuation while retaining opted-in recovery's original deadline.
Reuse the 485 passing Python checks (2 skipped), full debug Rust suite, changed-file
Ruff and Cargo formatting from publication, and the finalized-source 29-test
allocation/recovery rerun above. No source changed during this gate.

Missing required checks completed on `90c551388`:

- `uv run cargo clippy --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings` — passed; bare Cargo initially selected unsupported system Python 3.14, so the successful check used the pinned uv environment.
- `uv run --extra dev --extra artifacts --extra notebook --extra reports pytest tests/training/test_experiment_report.py tests/training/test_experiment_runner.py -q` — 20 passed in 95.29 seconds, including live report discovery.
- `uv run --python 3.12 --extra dev --extra artifacts --extra notebook --extra reports pytest tests/semantic/test_elemental_world.py -q` — 20 passed in 1.41 seconds, including final-checkpoint reload.
- `uv run --extra dev --extra artifacts --extra notebook --extra reports pytest --collect-only tests/env/ tests/agent/ tests/model/ tests/remote/ tests/training/ tests/test_package_boundaries.py -q` — 977 collected; no new module-name collision. Collection is not execution evidence.
- `git diff --check` — passed.

Current-head CI has passed Rust, Python unit, long integration, Academy integration,
protocol/semantic conformance, clean-machine Play, package/install and diagnostic
checks. Main integration and Academy UI remain running, with no observed failure;
their results remain owned by CI and the following waited land operation. No
frontend interaction or world contract changed here, and no full local matrix
was repeated.

The published PR and `docs/game-crash-recovery.md` retain the consequential limits:
controlled payment fixture rather than the original missing Gnarled Professor
prefix; no production rules-repair or learning-strength claim; fake remote effects;
no CUDA/Etude process-recovery proof; hashes are not source/binary archives; native
aborts/hangs still need supervision. Gate passes for the authorized land step,
subject to its CI completion boundary. This gate does not merge or settle MTG-176.
