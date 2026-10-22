# Game crash recovery — MTG-176

Status: automatic capture implemented and focused checks passed, 2026-10-22; gate acceptance remains. The accepted outcome is
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
- Run summaries expose counts, samples, measured window and stop reason. Realign
  repaired a privacy leak: summaries now use the retained threshold reason or
  typed run status, never raw `run.error`; exception text stays private.

## Related work and evidence limits

The current base `95ca2059e` includes MTG-165's both-seat collector and the
MTG-174 shared payment repair plus native Elemental uniform sampler (PR #353).
MTG-172's planned policy identity, drift and KL-stop work remains separate.
The sync failure retained the entire prior implementation in stash
`d815bbfc94e03881681ca30283be0a4b3ccfea8d`. Code, tests and design were restored
onto this base; the stash remains intact. The proposed Wave memory patch remains
in that stash, outside this implementation's write authority.

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
  original absolute deadline and honest exported counts. It does not exercise
  provider reservations or submission-attempt reconciliation. Arena uses the same fixture and covers
  death/hang after acknowledgement, repaired advancement and unchanged evidence.
- Live arena events and journal recovery share `apply_game_event`, so pending and
  completed decisions have one projection. Failure replay validates the prefix
  once in `_replay_game` and attempts the pending Command in that same environment;
  callers close it even when replay raises. Receipt fields own counter defaults.
  The payment fixture shares native shadow setup and parses its ledger once.

## Superseded mechanisms

- Removed evaluation's success-only decision construction; pending evidence now
  precedes native execution, with post-state added only after success.
- Replaced `InjectedCollector` (invalid action per collection) in the combined
  restart proof with the shared offered-cast payment fixture. The separate native
  invalid-action replay test remains because it checks the actual native isolation
  API, not the payment class.
- Removed the duplicate live/journal event reducers and failure replay's second,
  unchecked reconstruction of the successful prefix. `replay_prefix` remains for
  its independent profiling caller.

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

## Delete — do not maintain

- Removed direct strict native calls from the audited scalar/vector and native
  pool/flat-MC consumers; the instrumented boundary is their sole live path.
- Replaced success-only semantic execution in compound/selection with pre-command
  capture while preserving their complete-log replay and learner data contracts.
- Removed `producers._NativeVector` and its cast: shards now own the concrete
  `CapturedVector` boundary. No exclusive fixtures remain. Removed Trainer's
  `_capture_update` mirror; capture reads the loop coordinate directly. Tolerant
  receipts reuse the strict capture scope's live run context instead of maintaining
  a second dictionary of update/cost/deadline fields.
- No recipe flag, second recovery counter, scheduler, payment solver or retry
  path replaces these calls. Existing tolerance and arena journals retain their
  separate accounting and process-supervision responsibilities.

## Remaining acceptance work

Gate owns broader affected-suite validation and acceptance review, including
full native debug checks, scientific cost/evaluation accounting and evidence
limits. No production rules repair, original MTG-175 reproduction, source archive,
provider reconciliation or portable CUDA restart is claimed. The missing original
prefix remains a historical limit. No paid execution or publication occurred.
Wave memory is unchanged; the prior proposed patch remains in the retained stash.

Check: `uv run --extra dev pytest tests/training/test_automatic_game_capture.py tests/training/test_game_errors.py tests/training/test_elemental_uniform.py -q` — 41 passed; `cargo test --manifest-path managym/Cargo.toml agent::vector_env --lib` — 12 passed; focused dependency-blocked play plus Study protocol/runtime checks — 20 passed. Broader acceptance remains with gate.
Changed-file Ruff, Rust formatting and `git diff --check` pass. The release wheel
was rebuilt from the pinned root environment and its cp312 extension installed
atomically. The first post-sync Python run exposed the vector diagnostic suffix
and correctly rejected mid-run source changes; the passing run held source stable.
