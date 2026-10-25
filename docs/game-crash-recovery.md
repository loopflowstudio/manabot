# Training game failures

New single-stage `TrainSelfPlay` recipes can opt into `game_errors`:

```json
"game_errors": {
  "threshold": 0.1,
  "window": 100,
  "warmup": 100,
  "max_game_actions": 40000
}
```

These are engineering defaults. The denominator is settled native game attempts:
completed games plus crashed games, once per game rather than per seat or vector
call. The latest 100 outcomes are assessed after 100 have settled; 10 failures
are tolerated, 11 stop the run. Warmup must fit the window. A threshold stop is
retained across restart and cannot be cleared by restarting. Without this field,
frozen recipes retain their strict behavior and their serialized identity.

An engine rejection or native gameplay panic quarantines only that stream's
current game. Other streams advance exactly once, including streams preceding
or following a failed row in a vector/sharded call. Both seats' pending decisions
and finalized rows from that game are removed. Complete earlier games remain.
Replacement uses the native next seed and declared matchup distribution. Fixed
streams keep their setup; Elemental uniform streams sample the next pair through
the native root sampler. Receipts retain each actual sampled deck/sideboard pair,
including replacements. No seed, card or pairing is blacklisted. The unchanged behavior policy gathers
replacement experience before the existing optimizer update.

Opted-in one-seat collection also finishes games before updating. This is a
deliberate stronger buffering policy: no segment of a subsequently crashed game
has already reached an optimizer. It changes one-seat batch size/freshness and
must be declared in comparisons. Both-seat collection already used this boundary.
There are no fabricated terminal targets or bootstraps across replacement.

Native truncations and the per-game action limit count as failed attempts.
When the run itself stops mid-collection, each unfinished game's prefix is
retained as a `deadline` attempt (allocation deadline or wall allowance) or an
`abandoned` attempt (cancellation or another supervision limit). These are run
interruptions, not game errors: they count in `attempted` and `abandoned`, never
in the error window, `crashed`, discarded samples or replacement debt, and they
grant no time for regather. Reports name a paused run's stop reason
`allocation_deadline`. Setup,
reset, buffer, policy/numerical and infrastructure errors remain fatal. Game
recovery contains Rust unwinds, not process aborts, out-of-memory termination or
native hangs; external process supervision still owns hard execution limits.

## Private evidence and replay

`game-attempts.jsonl` is an append-only, fsynced, private receipt ledger registered
in TrainingRun and the stage's rejected artifacts. Failure rows retain the exact
setup/seed, stream, actor seat, complete explicit action prefix including the
rejected action, native pre-command frame, error class/message/native trace,
source/native/content/world/input identities, update/checkpoint coordinates,
elapsed game time and allocation deadline. Automatic trivial native decisions
replay under the same `skip_trivial=True` setting. Never publish this ledger as
a viewer-safe replay: seeds, deck lists and frames can reveal private state.

Replay a failed receipt by its one-based ledger line, using the retained source
and built native extension:

```sh
uv run python -m manabot.training.state.game_errors .runs/example/game-attempts.jsonl --attempt 1 --seconds 30
```

This launches a bounded local replay subprocess, without training or a rental.
Replay reports whether the retained input binding matches, whether execution
reaches the retained failure boundary, and whether the observed error matches.
A changed offer frame is a diagnostic mismatch. A different error at the same
action is a new failure. Advancement after repair describes only this replay;
the original failure or unknown outcome remains unchanged. Older evidence lacking
a frame or comparable error reports the corresponding check as unavailable.

Training, automatic capture and arena replay use the same diagnostic fields:

| Field | Meaning |
| --- | --- |
| `matches_binding` | `true` when the retained pre-command frame agrees; arena also checks the saved prefix, selected offer, Command and state bindings. `false` identifies divergence. `null` means required metadata was absent or the check was not reached. |
| `at_retained_failure` | Whether replay attempted execution at the retained boundary (phase and action index). It stays `false` until that attempt, including on earlier binding divergence. It can be `true` for rejection or advancement after repair; it alone establishes neither binding nor error agreement. |
| `matches_error` | Whether the observed exception class and message agree. `null` means comparable original metadata or an observed exception is unavailable. Vector `; native_failure=` diagnostic suffixes are excluded only from comparison; evidence and returned error text keep the full payload. |
| `original_outcome` | The retained outcome or arena failure label, independent of this replay's `status`. Interrupted journals remain `execution_outcome_unknown`, even when replay now rejects the Command. |

An available frame is checked before the retained action executes. Training and
automatic replay report `binding_mismatch` on divergence; arena reports
`prefix_mismatch` or `pending_mismatch`. These results do not report a native
execution failure. `status=failed` describes an observed replay exception; only
agreement on binding, boundary and error supports reproduction of a known original
failure. A successful replay after repair reports advancement/completion. Replay
never rewrites evidence or restores failure allowance.
A matching native error earlier in arena's saved prefix remains `prefix_failed`,
with `at_retained_failure=false`; error agreement alone is not reproduction.
Error comparison recognizes isolated Rust `AgentError` / `NativePanic` as the
same classes PyO3 exposes as `PyAgentError` / `PanicException`; retained class
names stay unchanged.

`--allow-changed-runtime` explicitly permits different native
bytes for this diagnostic; world changes remain rejected. Keep the original
source/archive and native build available—the identity hashes are not a source
or binary archive. The deterministic regression uses native Mountain/Gray Ogre games plus a narrow
fixture payment validator: its faulty arithmetic spends every available red mana
on the one red symbol, leaving none for the two generic mana. The corrected
fixture spends one red. The same validator path is used by collection, bounded
replay and repaired continuation. This demonstrates an offered-action/payment
mismatch and compatible continuation without adding a production payment solver;
it does not reproduce or repair the original Gnarled Professor rules defect.

Run exports and experiment dashboards show settled/completed/crashed attempts,
discarded/replacement samples, the measured error window and stop reason. The
retained `by_cell` breakdown uses ordered deck+sideboard identities and actor
seat at settlement. Replacement samples count valid rows replacing discarded
ones, capped by the outstanding discarded count. Completed-game counts remain
separate from failures. Regather is observable recovery, not evidence of rules
correctness or unbiased strength.
Dashboard stop summaries use the retained threshold reason or run status; raw
exceptions remain in private operator records because they may contain game data.

## Committed-state continuation

For local CPU process recovery, also declare the existing current-game snapshot
policy on the regime, for example `"recovery": {"checkpoint_updates": 1,
"max_game_microsteps": 40000}`, and use an admitted recovery schedule. Legacy
whole-run journals cannot represent discarded games and are rejected with
`game_errors`. Existing CUDA process-recovery limitations remain; GPU game-level
regather does not imply portable optimizer/RNG restart.

Restart uses the existing VerifyStore, exclusive writer lease and digest-verified
snapshot. Learner, Adam, EMA and sampling RNG resume at the last committed update;
unfinished samples are regathered. Receipt ledgers from every ancestor are read
independently, including failures after that snapshot. These receipts alone own
the error allowance; optimizer snapshots cannot restore an older allowance.
Missing, malformed or
policy-incompatible ledgers fail closed. Original ledgers and snapshots remain
unchanged. Prior costs, stage/run allowances and any original absolute allocation
deadline carry forward. No launcher, reservation, evaluation retry or attempt-limit
policy is introduced.

The original-deadline requirement applies to recipes opting into `game_errors`.
Ordinary paused CPU recipes without that policy retain their existing explicit
continuation under a separately admitted new allocation.

After a compatible source repair, the operator must supply a reason:

```sh
uv run manabot train --regime recipe.json --seed 197 --out .runs/repaired \
  --resume-from ORIGINAL_RUN_ID \
  --source-repair-reason "Payment implementation repaired; world and input contract unchanged"
```

Keep the original recipe, seed, checkpoint cadence and allocation arguments.
The reason permits changed Python/native source and extension hashes only;
world, content, action/observation ABI, architecture and runtime compatibility
must still match. The new TrainingRun records its source and parent relationship.
Existing monitoring receipts at committed update coordinates are preserved.
An exhausted error threshold remains exhausted even after source repair.

## Evaluation failure capture

Arena evaluation and monitoring retain pending Commands before native execution.
The parent writes and fsyncs a private per-game journal under `failures/`, then
acknowledges the worker. The pending record contains the selected offer, complete
Command, frame digest, pre-state digest, actor and policy coordinates. Only a
successful step moves it into the completed decisions. A native rejection, panic,
worker exit or deadline cannot turn that pending Command into a completed one.
Python tracebacks, source/native/content/input identities, registrations, setup,
seed, limits and measured execution seconds remain private.

The journal is the durable attempted-command evidence; the existing cell trace
is its completed-attempt projection. Trace publication now uses fsync and atomic
replacement, preserving earlier games if a later write fails. Both artifacts have
private file permissions. Journals are create-only and replay never changes them.
If the parent disappeared without observing the worker's outcome, journal replay
labels it `execution_outcome_unknown`, not an engine failure or completed game.

Replay a failure from a cell trace or its one-game journal:

```sh
uv run python -m manabot.arena.replay .runs/arena/traces/CELL.commands.jsonl.gz --game 1 --seconds 30
uv run python -m manabot.arena.replay .runs/arena/failures/CELL.SEED.LEG.jsonl --game 1 --seconds 30
```

The subprocess bound includes prefix reconstruction and the pending native step.
Replay validates the saved prefix, Command/offer/frame bindings and pre-state
before attempting the pending action. It reports prefix divergence, an observed
failure with separate binding/boundary/error agreement, or advancement after repair.
`--allow-changed-runtime` permits changed native bytes for diagnostic replay;
a different world is still refused. Successful prefix receipts continue to prove
only retained successful Commands. They do not validate a failed game's outcome.
Monitoring still rejects incomplete cohorts and preserves failed denominators;
engine failures receive no loss, draw or strength credit. No evaluation is retried.

## Automatic capture on strict paths

Failure capture is independent of `game_errors`. Strict native scalar/vector
calls retain private evidence before propagating the original exception; they do
not reset a failed stream, retry collection, or enable process recovery. Existing
frozen recipes keep their serialized identities. The tolerance ledger remains the
only owner of settled-attempt counts and recovery allowance.

`manabot.env.failures` owns automatic capture; `manabot.env.replay` owns
input reconstruction and the shared CLI deadline/runtime checks. Strict training
through `execute_regime` writes private `failures/` under the TrainingRun output
directory and registers those files as the failed stage's rejected artifacts.
Standalone environment/data consumers without an owning capture scope use
`.runs/game-failures/` and attach the private path to the exception.
Files are create-only, mode 0600, fsynced with their directory entry. A storage
failure is attached to the original exception rather than replacing it or
pretending evidence was retained. These files must not become public replays.

Each record retains setup (including sideboards/content pack), game seed,
stream/actor, explicit input prefix and pre-call native offers, source/native/world
identities, available input/content identities, exception/traceback and elapsed
time. Regime scopes add run/stage, update/cost coordinates, parent attempt,
committed snapshot and original allocation deadline. Data generators retain
available game/sample/decision indices.
Capture does not create reservations, reconcile submissions, or change budgets.

| Execution boundary | Retained evidence |
| --- | --- |
| Strict self-play, including multistage and sharded producers | Current native game seeds, per-stream prefixes and active pre-call frames. Masked rows are excluded. Native auto-reset strides and per-stream setups survive replay. |
| Standalone `sim.rollout` and vector environments | The same strict vector capture, including deterministic built-in random/passive opponents. A failed batch produces no completed evaluation result. |
| Search players | Scalar live-game inputs; native flat-MC and rollout-pool calls additionally retain their exact root, search seed/parameters and sampled slot actions. Pool replay needs no model or policy sampling. |
| Compound and selection training | Fsynced pre-execution Command/offer journals, followed by success or failure. Complete-game logs remain separate and incomplete games cannot pass their existing replay admission. |
| Tolerant self-play and arena | Their existing settled-attempt ledger and pre-command arena journal remain authoritative, as described above. |

A strict vector exception does not say which other rows advanced before it was
raised. Capture retains every active row's pre-call inputs and labels no extra row
as a crashed or completed game. Bounded replay reports each row separately.
Its top-level `original_outcome` describes the batch; each row's original outcome
remains unknown, including rows whose replay now advances or rejects an input.
For built-in opponents, the frame binds the surfaced input; the retained native
seed and opponent policy reproduce automatic opponent decisions inside that call.

Replay any automatic failure file with the existing command:

```sh
uv run python -m manabot.training.state.game_errors .runs/example/failures/ID.jsonl --attempt 1 --seconds 30
```

The subprocess bound covers reconstruction and the attempted failing call. New
records require matching source and native bytes unless
`--allow-changed-runtime` is explicit; world mismatch is always refused. Replay
checks retained frames and distinguishes a prefix failure from the last attempted
input. An interrupted semantic journal has unknown execution outcome. Replay
never overwrites evidence, changes allowance, or runs an optimizer.

Positional and pool prefixes are in memory until a caught failure; process abort,
hard kill, out-of-memory termination or a native hang before capture still needs
external supervision. Semantic Commands and arena pending decisions are journaled
before execution. Identity hashes are not source/binary archives. The focused
strict-path checks include the shared payment-mismatch fixture; that fixture
remains distinct from the unavailable MTG-175 prefix
and from any production payment repair. Broader validation and acceptance review
remain with gate.
