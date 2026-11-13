# MTG-329 plan

2026-11-12 naming direction, Jack Heart: "why is _verify_checkpoint private -- in etude i saw that we had instructions to keep functions private if they werent used outside, but i dont think thats necessary and find that to be distracting".

2026-11-12 recipe interface review, Jack Heart: "im not sure i liek this set u     self_play: SelfPlay = Field(default_factory=elemental_self_play)" and "i liekd it better when there were explicit eval(), train() and launch() functions". He clarified: "the idea was just to expose more primitives in etude to facilitate the ML OPs part of the train loop" and "but not to change the overall structure of hte recipe files". Preserve the existing recipe-file structure and explicit operations. Expose and use Etude primitives for the loop's ML operations; do not introduce a recipe framework or require the prototype's nested `ElementalConfig.self_play` structure. This narrows the earlier prototype direction; the proposed recipe restructuring is not the intended outcome.

Further clarification, Jack Heart: "well, to move the thigns that were in the execution loop itno things that are provided as general tools by etude" and "so they dont have to be in a 1000 line file". The intended simplification is extracting reusable ML operations from the large manabot executor into general Etude tools, then calling those tools from ordinary training code while preserving recipe-file structure. Merely adding conveniences around the existing executor would not achieve this outcome.

The prototype and EXP-56 API implementation exist. The current approach is
ordinary typed recipe functions using Etude run primitives, with the reusable
SDK delivered separately before the Manabot dependency can be pinned.

## Current status — 2026-11-13

The cutover is implemented against EXP-56's real SDK source. See
`implementation.md` for the deletion inventory, focused proof and remaining
release dependency. `cutover.md` is the accepted implementation direction;
`config-api-direction.md` records the Pydantic interface direction.

Earlier prototype progress, smoke transcripts and parity results remain in
`f09f5c2919b8b1a5cf6d72c4f4f4721dac68f653:scratch/plan.md` and `etude-api.md`.
Those prototype results predate the main/backend merge; they do not prove
post-merge parity. The old final.pt, local stub and nested recipe config are
superseded by immutable checkpoint paths and ordinary typed train functions.

Fresh-run parity after the backend merge is proved against the retired
executor at main `6ce618b8a` for bounded PPO and damped-policy runs: raw and
EMA weights match bitwise, with no numerical difference. Persistence and
tracking differ intentionally (immutable update-named policies and state.pt;
fresh native/RNG streams on resume). Details are in implementation.md. Commit
`0621262c5`, after that proof, changed only the self_play docstring and
removed unused record-model scheduler methods, so the proof still applies.

## Delete — do not maintain

Deleted the executor, stage/recovery admission, recovery state, local scheduler,
monitor follower, study launcher, RunContext and proposed_etude stub. Their
exclusive tests were removed; mixed tests retain model, collector, record and
evaluation coverage. The complete path inventory is in implementation.md. Compress also removed
unused attack registration, Elemental parent staging, value-readout launch helpers,
CUDA capture hooks and profiler-only state. Public frozen execution refusals and
historical report/data readers remain. Removed the unused scheduler methods
`MonitoringBudget.protocols_for`, `Hardware.admit` and `HardwareInventory.select`
from the surviving record models; their fields and validators stay readable.
The selection collector’s executor deadline callback and no-op
`selection_unbounded` adapter are also removed: its surviving callers use the
per-game step cap. The update benchmark's frozen receipt/stub and the retired throughput driver's exclusive execution tests are also deleted (review 4). Review 5 also removes the unused duplicate experiments/belief_sampler.py; the live belief recipe and historical evidence survive. No further deletion target is pending.
The obsolete attack executor tests now check refusal without creating a cohort.
The former recovery CI group only held pretrained loading; it joins training.

The CLI selects each built-in function/config pair in one table. The loop saves
resume state once when persistence and checkpoint cadences coincide, before
publishing the policy; docs use the same ordering.

Preserve saved TrainingRegime/TrainingRun and comparison records, serialized
checkpoint keys, world binding, game-error ledgers/replay and failure capture,
paired deals and training seeds. Current callers use train(config), real Etude
tracking and jobs.training. Frozen execution refuses with the last executable
revision; it is not translated to different learning semantics.

## Remaining work

1. Release EXP-56, raise every Etude requirement and the runtime pin together,
   refresh uv.lock and verify isolated/installed consumers. SDK 0.2.16 cannot
   run this branch: QA saw every `manabot` subcommand fail importing
   `etude.stopping`, and `etude smoke experiments/elemental.py` fail the same
   way after building the revision's locked venv (nothing submitted). No local dependency path is committed. Etude must also
   distinguish refused admission and interrupted attempts from real failures;
   the boundary investigation below identifies the current status overwrite.
2. Gate the affected suites, including the retained game-error replay issue
   below. Demo/review the explicit loss of in-collection/source-repair recovery.
3. Prove actual launch output restoration/retry and service continuation once
   the SDK/service are available; local target continuation and etude.call are
   exercised.

Demo judgment remains explicit: the loop gives up current-game recovery, exact
uninterrupted equivalence and in-run source repair; frozen staged/adaptive/
frozen-opponent launchers refuse instead of migrating their methods. Whether
a local checkpoint follower is wanted is still open; none is implemented.
These choices do not waive fresh-run parity or measurement-quality requirements.

## Contrary finding from the main sync

The earlier merge check passed 173 learner/collector/backend/admission/artifact/
boundary tests. Three variants of
`test_resource_limit_receipt_replays_its_retained_encoding_boundary` failed
`binding_mismatch`: replayed frame schema 8 versus retained fixture schema 7
(`tests/fixtures/tla-pilot-permanent-capacity.jsonl`). Comparison and fixture
matched main, whose CI was green. This pass leaves frozen evidence unchanged;
gate must resolve or explicitly report the failure, not regenerate it away.

## Realign observations — 2026-11-13

Full recheck history is retained at
`9eb98a6aaee9f9eeaabea17ec40dd56717b63f22:scratch/plan.md`.
Eleventh recheck (after QA follow-through `2d9fcf85c`): origin/main has
nothing new, Etude PR #103 still open at `105abdd`, no release, pins 0.2.16.
QA's actionable items (Ctrl-C guidance, guide cadence, conditional error
directories, Elemental arguments, reused-out error) are all implemented.
Tenth recheck (after review 10, merge `fe3c794e1`): Etude PR #103 still open
at `105abdd`; pins remain 0.2.16; origin/main has nothing new. The main merge
left cli.py, the recipe tests, docs/training.md, training/ and the lock
untouched, so review 10's recorded pass still applies. Main `f4e1822fa` (#510)
owns the frozen-execution reference. The parity proof predates main's
observation changes; merged code still needs gate coverage.

## Concept reviews 1–3: implemented — 2026-11-13

The review items were proposals drawn from the design, not statements anyone
made. All are implemented; details and run lines are in git history
(`488d6194f:scratch/plan.md`) and implementation.md.

- Retained experiment records live in `experiment/records.py`; live learning
  types in `training/learning.py`. Serialized learning fields are unchanged;
  imports use defining modules without compatibility aliases.
- `state.pt` holds learner settings (`self_play.learner_settings` owns the
  machine-field exclusion); resume refuses a changed setting, naming saved and
  requested values.
- PPO has a fixed horizon: optional `Learning.horizon` (omitted when unset, so
  retained regime bytes hold); live `SelfPlay` requires an explicit learning
  record and refuses PPO without a horizon; `step`/`learn` take no target.
  Rates read `update / horizon`, clamped, so a run extended past its horizon
  trains on at the final rates. Gray Ogre and belief pass their own update
  count.
- The loop adds no `final` tag; Etude's `TrainingRun.final` (last checkpoint
  of a finished run) owns the endpoint. The already-finished early return
  still calls `record_checkpoint`, covering a crash between `save_state` and
  the record. `test_recipe_etude.py` checks the derived endpoint advances from
  update 1 to 2 and no checkpoint carries `final`.
- CLI `--updates` sets only the stop target; hidden `--steps` alias warns
  (removal 0.3.0) that, unlike on main, it leaves the local/baseline recipe's
  rates and checkpoint grid unchanged. docs/training.md says the same and its
  recipe/loop excerpts match the code. Whether the CLI should set any recipe
  config field stays open in questions.md; nothing was built for it.

Left alone on purpose: `training/state/` keeps its name; the repeated
`track`/`run.until` opening; `baseline.evaluate` refusing a run that ended
anywhere but `config.updates`. `local_launch` owns neither target nor deadline
in EXP-56 `105abdd`, so the CLI keeps its environment save/restore; adopt
`local_launch(..., until=, deadline=)` only if the released SDK provides it,
and remove the PR #103 notice only when the real pin lands.

Last focused check (after review 3):
`uv run --no-sync --with /Users/jack/src/etude.exp-56-give-a-training pytest tests/training/test_recipe_etude.py::test_local_cli_stop_resume_and_target_only_continuation tests/training/test_self_play_loop.py::test_a_rerun_resumes_and_a_finished_loop_returns_at_once -q`: 2 passed; Ruff clean.
PPO fresh-run parity with horizon equal to the target is recorded in
implementation.md.

## Concept review 4: implemented — 2026-11-13

The review found unlisted monitoring loss: blank Perf phase/RSS/load panels,
missing device memory rows, a benchmark that called the retired executor and
no active-time clock for loop reports. Historical observations and the six-item
implementation direction are at `5ceb2d7dc:scratch/plan.md`.

- Enable collector timing in the loop; flatten live profiler totals and log
  each update's deltas with historical `time/...` names. The profilers are the
  one phase measurement: `training_seconds` adds each update's
  `time/collection` + `time/learning` (compress removed a duplicate wall clock
  that overrode those keys). Keep record timers
  only as retained evidence. Resources are sampled once per completed update:
  RSS, host load, process CUDA peaks and current MPS tensor/driver bytes.
- `Counts.training_seconds` persists completed collection-and-learning time.
  Current recipes (including batched baseline) declare Etude's Time clock;
  rows, tables, policy and belief checkpoints carry both clocks. No reader
  requires a transitions clock; its cumulative metric remains.
- Benchmark takes a zero-argument SelfPlay builder, runs the loop untracked and
  writes named settings, seed, rows and summary. Regime-file input, the
  TrainingRun receipt and the dead executor stub are deleted. Fresh output is
  required. The frozen throughput driver refuses before launch/spawn; remove
  its exclusive execution tests and retain setup/packing coverage.
- Frozen execution refusal has one constant/raising function in regime.py;
  CLI and frozen callers use it. training/__init__.py no longer re-exports or
  names the executor. Guide, workflows, panels and two skills match the rows.

The open CLI-config/local-follower questions and all delivery obligations above
remain. Validation and fresh-run parity results are in implementation.md.

## Concept review 5: implemented — 2026-11-13

The previous refusal named branch-only prototype `f09f5c2919b8…`, whose
executor differed from main; it was unsuitable as a durable execution
reference. The constant, both guides and refusal tests now name pre-cutover
main (now `f4e1822fa…` after the #510 sync; see the sixth recheck). Exact reproductions use each experiment's own
recorded revision and lock. Older collector snapshot revision `439379869`
stays. Gate should recheck ancestry after any main sync; move the constant if
new main commits add frozen entry points.

The experiment index distinguishes runnable batched-baseline `train(config)`
and `--recipe belief` from the retained Ataraxos, value-model and direct
self-play regime builders. It says the ablation module refuses execution.
Deleted unused `experiments/belief_sampler.py`; the live owner remains
`manabot/recipes/belief_sampler.py`. Historical report illustrations and
source manifests retain their original paths; no evidence was rewritten.

SDK release/pin, installed-consumer and actual launcher/service proofs remain
delivery obligations. Open CLI-config/local-follower judgments remain in
questions.md. Focused validation is recorded in implementation.md.

## Concept reviews 6–7: implemented — 2026-11-13

The guides launch current `experiments/elemental.py` with its required budget
arguments. Every frozen experiment header that shows a launch command now
marks it historical and names the pre-cutover revision, including indirect
refusals through learning_signal, l4_step_profile and the retired CLI. Reports,
evaluation commands and frozen evidence stay unchanged. Nightly budgets remain
1680 seconds per train/evaluate step under the 1800-second timeout.

The complete module inventory and check results remain at
`e66568121557afbc094f410e8f3a2a3ba206e3c2:scratch/plan.md`.
Docstring-only AST checks, Ruff check/format and `git diff --check` passed;
no test or parity rerun followed. Open choices and delivery obligations remain.

## Concept review 8: implemented — 2026-11-13

docs/training.md only. The loop template and `record_checkpoint` row pass
`**clocks(state)`, and `clocks(state)` is among the pieces. The belief excerpt
matches `manabot/recipes/belief_sampler.py`: it checks
`run.latest_checkpoint()` and records the sampler at `**policy_record.at`;
the Etude table lists that lookup. The remote excerpt has the shape of
`experiments.elemental.runs`: `runs(revision: str) -> list[Run]` from
`training(..., revision=, setup=machine_setup(...)).runs(197, 198)`, and the
smoke command passes `--revision`. The single-Job README example is valid and
stays.

Validation (rerun at compress): the exact Python fence, extracted to a
temporary file, through `read_job_file(Path, ["--revision", HEAD])` with
`uv run --no-sync --with /Users/jack/src/etude.exp-56-give-a-training python`:
two Runs (seeds 197, 198), bound revision, setup, 12 h / USD 9 and the file
stem as experiment. `git diff --check` passed. No test or parity rerun for
prose. Compress found nothing further to cut in the guide.

Open choices (questions.md: CLI recipe-config fields, local follower) and the
SDK release/pin, installed-consumer, launcher/service, recovery review and
retained replay obligations are unchanged.

## Concept review 9: implemented — 2026-11-13

On entry at any checkpoint-grid update, the loop saves state and records the
checkpoint before opening the collector. This repairs an attempt lost after
saving the milestone; re-entry preserves existing policy bytes and Etude's
single record at the original complete clock coordinate. The at-target repair
stays. The tracked-resume test covers a missing policy file and a policy already
recorded before the failure. docs/training.md's template and cadence section
describe this rule (compress merged the piece-table copy into the cadence text). Conditional search's import guard now
names the surviving self_play module.

The review's full counterexample and direction remain at
`5da3574799fb39c8525c2e937beff50583be6741:scratch/plan.md`.
Validation for this pass is in implementation.md. SDK release/pin, installed
consumers, actual launcher/service proofs, recovery review and retained replay
mismatch remain later delivery obligations; open choices are unchanged.

## Concept review 10: implemented — 2026-11-13

The CLI passes the module name to local_launch for module:function recipes;
function-specific default output directories remain distinct. The training
guide describes both names. The regression exercises two function selections
through the CLI and reads their real Etude runs under the module experiment.
It reproduced unsafe_run_or_experiment_name before the fix, then passed.
Validation is in implementation.md. No learner behavior changed.

The original counterexample and direction are retained at
`751deeb2fd0f6a2177217cbcefed803d05a2eec1:scratch/plan.md`. Open CLI-config/local-follower choices and SDK release/pin,
installed-consumer, launcher/service, recovery review and retained replay
obligations remain unchanged.

## Concept review 11: implemented — 2026-11-13

Recovery guidance separates each checkpoint's writing revision (and Etude run,
set only under a launch) from Etude's attempt records. The fixed-revision rule
applies to launched runs; a local rerun after a code change resumes when
world, seed and learner settings match. No revision refusal was added. Prose
only; full text at `626dab4b48:scratch/plan.md`. Obligations unchanged.

## Concept review 12: nothing to change in the design — 2026-11-13

Read the loop, CLI, four recipes and docs/training.md against this plan
(fresh run, stop, rerun, extension, belief rerun). No concept, name or
interaction needs to change. Etude PR #103 is still open at `105abdd`.

Checked and left as is: after a finished belief run, `run.latest_checkpoint()`
and `TrainingRun.final` name the sampler (tag `belief`), not the policy. The
recipe reads only `.at`, and the latest record always shares the latest
policy's coordinate (`run.checkpoint` returns the existing record for an
already recorded policy), so reruns and extensions stay correct. An
`eval(state)` over a multi-step recipe must select by tag; none exists today.

One wording fix for the next step that edits the guide (not a design change):
docs/training.md "Checkpoint cadence" says "1000 is the usual grid", but no
current recipe uses it (Elemental 50, batched baseline 64, baseline
`updates // 2`). Name Elemental's 50 or drop the sentence; the 26,000/27,000
continuation example may stay as an example.

Open choices (CLI recipe-config fields, local follower) stay in questions.md
for demo, additive and non-blocking as before. Delivery obligations unchanged.

## Concept review 13: nothing to change in the design — 2026-11-13

Read the QA follow-through (`82991cf85`, `2d9fcf85c`) against the loop, CLI
and guide. Three endings of a local command are now distinct and each says
what to do next: planned stop (75, state saved), Ctrl-C (130, last saved state
kept), reused `--out` for another run (`StateMismatch`, CLI usage error). No
name or interaction needs to change. PR #103 still open at `105abdd`; origin/main
has nothing new.

For the delivery check with the released SDK, not a Manabot change: recipes
call `self_play` inside `track`, so a refused `--out` is raised after Etude
opened the attempt. Under `105abdd` that marks a finished run failed (QA
probe), so `StateMismatch`'s "before anything is trained or written" and the
guide's "refused before training" hold for `state.pt` and checkpoints only.
When the Etude fix lands, confirm the refused reuse leaves the run's state,
seed and config record as they were; do not move the check ahead of `track`
or add a Manabot status store to compensate.

## QA follow-through — 2026-11-13

Ctrl-C now exits 130 with the output directory to reuse and guidance to resume
from the last saved state. It does not save a possibly partial learner update.
The regression interrupts learning after a saved update, verifies the saved
state and policy bytes survive, and resumes to the next update. The guide and
package README name Elemental's 50-update grid; the guide marks conditional
game-error directories and supplies Elemental's required smoke/launch arguments.

Etude PR #103 remains open at `105abdd`. A bounded real-SDK probe (no games)
confirmed both a rowless refusal and KeyboardInterrupt overwrite a finished
run's state with failed while keeping its reached update (mechanism and probe
in implementation.md). This belongs to Etude; no Manabot status store or
exception translation was added. A genuine failure before the first row must
still be reportable, so skipping every rowless exception is not a sufficient
general fix. SDK release/pin, installed consumers and launcher/service proofs
remain blocked there. Validation: implementation.md.
