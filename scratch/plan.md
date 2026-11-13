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
per-game step cap. The update benchmark's frozen receipt/stub and the retired throughput driver's exclusive execution tests are also deleted (review 4). No further deletion target is pending.
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
   run this branch. No local dependency path is committed.
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

Fifth recheck (after review 4 implementation): PR #103 still open at
`105abdd`; origin/main still `105e7e2aa`; pins still 0.2.16. Code matches
the review-4 section (loop phase/resource rows, `training_seconds`, the
benchmark calling `self_play`, `FROZEN_EXECUTION_REVISION` in regime.py). No
tests rerun.
Fourth recheck (after review 3 implementation): PR #103 unchanged, open at
`105abdd`; main has not moved past `105e7e2aa`; nothing new to merge.
Third realign recheck: Etude PR #103 is still OPEN at
`105abdd323016b7f7baaaf57aaaa7f5d896c71bf` (`mergedAt: null`); `pyproject.toml`
still requires 0.2.16. Main advanced to `105e7e2aa` (player-chosen mana
spending, Heartbeat of Spring), merged here as `117f2d22b`/`85fd4607e`. Among
training-adjacent paths it changed only `manabot/env/observation.py` and
`manabot/semantic/text_observation.py`; no loop, learner, recipe, fixture or
dependency file. The parity proof compared against main `6ce618b8a`, before
that observation change; it is not rerun here. Gate's focused checks cover the
merged observation code. No tests were rerun for realign.

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

## Concept review 5: next implement direction — 2026-11-13

Proposals drawn from the design, not statements anyone made. Two places where
a name sends a person somewhere that does not do what it says.

1. **The frozen-execution revision is not a revision of main.**
   `FROZEN_EXECUTION_REVISION` (regime.py), docs/training.md "Saved regimes",
   docs/experiments.md "Retained comparisons" and two tests name
   `f09f5c2919b8…`. Observed today: that commit is on this branch only
   (`git merge-base --is-ancestor f09f5c2919b origin/main` fails), is a
   mid-prototype documentation commit, and its `execution.py`,
   `state/recovery.py`, `state/game_errors.py` and `cli.py` differ from main's
   (255 insertions, 114 deletions against its merge base). The guide tells
   people to check it out "to reproduce a frozen experiment exactly"; no frozen
   experiment ran on that executor, main CI never ran it, and a squash merge
   leaves it off main's history. Name the main commit this cutover lands on:
   `105e7e2aa269b1af3946827dd6f4ab29af01f312` (current origin/main and merge
   base; the parity reference `6ce618b8a` is its ancestor). Change the
   constant, both guides, `test_attack_execution.py` and
   `test_recipe_etude.py`. If a later sync brings main commits that add frozen
   entry points, move the constant to that merge base. Gate checks the constant
   is an ancestor of origin/main. `439379869` (older collector snapshots) is
   on main and stays.
2. **experiments/README lists recipes the CLI cannot run.** Its table gives
   `ataraxos_move`, `value_model_baseline`, `direct_self_play` and
   `belief_sampler` as `--recipe` names run by `manabot train`. Observed:
   `manabot train --list` prints local, baseline, elemental, belief and
   batched_baseline only; the four modules hold zero-argument `TrainingRegime`
   builders and no `train(config)`. Rewrite the table so each row says what
   the module is now: `batched_baseline` runs (`train(config)`); the regime
   builders are retained records that execute only from the revision in item
   1; `ataraxos_ablation`'s `python -m` row says it refuses. The belief row
   points to `--recipe belief` (`manabot/recipes/belief_sampler.py`), the one
   live owner. `experiments/belief_sampler.py` then describes the same pilot a
   second time as a regime nothing reads (no test, saved file or report data
   names `frozen-policy-belief-pilot`; `ataraxos-transfer.md` imports a
   function it no longer has): delete it. The other three builders stay; tests
   and retained fixtures read them.

Checked and left alone: the loop file (835 lines: settings, state file, one
update, the loop) reads top to bottom with no driver object; the recovery
guide's kept/started-again table matches the code; `updates` meaning the
horizon in baseline/local/belief configs but only the default target in
`ElementalConfig` is the open CLI-config question already in questions.md.
No new product choice.
