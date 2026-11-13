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
per-game step cap. No further deletion target is pending.
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

## Concept review 4: next implement direction — 2026-11-13

Proposals drawn from the design and the code, not statements anyone made.
The walk-through was "where does an update's time go on this machine", the
question docs/training.md (Monitoring), docs/workflows.md and the
`experiment-measure` / `qa` skills answer with `scripts/bench_training_update.py`
and the Perf panel. The cutover lost that path without listing it: the loss
reviewed so far is recovery only.

Observed at `569653ccd`, against main's executor:

- The loop's row has `time/collection` and `time/learning` only. The executor
  enabled the collector profiler and logged every subphase per update
  (`time_since(phase_seconds(...))`), plus `cuda_peak_metrics`,
  `mps_memory_metrics`, `coordinates/rss_bytes`, `coordinates/host_load_1m`,
  `training_seconds`, and a `transitions` clock. `open_collector` now builds
  `Profiler(enabled=False)` and nothing enables it; `phase_seconds` and
  `time_since` have no caller; `cuda_peak_metrics` is gone.
- `training/panels.py` still draws "Update time" from seven
  `time/collection/...` keys, "Process memory" and "Host load"; for a run of
  the new loop all are blank except `time/learning`. docs/training.md "Update
  metrics" still lists `time/learning/minibatches/forward_and_loss` and the
  CUDA high-water marks as logged rows.
- `scripts/bench_training_update.py` parses its arguments, then `_execute`
  raises the frozen-execution refusal; its docstring still says it runs
  through `execute_regime`. `experiments/training_throughput.py` still spawns
  it as its child and does not refuse.
- Recipes declare only `Step("updates")`, so `manabot training-report`
  (default `--training-clock training_seconds`) reports active time
  unavailable for every new run.

Direction (settled by the design: these are rows a loop logs through the
tracker, and the docs written on this branch already promise them):

1. The loop logs the per-update phase split again. Enable the collector's
   profiler where the loop opens it; each `step` adds the seconds the collector
   and learner profilers gained since the previous step, under the existing
   `time/...` key spelling so new runs overlay retained ones. Make
   `phase_seconds`/`time_since` read profiler timings rather than a
   `StageRecord`, keeping one implementation; the retained-record reader keeps
   what it needs. Timers only: the bitwise parity proof must still pass.
2. The row carries the resource observations the panels and guide name: CUDA
   (and MPS) high-water marks, process RSS and host load. One small function
   beside the loop; no sampling thread.
3. `bench_training_update.py` measures the loop: `--recipe` names a function
   returning `SelfPlay` (default `elemental_self_play`), it runs a few updates
   untracked or under `local_launch`, and prints the split from the rows of
   item 1. Drop the regime-file form and the `TrainingRun` receipt, and say so
   in docs/workflows.md and the two skills. Then `training_throughput.py`
   either reads the new output or refuses like the other frozen drivers; it
   must not spawn a child that can only fail.
4. Check Etude `105abdd` for a time clock a run can declare. If there is one,
   recipes declare `training_seconds` and the loop logs cumulative active
   seconds from `state.pt` (a new `Counts` field, so it survives resume);
   otherwise docs/training.md says the time report has no active time for loop
   runs. Do not rebuild the `transitions` clock unless a reader needs it.
5. One owner for the retired revision: a constant and one raising function in
   `manabot/training/regime.py` (the module that owns saved regimes), used by
   the CLI and the ~18 refusing entry points that each spell the SHA out.
   Fix `manabot/training/__init__.py`, whose docstring names `execute_regime`
   and which re-exports against the package rule; nothing imports from it.
6. After 1–4, make panels.py's header, docs/training.md "Update metrics" and
   "Monitoring" match exactly what the loop logs.

Not changed by this review: the open CLI-config and local-follower questions
(questions.md), the left-alone list above, and every delivery item.
