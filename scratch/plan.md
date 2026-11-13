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

Rechecked this pass: Etude PR #103 is still OPEN at
`105abdd323016b7f7baaaf57aaaa7f5d896c71bf` (`mergedAt: null`), so the release
dependency has not cleared. `origin/main` is still
`6ce618b8adacee3a4894d05572399853680bb9fe`, already incorporated. Requirements
and the runtime pin in `pyproject.toml` remain 0.2.16. No training test was
rerun for this reconciliation; earlier results in implementation.md stand.

## Concept review follow-through — 2026-11-13

Implemented the four independent items: retained experiment records moved to
experiment/records.py; live learning types moved to training/learning.py;
state.pt includes learner settings and refusal names differing fields with
saved/requested values; CLI exposes --updates with a hidden warning --steps
alias (removal 0.3.0). Recipe config discovery requires seed/device, which the
CLI actually passes. Serialized learning fields are unchanged; imports use the
new defining modules without compatibility aliases.

Deletion inventory addition: experiment/execution.py and the six live learning
definitions in regime.py were replaced by those modules. Retained stage/regime
records still use the learning types; offline records and schemas survive.

Still tied to SDK release: local_launch currently owns neither target nor
deadline arguments, so the CLI retains its environment save/restore. Seek
local_launch(out=, experiment=, until=, deadline=) in EXP-56; if unavailable,
keep the current code. Remove the PR #103 notice only when the real pin lands.
The repeated track/run.until opening and experiment runs/eval/report names stay.

Focused validation: 139 checks passed across resume/CLI, retained regime and
experiment records, package boundaries, CI selection and release checks; one
stale boundary expectation was corrected and its failed check rerun successfully.
Full gate and installed-consumer checks remain in Remaining work above.

## Compress follow-through after concept implementation — 2026-11-13

`self_play.learner_settings` now owns the machine-field exclusion for state
serialization, digests and refusal diagnostics, so those views cannot drift.
Loop helpers have descriptive function names without private prefixes, matching
the recorded naming direction. Recipe clock labels say collection/update
iterations; optimizer-step counts remain separate metrics. No serialized keys,
learner calculations or frozen evidence changed.
