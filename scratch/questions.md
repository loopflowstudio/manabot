# MTG-329 decisions and limits

Exact earlier findings/remarks: 3670249eb3:scratch/questions.md. Current implement context limits require condensing the appended, completed
concept reviews in design-train-api.md; their full text remains at that git
revision. The underlying design and exact decision notes are preserved. lf context reports default 16k-token / 131072-byte targets
and no usage entries; historical expanded plan/evidence are in git.

Reversible choices in use:
- Only explicit recipe labels become run/kind and run/recipe text metrics;
  Etude 0.4.1 track cannot set kind/job. Record fields win; unlabeled means
  unclassified. Local smoke declares labels, baseline/Elemental do not.
- Silent attempts retain earlier observed totals and warn about unmeasured
  duration. A wholly silent run has no invented totals.
- Elemental refuses all-invalid groups after preserving evidence. Native
  failures alone become score-null rows; ordinary Python errors propagate.
  Every callback invocation atomically allocates its own attempt-* directory.
- tla_frontier stays, with code settings and frozen JSON deck_source only;
  training-report stays as the Etude cross-run list. Notebook/profile reports
  and unused host schemas were deleted with the regime readers.
- GPU measurement method, attempts and launcher state: experiments/reports/etude-execution.md.

Etude questions, not Manabot blockers: same-seed dependent jobs in one command;
whether Agents(build=) builds every seat; command/track job and kind metadata
(then delete the text-label stopgap); launch options outside config identity,
so instrumentation (record_games, diagnostics) can change on resume. Belief's completed-checkpoint two-pass,
one-seed form is tested; no automatic scheduling claim.

Known limits: cooperative match deadlines cannot interrupt native calls;
Etude 2d1933 owns hard deadlines/dead processes, no nested Manabot workaround.
Frozen eval identities/schema retain their source revisions. Resume keeps
Adam/EMA/counters but restarts games/RNG; no in-flight recovery equivalence.
LLM declarations await EXP-70 and compatible frozen checkpoints.

Ideas outside this plan: move setup/numerics out of experiment/, move
regime_tag to its single caller; simplify all-invalid admission only if Etude
owns a declared minimum. No scope expansion authorized by these notes.

Sync onto main dcee96ccd1 (one game record), 2026-11-17 — choices made:
- Recording, statistics and diagnostics reach training as `SelfPlay.run_settings`
  (excluded from `learner_digest`); local/baseline configs carry the flat
  `record_games`/`statistics`/`diagnostics`/`stall_seconds` fields.
  `training_game_logs` on BaselineConfig/SelfPlay was branch-only and is replaced,
  not aliased.
- Managed (in-process) evaluation passes `Reduction` straight to replay; only the
  standalone `replay_retained` child keeps main's cloudpickle transport.
  `eval.game.play_game` writes ordinary games through the capture journal and
  keeps a canonical record on failure, as `supervise_game` does.
- `experiments/recording_cost.py` drove the retired regime executor and was
  removed (report notes `dcee96ccd1:experiments/recording_cost.py`).
- Not caused by the sync: `test_resource_limit_replay_rejects_old_schema...`
  diverges at action 32 of the frozen TLA fixture under the current engine
  (`replay_attempt` is byte-identical to main's).

The shared instrumentation fields preserve each recipe’s existing Pydantic extra-field and mutability behavior. The Elemental recording check uses two CPU streams and seed 197.
