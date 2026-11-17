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
- GPU calibration: alternating three-update Elemental arms, six-update resume
  baseline stopped after update 3, 16 paired eight-leg deals per endpoint.
  This small measurement cannot establish strength equivalence. Same GPU,
  own locked SDK per source archive, direct SSH rather than queued launch.
- Use fresh isolated launcher /tmp/mtg329-runpod-vast-etude. The old Lium intent
  remains untouched evidence in /tmp/mtg329-performance-etude; its paid pod
  was deleted and confirmed absent. Lium excluded by latest steer (73e1a9).

Etude questions, not Manabot blockers: same-seed dependent jobs in one command;
whether Agents(build=) builds every seat; command/track job and kind metadata
(then delete the text-label stopgap). Belief's completed-checkpoint two-pass,
one-seed form is tested; no automatic scheduling claim.

Known limits: cooperative match deadlines cannot interrupt native calls;
Etude 2d1933 owns hard deadlines/dead processes, no nested Manabot workaround.
Frozen eval identities/schema retain their source revisions. Resume keeps
Adam/EMA/counters but restarts games/RNG; no in-flight recovery equivalence.
LLM declarations await EXP-70 and compatible frozen checkpoints.

Ideas outside this plan: move setup/numerics out of experiment/, move
regime_tag to its single caller; simplify all-invalid admission only if Etude
owns a declared minimum. No scope expansion authorized by these notes.
