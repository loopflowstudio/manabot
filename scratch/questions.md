# MTG-329 live limits and decisions

Jack Heart, 2026-11-16: "the whole point is the adoption" and
"These should be done/mainatined in this task". Full direction:
.lf/prompts/20261116-151510-YzKGdb-steer.md,
20261116-151754-HEl9sc-steer.md and 20261116-152021-ptKlWA-steer.md /
20261116-152021-BI2RUI-steer.md.

Etude d9a6bc8 has only a job deadline: play_together uses ETUDE_DEADLINE;
BrokenProcessPool requests grouped retry. Cooperative checks cannot interrupt
native calls. The steer assigns per-match hard deadlines/isolated dead processes
to Etude Task 2d1933, lifts the deletion stop and forbids nested Manabot children.
Keep cooperative limits, durable Commands, native panic/AgentError score-null
rows and subsequent games. No induced hang/load probe was run.

Frozen identity inventory: EvalCompute v2 hashes version/device/dtype, v1 also
allocation. Player spec/compute hashes feed player and match keys. Preserve
seed_compute_envelope_id for old paired seeds; never label auto CUDA as the old
CPU contract. New tournaments/registrations have fresh names. Runtime schema 8
cannot replay retained schema 7: run the named old revision, never rewrite it.

Actual local d9 belief CLI completed policy, sampler, then nothing to train
(exit 0 each). Automatic same-seed scheduling remains unsettled; explicit
completed-CheckpointRecord handoff works. No in-flight game/RNG/source recovery
claim. No CUDA, paid/provider, browser or real Rust panic injection proof.
LLM experiments were already non-runnable until EXP-70 and use frozen worlds;
keep that boundary explicit rather than claiming working managed LLM sessions.

Task steer: "Keep the scratch notes (design-train-api.md, ontology.md,
positioning.md, exp-56-compat.md, sketch.html, explore.html,
etude-readme-draft.md); do not delete or shorten them." These stay untouched;
other notes are condensed. Never increase context limits. Budget conflict: those
kept files alone are about 12k tokens, so scratch stays over its 12k target
(see `lf context`) until that steer is lifted. Earlier exact remarks
and findings: 4de3a76ba0:scratch/questions.md and 382c653ec8:scratch/questions.md.

2026-11-16 main sync: preserve the Etude d9 pin and retired execution/experiment
deletions; combine main's native recording and disclosure with function-based
collectors. Private training journals use the Etude recipe's `training_game_logs`
config field; old CLI/TrainingJob forwarding tests become recipe/loop checks.
Exact native snapshot/restore assertions remain retired; preserve auto-reset,
resolved-disclosure and export/replay checks. No launcher database was touched.

Open after the second compress pass (for gate):
- `test_recipe_report_preserves_paired_deals_and_policy_inspection` still fails,
  now in the choice-corpus audit: `native corpus audit incomplete or failed ...
  'error': 'KeyError: 79'` at `objects[index] for index in action.focus`
  (verify/choice_corpus.py, unchanged on this branch; seed 82000, family 10,
  focus of a type-11 action). The nightly baseline's `evaluate` takes the same
  path, so it would report software error until this is fixed.
- Nothing records host facts now: `infra.host.host_facts()` has no caller since
  the executor went. Left in place with its tests; wire it into the loop's rows
  or delete it.

