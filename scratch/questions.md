# MTG-329 live limits and decisions

2026-11-16 QA fix choice: reject an Elemental callback with zero valid games
after playing its whole group and retaining evidence. The callback is Etude's
success boundary, so an all-invalid group fails even in a larger evaluation;
mixed groups still return native failures as score-null rows. Ordinary Python
errors propagate from play and replay. No cached/frozen results are rewritten.

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
other notes are condensed. Never increase context limits. `lf context` on
2026-11-16 reports the default 16k-token/131072-byte targets and no usage
entries; the earlier 12k-target warning is historical. Earlier exact remarks
and findings: 4de3a76ba0:scratch/questions.md and 382c653ec8:scratch/questions.md.

2026-11-16 main sync: preserve the Etude d9 pin and retired execution/experiment
deletions; combine main's native recording and disclosure with function-based
collectors. Private training journals use the Etude recipe's `training_game_logs`
config field; old CLI/TrainingJob forwarding tests become recipe/loop checks.
Exact native snapshot/restore assertions remain retired; preserve auto-reset,
resolved-disclosure and export/replay checks. No launcher database was touched.

2026-11-16 follow-up: delete unused host collection instead of adding a second
machine-collection path to the loops. Retain HostFacts/MemoryLimit schemas and
round-trip tests. Replayed seed 82000's 23-action prefix: focus 79 is the card
identity of permanent 103 (Water Tribe Rallier) during waterbend payment. The
corpus must resolve both aliases to one descriptor, as the v11 encoders do.
Spell card/stack aliases also share a row; collision checks must include stack.
No native rule, observation layout, frozen evidence or strategic label changes.

2026-11-16 retry choice: each Elemental callback invocation gets a retained
`attempt-*` directory under its match key, locally and in launched output.
Use atomic directory allocation rather than replacing journals or resuming a
partly played game. Existing flat match evidence stays untouched.

2026-11-16 idea, not planned: `manabot.training.report.job_labels` takes
`(word, TrainingRegime)` but reads only `regime.agent` (width, layers). Taking
a `ModelConfig` would let a recipe-only experiment label its jobs and drop the
module's saved-regime import; one caller (`experiments/tla_frontier.py`).

2026-11-16 idea, not planned: an evaluation with no valid game is refused by
three rules (Elemental's callback; Duel's 200-valid reader; the shared deal
callback returns score-null rows). At Etude d9a6bc8 a raising callback is a
`failed` match record that a later `done` replaces, so Elemental's rule needs
no state. A declared minimum of valid results per match would be Etude's.
