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

2026-11-16 idea, not planned: an evaluation with no valid game is refused by
three rules (Elemental's callback; Duel's 200-valid reader; the shared deal
callback returns score-null rows). At Etude d9a6bc8 a raising callback is a
`failed` match record that a later `done` replaces, so Elemental's rule needs
no state. A declared minimum of valid results per match would be Etude's.

2026-11-17 reversible choice: use an isolated launcher database for the approved
L4 comparison instead of upgrading the shared launcher. Copy provider/storage
configuration only, so no other queued work can occupy the comparison machine.
RunPod/Vast/Lium report no L4 capacity; leave performance pending at USD 3.

2026-11-17 write-docs choices for the regime-reader deletion (mine, reversible;
nobody answered these). The docs are written to them:
- tla_frontier is treated as current: it stays, restating its settings in code.
  It reads one thing from its regime JSON, the 134-deck `deck_source`, as a
  `UniformDecks` (my choice in compress: too large to restate, and the pairs
  index it). Open: is it current, or should the file go with the readers and
  its report stand as the record?
- `manabot training-report` stays for Etude sources only. From Etude logs it
  has no phase breakdown or cost (one warning per run), so it is a run list by
  active time, throughput and a timeline. Open: keep that, or delete it too
  and rely on `etude report`'s PERF panels?
- `SelfPlayPoint` loses `stage`; `self_play_points` takes one run's rows.
  `self_play_figures` (drew from the notebook evidence object) goes.
- The guide names f4e1822faf for reading and running regime records because it
  is on main. The two revisions already named in docs/training.md,
  f577362135 (state.pt) and 83b77e6c64 (experiment modules), are commits of
  this branch and not on main; a squash merge leaves them reachable only
  through the PR ref. Not changed here.

2026-11-17 compress choices (mine, reversible): deleted `HostFacts` /
`MemoryLimit`, which the plan had kept for historical receipts; their only
reader was the deleted run record. Kept the training-report page's script and
`CohortRun` payload unchanged though its flamegraph and phase table can no
longer show anything; stripping them, or deleting the command, waits on the
open question above. Left the `notebook` extra and plotting dependencies in
pyproject for the same reason: CLAUDE.md's commands name `--extra notebook`.
