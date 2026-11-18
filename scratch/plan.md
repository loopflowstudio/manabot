# MTG-353 plan

## Revision after human review, 2026-11-18 — build this next

Direction, Jack Heart, 2026-11-18:

- "i feel like we may be layering too kmuch in at at time. I am not sure i
  want a \"manabot recheck\" command. Mostly Etude's report framework is the
  right way to line up data against previous data. The job here is just to
  define when its valid, and also to make it possible to use a previous
  checkpoint in new evals in a way that is safe even when minor evolutions
  have happened"
- "I thought the prpoosal was also getting rid of the world number."
- On a trained card whose engine serialization changed: "i would say we just
  always assume respelling", and "honestly, rules change still probably doesnt
  affect that muc about what the agent thinks".
- "Learning speed and throughput are even more important ... its the early
  part of the curve im more focused on".
- Asked whether to write this into the plan and send the Task back to
  build: "yes".

What the PR is after this revision: a checkpoint keeps loading and can be
seated in a new evaluation as cards, names and columns are added; the docs
say when two measurements can be lined up. Nothing else.

1. **Remove `manabot recheck`.** Delete `manabot/eval/recheck.py`, the
   `recheck` CLI entry, `StrengthReference`, `experiments/references/`,
   `tests/eval/test_strength_recheck.py`, and every doc, AGENTS.md, WORLDS.md,
   README and `.lf/skills/qa.md` passage about rechecking or strength
   references. `experiments/best_player_on_main.py` and its report stay as the
   exploratory evidence that u1952 plays on main; both arms run through
   ordinary local evaluation, and the report's commands no longer name
   `recheck`. Published data stays as it is.
2. **Remove the world number as a concept.** No rule moves it and nothing
   gates on it. `WORLD_VERSION` stays as a fixed label that existing records,
   bindings and frozen evidence carry; do not rename serialized keys or delete
   the constant here. WORLDS.md becomes history of past numbers with one
   opening paragraph saying so. Remove "world" from the terms list in
   `docs/observation-and-model.md` and from the AGENTS.md compatibility lines.
3. **Say when measurements can be lined up**, in `docs/evaluation.md`, in a
   short section and with no new tooling:
   - An old checkpoint seated in the same evaluation at the same revision as
     the thing it is compared with is always a valid comparison.
   - Two measurements from different revisions can be lined up when they
     share the measurement itself (opponent identity, decks and deals,
     decision settings, raw or averaged policy, and the machine for
     throughput) and no declared change that affects it sits between their
     revisions. Declared changes already exist: pinned-pilot break reasons,
     Engine Budget reference reasons, and the heuristic's source hash.
   - Early learning is scored on the raw learner; an averaged export still
     close to initialization hides it.
   - Lining data up against previous data is Etude's report framework's job.
4. **Trained cards always read their stored program.** Keep the behaviour as
   built and say it plainly in the docs: a change to how the engine
   serializes a trained card is always treated as a respelling, including
   when the card's rules changed; the effect shows up as strength in an
   evaluation. No classification, fingerprint file or declaration.
5. **One count in the evaluation's record.** When a checkpoint is admitted to
   an evaluation, its participant record carries the revision that wrote it
   and the number of cards read from a stored program. Use the existing
   admission and record path. If Etude's participant record has no place for
   it, write it to the evaluation's local output and note the Etude gap in
   `scratch/questions.md`; do not build a manabot-side workaround.
6. Keep as built: name binding and `model_interface`, the rename table and
   writer catalogs, pinned pilots in CI, Academy's refusal reasons.
   `docs/academy.md`'s refusal examples quote the loader's real sentences.

Not in this PR: coverage headlines in the load report, paired verdicts,
run-derived references, a decision-level field, classifying changed
programs. Follow-up work is MTG-386.

Build to the published PR and stop for review; do not land. No paid run.


## Implementation status, 2026-11-18

The dedicated strength command and reference files are gone. Both arms of
`experiments/best_player_on_main.py` use the local checkpoint admission and
evaluation APIs. Generating the historical cohort from the frontier experiment
reproduced all 16 deleted pair records exactly, including deck order and
seeds. The retained corpus produced four eligible new-card pairs. The
published 160-game measurement and its artifacts remain the evidence.

The evaluation guide now defines same-revision and cross-revision comparisons
and raw-learner scoring. The compatibility guide describes the stored program
for every trained card, including cards whose rules changed. WORLDS.md records
the historical labels. Pinned pilots, writer catalogs and loader refusal
examples in Academy were already built.

Item 5 is built. Local and Etude admissions record the writer commit and
stored card-program count from the admitted checkpoint. Etude match players
already retain their registration in config. Old registrations still serialize
with their original fields and hashes. The authored matchup, ABI and viewing
boundary govern arena compatibility. A retained w2 participant served a
checkpoint in a current selected matchup and its games replayed.

## Concept review of the built change, 2026-11-18 — build this next

The build matches the revision above. Three small corrections remain, all
inside items 2, 4 and 5; nothing else changes.

1. Rename `PlayerRegistration.checkpoint_source` to `writer_revision`
   (`manabot/sim/registration.py`, `manabot/eval/admission.py`,
   `manabot/eval/agents.py`, `tests/eval/test_local_evaluation.py`,
   `tests/eval/test_etude_agents.py`, `tests/eval/test_evaluation_match.py`,
   and the paragraph in `docs/evaluation.md`). Elsewhere `checkpoint_source`
   names the record of commit, dirty flag and Etude run; the registration
   holds only the commit, which `PolicyCheckpoint.writer_revision` already
   names. The key is serialized into evaluation records, so it is renamed
   before any retained record carries it. Keep it omitted when absent so old
   registrations hash unchanged. Passed over: storing the whole source record.
2. `docs/observation-and-model.md`, "Running a checkpoint on a later
   revision": `BindingReport` holds counts only. Remove "and changed card
   keys" and "The report names the 136 that serialize differently"; say the
   count instead.
3. `README.md` "What it trains on now" links "world w17" as the current
   world. Link the card count to the pool without a world number.

Then gate and publish for human review. No landing or paid run.

Left as they are and shown on `scratch/concept-review.html`: the several
names for the two kinds of checkpoint (world-bound, deck-source; setup-bound,
fixed-match), and fixed-match checkpoints refusing after cards are added.
`scratch/` goes before landing.
