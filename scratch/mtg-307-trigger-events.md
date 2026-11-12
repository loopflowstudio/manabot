# MTG-307: triggers as an event over an actor

Jack Heart, 2026-11-12: "it looks like there may be more opportunity to
modularize our triggers?" then "whenever X does Y basically".

## Built

`TriggerCondition` (`managym/state/src/ability.rs`) replaces 21 actor-named
variants with `Casts { who, predicate, nth }`, `Draws { who, nth }`,
`BeginningOf { step: TurnStep, whose }`, `Performs { who, any_of:
Vec<KeywordAction> }`, `PutIntoGraveyard { whose, predicate }` and
`DealsCombatDamage { subject, to: Actor, per: Per }`; `Actor::includes` is the
one actor test. `TriggerSubject::Any(predicate)` expresses "a creature deals
combat damage to you". Matching is in `flow/triggers.rs` (`turn_step`,
`performed_action`, `keyword_action_event`); draws and keyword actions now set
"that player". Descriptions are one template per event
(`agent/ability_description.rs`, test
`trigger_descriptions_fill_one_template_per_event`); guards and
`every_trigger_event_parameter_reaches_the_learning_tokens` in
`cards/tests/cards/observability.rs`. Docs (AGENTS.md, docs/engine.md) match.

Card code uses `TriggerCondition::casts(who, predicate)` and
`beginning_of(step, whose)` for the ~50 common sites; descriptions share one
`agree` verb helper.

Tokens and digests move (decision in `scratch/questions.md`).

Concept review (2026-11-12) folded the last actor/batch-named shapes in:
`Attacks { subject, per, attackers: AttackSize::{Any, Alone, WithAnother} }`
(multiplicity in `trigger_multiplicity`, like `DealsCombatDamage`; card code
uses `TriggerCondition::attacks(subject, per)`), `Sacrificed { subject }`
(matched by `subject_matches_moved_card`, like `Dies`) and
`PutsCounters { who, kind, on }`. `Per::Batch` on Fire Navy Trebuchet, Teo,
The Earth King, Bitter Work and Obsessive Pursuit; Beastmaster Ascension is
`Per::Object` and now gets one trigger per attacker
(`zen::beastmaster_ascension_triggers_once_for_each_attacker`).
`trigger_multiplicity` owns both: each arm checks its gate (defender, attack
size) and yields the creatures; one count applies `subject` and `per`.

Kept on purpose (actor fixed by the rules, or single-consumer meta-event):
`YouCastThis`, `SpellTargetsOnlyThis`, `YourAttackingCreatureTriggers`.
Later Task: merge `DelayedEvent` into `TriggerCondition` over a captured
subject.

## Object actor controller (concept review 2, 2026-11-12)

The `Any`/`Another` × `YouControl`/whoever grid is now one
`TriggerSubject::Matching { controlled_by: Actor, predicate, another }`
beside `This` and `EnchantedCreature`; card code uses `you_control`,
`another_you_control`, `any`, `another`. Each matcher in `flow/triggers.rs`
is one test through `Actor::includes` (moved-card path keeps the
`source_was_on_battlefield` gate); `subject_description`/`one_or_more` are one
`subject_phrase` differing only in quantity words ("a creature an opponent controls" covered in
`trigger_descriptions_fill_one_template_per_event`). The learning-token test
now asserts all six `controlled_by` × `another` subjects tokenize distinctly.

## Concept review 3 (2026-11-12)

No change. Remaining non-event variants each have a rules-fixed actor or one
consumer: `BecomesTargeted` (Ward's "an opponent controls", CR 702.21),
`YouCastThis`, `SpellTargetsOnlyThis` (emn), `YourAttackingCreatureTriggers`
(tla); `Chapter` and `NoCounters` are state triggers. Every printed "one or
more" trigger in the card sets is an `Attacks`, `DealsCombatDamage` or
`PutsCounters` field, so `Per` needs no other home yet.

Concept review 4 (head eeb9af27e): no change. Since review 3 only the
`TriggerSubject::matching` constructor (same model) and a base merge landed.
Concept review 5 (head 80d3d80af): no change. Since review 4 only the
engine-guide reflow, the Engine Budget reference move and base docs landed.
Concept review 6 (head 9991e7088): no change. Since review 5 only scratch
changed; main's two new commits touch no `managym/` or docs.

## Checks

- `cargo nextest run --workspace --profile ci` (debug): 1163/1164 pass;
  `checked_semantic_conformance_evidence_reproduces` fails as on base
  d95734088 (state_hash/content_digest only). Clippy: no findings in changed code.
- Subject fold: debug nextest filter `trigger|attack|combat_damage|dies|sacrific|observability|description|token` 249/249; extension rebuilt, imports.
- Compress passes: trigger-filtered debug nextest 139/139, then after the
  `subject_phrase` fold (`trigger|description|attack|combat_damage|dies|sacrific|observability|token`) 249/249.
- `uv run --extra dev pytest tests/semantic -q`: 197 passed (rebuilt extension).

## Gate at publish (2026-11-12, after syncing onto base 3eb1f283a)

- Re-recorded w17 (017d0ebfd): semantic-kernel evidence (same counts and
  outcomes), Academy fixtures, authored-match receipt and parity receipts with
  pins, belief v3 pins. Boxed `TriggerSubject::Matching.predicate` for clippy.
- Debug nextest 1167/1167; clippy and fmt clean; Python semantic, world-change,
  academy, env and model suites 960 passed.
- `tests/belief/test_learning.py::test_real_population_learning_is_held_out_and_beats_p0`
  fails here and on the base worktree at 3eb1f283a (held-out NLL -0.046 < 0.1).
- Engine Budget digest likely moves (definitions changed): adopt the PR's run.

Inherited MTG-286 design note removed from scratch; read it at
`d95734088:scratch/tle-production-design.md`.

## QA (2026-11-12, head 0c2ce18e1)

`scratch/qa.html`: trigger text for all 276 catalog triggered abilities diffed
base→head (91 changed, all closer to printed text); 1332 random full games over
every authored setup, 0 errors; Academy table played to Game Over twice with
triggers (incl. Tiger-Seal upkeep) on the stack. Pre-existing: non-self ETB/dies
still say "When" (fix moves the w17 authored-match receipt).

## Docs (write-docs, 2026-11-12)

Shipping docs: `docs/engine.md` (card definitions: prowess example, event list)
and the trigger rule in `AGENTS.md`/`CLAUDE.md`. No stale variant names remain
in docs. This pass only reworded the event list in `docs/engine.md`.
Re-checked at d002b743b against `ability.rs` (variants, constructors, links): no edits.
Re-checked at 27eb17aab: since d002b743b only `docs/development.md` (base) moved; no stale variant names; no edits.

## Compress (2026-11-12, head 079a8de32)

`TriggerSubject::matching(controlled_by, predicate, another)` owns the box;
the four named constructors and both tests call it. Debug nextest
`trigger|description|observability|token` 149/149.
Re-checked at 17768efe2: only that fold and a docs reflow touched code or
docs since; nothing further to reduce, scratch under budget.

## Status (realign, 2026-11-12, head 6fe2d75b9)

Nothing remains to build. The latest base merge (59cf727bc) only moves
`managym/perf/baseline.json` to PR #469's Engine Budget run and documents that
in `docs/development.md`; no trigger code moved. Remaining: parent PR #469
lands first (open, blocked), then PR #484 CI. Realign at 0fa09f9e4: main has
not moved past 59cf727bc; #484 passes Python unit, Semantic Kernel, Academy,
Clean-machine Play and packaging so far, with Rust Tests, Integration, Protocol
v1 and Engine Budget still pending. This branch changes definitions, so its Engine Budget digest
likely moves again: adopt #484's own run against the new reference.

QA re-run (head da964c606): catalog trigger text byte-identical to 0c2ce18e1
(276 lines); 1332 random games 0 errors; Academy game to Game Over with a
draw-event trigger-order decision. No fixes. See `scratch/qa.html`.

## Gate (head b206873cd, 2026-11-12)

Rust changed since the publish gate only through the `matching` constructor
(serde-transparent `Box`, no token or digest change). `cargo fmt --check` clean;
debug nextest `trigger|description|attack|combat_damage|dies|sacrific|observability|token`
249/249; clippy `--workspace --all-targets` no findings. Base merges since then
are Python token-stream work this branch doesn't touch; full matrix left to CI.
Reflowed the event paragraph in `docs/engine.md`.

Implement pass (head d002b743b): nothing to build. Main's only new commit,
c32b321b0 (#480), is Python policy-input admission with no `managym/` or docs
changes and merges cleanly (`git merge-tree`).
Implement pass (head d230b3173): nothing to build. Main's new f4c948f4a
(MTG-301) adds only `experiments/` files and merges cleanly.

QA re-run (head bc35142da): 1332 random games, 0 errors, output identical to
da964c606; two Academy games to Game Over, no page errors. No fixes.

Gate (head aaa6dd564): since b206873cd only the `docs/engine.md` reflow, the
base merge (`managym/perf/baseline.json`, `docs/development.md`) and scratch
changed; no suite inputs moved, so no reruns. PR #484 CI pending at this pass.
Compress (head 5a7ab52ea): since 17768efe2 only the base merge and scratch
changed; no code to reduce. Dropped inherited MTG-286 notes from
`scratch/questions.md` (at `5a7ab52ea:scratch/questions.md`); scratch 2651 tokens.
