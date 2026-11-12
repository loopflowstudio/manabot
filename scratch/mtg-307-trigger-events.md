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

## Compress (2026-11-12, head 079a8de32)

`TriggerSubject::matching(controlled_by, predicate, another)` owns the box;
the four named constructors and both tests call it. Debug nextest
`trigger|description|observability|token` 149/149.

## Status (realign, 2026-11-12, head f9b297fc8)

Nothing remains to build. The latest base merge (7ef6ce588) only adds TLE
spells to `tests/sim/test_heuristic.py`; no trigger code moved. Remaining:
parent PR #469 lands first (open, blocked), then PR #484 CI (preflight pending
at this pass) and the Engine Budget digest adoption noted under Gate.

QA re-run (head da964c606): catalog trigger text byte-identical to 0c2ce18e1
(276 lines); 1332 random games 0 errors; Academy game to Game Over with a
draw-event trigger-order decision. No fixes. See `scratch/qa.html`.
