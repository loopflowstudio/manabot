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

## Checks

- `cargo nextest run --workspace --profile ci` (debug): 1163/1164 pass;
  `checked_semantic_conformance_evidence_reproduces` fails as on base
  d95734088 (state_hash/content_digest only). Clippy: no findings in changed code.
- Subject fold: debug nextest filter `trigger|attack|combat_damage|dies|sacrific|observability|description|token` 249/249; extension rebuilt, imports.
- Compress passes: trigger-filtered debug nextest 139/139, then after the
  `subject_phrase` fold (`trigger|description|attack|combat_damage|dies|sacrific|observability|token`) 249/249.
- `uv run --extra dev pytest tests/semantic -q`: 197 passed (rebuilt extension).

## Remaining

- Gate: full debug nextest and semantic pytest at the current head (the last
  full runs predate compress e511f5adc, merge 55bc4a2b8 and the subject fold
  3a96a0d1e–395e108df), clippy, Python
  suites, engine budget digest (definitions changed).
- Conformance evidence re-record belongs to whichever change moved it first
  (the stack parent); re-record after rebase if still stale.

Inherited MTG-286 design note removed from scratch; read it at
`d95734088:scratch/tle-production-design.md`.
