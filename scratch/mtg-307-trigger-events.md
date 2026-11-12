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

Tokens and digests move (decision in `scratch/questions.md`).

## Checks

- `cargo nextest run --workspace --profile ci` (debug): 1162/1163 pass;
  `checked_semantic_conformance_evidence_reproduces` fails identically on the
  base commit d95734088. All 14 regenerated traces equal base's except
  `state_hash`/`content_digest`, so play is unchanged.
- `uv run --extra dev pytest tests/semantic -q`: 197 passed (rebuilt extension).

## Remaining

- Gate: clippy, Python suites, engine budget digest (definitions changed).
- Conformance evidence re-record belongs to whichever change moved it first
  (the stack parent); re-record after rebase if still stale.

Inherited MTG-286 design note removed from scratch; read it at
`d95734088:scratch/tle-production-design.md`.
