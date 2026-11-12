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

Concept reviews 4–7 (through head eb17d1124): no change; only the
`matching` constructor, docs reflows, the Engine Budget reference move and
scratch landed.

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
Re-runs at da964c606, bc35142da and abeab7a87: catalog trigger text
byte-identical, 1332 games 0 errors, Academy games to Game Over (a draw-event
trigger-order decision; Tiger-Seal upkeep on the stack), no page errors.

## Docs

Shipping docs: `docs/engine.md` (card definitions: prowess example, event list)
and the trigger rule in `AGENTS.md`/`CLAUDE.md`. Re-checked against
`ability.rs` through e95481614: no stale variant names.

## Compress (head 079a8de32)

`TriggerSubject::matching(controlled_by, predicate, another)` owns the box;
the four named constructors and both tests call it. Debug nextest
`trigger|description|observability|token` 149/149. Re-checked at 17768efe2,
5a7ab52ea and 21e161b2a: no code or docs changed since; nothing to reduce.

## Gate (head b206873cd)

Rust changed since the publish gate only through the `matching` constructor
(serde-transparent `Box`, no token or digest change). `cargo fmt --check` clean;
debug nextest `trigger|description|attack|combat_damage|dies|sacrific|observability|token`
249/249; clippy `--workspace --all-targets` no findings. Later gates
(aaa6dd564, 9479b26a5): no suite inputs moved, no reruns.

## Status (head 38f6311a6)

Nothing remains to build. Parent PR #469 landed on main as 306612865
(2026-11-12) and is merged here with c32b321b0 (#480) and f4c948f4a
(MTG-301); relative to 21e161b2a the merges change no `managym/`, trigger docs
or AGENTS/CLAUDE files, so earlier checks stand. Branch is level with
origin/main. Remaining: PR #484 CI on 38f6311a6 (started 13:36Z). Its Engine
Budget digest likely moves (definitions changed): adopt #484's own run.
QA re-run at 6430cd60c (after the #469/#480/MTG-301 merges; no `managym/`
change since abeab7a87): 1332 games byte-identical, 0 errors; Academy game to
Game Over (39 clicks, no page errors) with the Tiger-Seal upkeep trigger on the stack.
