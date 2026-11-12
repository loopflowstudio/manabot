# MTG-297: Heartbeat of Spring

Design: `scratch/tle-production-design.md` with the review decisions in
`scratch/heartbeat-design.md`.

## Scope in this PR: the production half

Steer 41eb2776 (2026-11-12): "Do the production half now. Build spending
(pool classes, receipts, Animal Attendant, Toph, Uncle's Musings) on MTG-242's
ManaRestriction/SpendingContext and proposed restricted-mana fact row; don't
rewrite pool storage in parallel. If spending must wait for MTG-242 to reach
main, finish production and stop there."

MTG-242 is PR #496 (open, not merged, last updated 2026-11-12); it touches
`state/mana.rs`, `flow/mana.rs`, `agent/continuation.rs`, `cards/src/tla.rs`
and `observability.rs`, so expect conflicts there. This PR stops after
production. Animal Attendant, Toph, Greatest Earthbender and Uncle's Musings
stay undefined; their Scryfall ids are recorded below for the spending Task.

## Built

- `TriggeredManaAbility { who: Actor, predicate, output }`,
  `TriggeredManaOutput::{Fixed(Mana), OneProducedType}` (`state/card.rs`).
  Badgermole Cub is `who: You, Fixed(G)`; Heartbeat of Spring
  (`cards/src/chk.rs`) is `who: AnyPlayer`, land, `OneProducedType`.
- Tapping no longer fires bonuses (`flow/zones.rs::tap_permanent`). Deleted
  `tap_mana_bonus`, `tap_mana_bonus_from`, `tap_bonus_sources`,
  `fire_triggered_mana_abilities`.
- `flow/mana.rs`: `mana_triggers` (both battlefields), `tap_bonuses` (read
  before the tap and any sacrifice, so the source keeps its LKI),
  `resolve_mana_bonuses` (after the output enters the pool; a choice among
  several produced types suspends as a `ManaChoice` with `produced` and the
  remaining `bonuses`). No mana produced, no trigger (CR 106.12a).
- `Decision::ManaChoice` gains `produced` and `bonuses` (serde-skipped when
  empty). The output choice carries the captured bonuses; Henge/Citadel
  suffixes still run before the choice, so bonuses resolve after the whole
  mana ability (CR 605.4a).
- Planner: activation outputs and fixed sources fold in every bonus
  combination (`production_alternatives`); `mana_choice_completes_payment`
  adds the pending bonuses as an Output obligation. A fixed source whose bonus
  would need a type choice is left out of automatic production; no current
  fixed source produces two types.
- Decision B's `producer` field became `produced` (the mana the ability
  made) plus `bonuses`; the bonus choice is offered as ordinary `ManaChoice`
  actions, so no new action block. `TriggeredManaOutput` variants and the
  tapping `Actor` are guarded into the learning tokens
  (`observability.rs::every_triggered_mana_output_and_actor_reaches_the_learning_tokens`).
- `docs/engine.md`: Heartbeat row removed, 59 of 61, corpus count updated.

## Remaining

- Spending half (pool classes, receipts, Animal Attendant, Toph, Uncle's
  Musings): waits on PR #496 reaching main (see heartbeat-design.md). The
  Task outcome names all four cards, so this PR alone does not complete it;
  spending is either a later PR in this Task or a linked Task, which is the
  caller's decision.
- Rebase onto main after #496 merges if this PR is still open.

## Spending cards (for the follow-up)

- Animal Attendant: tle 128, `a3e6ff4d-bf7d-488f-a02f-f40e7f8761ca`, {1}{G}
  Creature — Human Citizen 2/2.
- Toph, Greatest Earthbender: tle 70, `b15acc09-dd32-4e93-a788-7c0ae965a0c8`,
  {2}{R}{G} Legendary Creature — Human Warrior Ally 3/3.
- Uncle's Musings: tle 73, `55724dab-93c8-4a7f-89c2-13e94820635b`, {2}{G}{G}
  Sorcery.

## Checks

- `cargo test --manifest-path managym/cards/Cargo.toml --test cards`: 691 passed;
  the added token guard passes (debug).
- `rules_tests`, `content_pack_tests`, `search_state_contract`: passed.
- Corpus: first 1,000 MTG-213 trophy decks re-admitted from source with the
  rebuilt extension: 998 admit (Heartbeat decks 173, 925, 932 included); the
  two failures are Lita.
- Clippy (workspace, tests) clean.
