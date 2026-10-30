# T5 (a7234f): permanent copying — plan and status

Status (2026-10-30): implemented and stacked on T3 #399 (merged T3
`08bf54de2`, which contains main `10eeb1f55`; main `6f48f5ca0` merged since).
T3's PR head `4c4dd0217` adds only that main merge and a scratch note, so there
is nothing to sync from T3. All three cards are defined and supported. Stop at
demo. T4 488a40 is still implementing with no PR; sync onto it when it
publishes.

T3's plan, inherited through the stack, is at
`08bf54de2:scratch/6875f0-tla-t3-characteristics.md`. Two of its accepted
models are the base this builds on: continuous effects as records
(`state/effect.rs`: `Modification` × `Affected` × `EffectDuration`), and open
vocabularies as v5 fact rows (`FactKind`).

## Design review (Jack Heart, 2026-10-30, comment 10a31948) — accepted, done

1. Stack on T3 #399 now; drop the duplicate `sorcery_timing`. Done: the
   merge kept T3's identical field and check (`check_activation_terms`).
2. Replace tensor-only copy identity with generic, composable
   modified-characteristics observation (copy + general exceptions +
   layered effects, tensor and tokens). The shape was proposed in scratch
   first, then implemented as below.
3. Sacrifice history on `TurnState`, shared with T4 raid. Done in T4's shape.
4. Define Phoenix Fleet Airship. Done.

## Cards

| Card | State |
| --- | --- |
| Joo Dee, One of Many | supported; surveil, token copy, sorcery timing |
| Ran and Shaw | supported (out of `tla::INCOMPLETE`); nonlegendary copy |
| Phoenix Fleet Airship | supported; end-step self-copy after a sacrifice, eight-namesake creature, crew 1, flying |

## Model as implemented

- **Copy effect (layer 1) as data.** `Card.copy: Option<CopyEffect { of,
  exceptions }>`. `of` is the printed definition at the root of the copy
  chain. `exceptions: Vec<Modification>` accumulate down the chain
  (CR 707.9b). `Game::create_token_copy` (`flow/zones.rs`) is the one
  authority. It materializes the copiable values into the card's definition
  via `Modification::apply_copiable`, so every layer reader starts from them.
  It also records the inputs. A departed source is copied from its retained
  card (LKI). The state hash includes `copy` when present.
- **Exceptions share T3's vocabulary.** The new `Modification::Nonlegendary`
  is either a copy exception or a record. The legend rule reads it through
  `Game::permanent_is_legendary` (`flow/sba.rs`). `apply_copiable` returns
  false for rules effects; a debug assert guards authoring.
  `Effect::CreateTokenCopy { exceptions: Vec<Modification<EffectValue>> }`.
- **Observation.**
  - `PermanentData.copy: Option<CopyData { registry_key, name, exceptions }>`.
    Exceptions are `ContinuousEffectData` entries (Indefinite, the copy's
    timestamp, the copied definition as source).
  - v5 adds `FactKind::Copy` (7: id `name_value("CopyEffect")`, source = the
    copied definition + 1, order = the copy's timestamp rank) and
    `FactKind::CopyException` (8: encoded like `Effect` rows through
    `effect_facts`). Native and Python encoders match (`FACT_KINDS` 8).
  - The permanent row's definition index stays the copied definition.
  - Exceptions tokenize through `CreateTokenCopy`'s serialized
    `Modification` values.
  - LLM text: "copy of D<n>, except isn't legendary". Academy note: "copy of
    <name>, except isn't legendary".
  - Experience protocol: `copy_of: CopyView`. The wire name avoids pydantic's
    `BaseModel.copy`.
- **Turn history.**
  - `PlayerTurnHistory { permanents_sacrificed }` on `TurnState.history`
    uses T4's names. Writes go through `Game::update_turn_history`
    (journaled), incremented in `sacrifice_permanent`, and the history resets
    each turn.
  - It is hashed only when non-empty, so sacrifice-free state hashes are
    unchanged.
  - Observed in `PlayerData.turn_history`, a v5 player column
    (`PLAYER_SACRIFICED_INDEX`, width 63 → 64), the protocol
    (`PlayerTurnHistoryView`), LLM text ("This turn: sacrificed N
    permanents.") and Academy's PlayerArea.
  - `StaticCondition::SacrificedThisTurnAtLeast` reads it.
  - `cards_drawn_this_turn` stays main's field. T4 folds it into the struct.
- **Airship.** `StaticAbility::CreatureWithNamesakes(8)` is read in
  `permanent_is_animated` and counts the controller's permanents with this
  permanent's `permanent_name`, so no name enters the tokens. The trigger is
  `ActiveIf { SacrificedThisTurnAtLeast(1), BeginningOfYourEndStep }` with an
  `IfCondition` recheck (CR 603.4).

## Concept review (headless, 2026-10-30, HEAD `00724828c`) — nothing to change

Checked the code against Jack Heart's design review (comment 10a31948). The
copy authority is `create_token_copy`, which stores `CopyEffect { of,
exceptions }`. Exceptions use T3's `Modification` and are observed through
`FactKind::Copy` / `CopyException`. Turn history lives on `TurnState` in T4's
shape. The Airship is built only from existing condition, trigger and crew
values plus one self-relative static. No product choice is open.

Considered but not adopted: folding `CreatureWithCounters` and
`CreatureWithNamesakes` into one `CreatureWhile(<self condition>)` variant.
Both cover "this is an artifact creature as long as …", but the change would
re-key an existing card's learning tokens and no accepted decision calls for
it. If a third self-animation condition appears, merge them then.

## Remaining

- Done at publish (2026-10-30): w17 evidence re-recorded in place for T5's
  content digest. Semantic kernel: same 14 cases / 3401 commands, only
  state hashes and content digests moved. Also re-recorded: authored-match
  receipt, with its SHA pinned in both parity modules; authored-match parity
  (142 commands); public-commitment parity (69 commitments, no gaps); Academy
  demos and choice navigation (same decision revisions); trigger-order
  digests; belief v3 receipt pins.
- Sync onto T4 488a40 when it publishes. On that sync:
  - `PlayerTurnHistory` takes T4's other fields.
  - `cards_drawn_this_turn` moves into the struct.
  - The player column moves to T4's `PLAYER_SACRIFICED_INDEX` position, after
    attacked.
- Pre-existing on T3 `08bf54de2`, not from this change: 4 of
  `tests/model/test_action_identity.py::test_policy_choice_can_learn_either_preference`
  fail (target cases); they fail identically on T3's worktree.

## Delete — do not maintain

- `Effect::CreateNonlegendaryCopy` (main) and `CreateTokenCopy { nonlegendary }`
  (this branch's first slice): replaced by `{ exceptions }`. Done.
- This branch's duplicate `ActivatedAbilityDefinition.sorcery_timing`: T3's
  is the one field. Done.
- The tensor-only "copy identity" model (shared key plus card-row bits as the
  only signal): superseded by `CopyEffect` and its fact rows. Done.

## Checks

- 2026-10-30 (realign, HEAD `b7ddfc29c`): debug `cargo test --test
  semantic_conformance_tests` passes 4 and fails 2 on "ur-vs-gw-5eed content
  digest drift", as expected before gate re-records. Earlier passes: debug
  `cargo test` (cards 408, rules 137, lib 122), clippy, Python parity/text/
  protocol (55), Academy vitest (13) and svelte-check — see `af0c01977`.
