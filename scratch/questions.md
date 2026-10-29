# TLA direct-Rust decisions

Reversible operator choices made headless (2026-10-28 unless dated); each is
noted in the PR. Jack Heart's 2026-10-29 direction superseded the qualification
and pool-membership choices that were here (history: `be2b7586b:scratch/questions.md`).

- Old worlds run from their recorded git revision; no in-process rules shims.
- No w17: main is w15 and w16 is this PR's unreleased world, so renames, basics
  moving to `lea.*`, and support-by-definition all stay in w16.
- IR keeps the retired compiler's JSON shape, built in Rust on `serde_json::Value`
  (made byte-identity provable); lowering straight to `Step` is a later option.
- Learning tokens come from `learning.py` over that IR; schema 8 unchanged.
- 2026-10-29: the 14 setup-used incomplete TLA cards keep their definitions and
  are listed in one `tla::INCOMPLETE` const (key, missing behavior) beside
  `cards()`, rather than a field on all 269 `Card` literals. Native cards sit in
  set files as native `CardDefinition`s and are unsupported (no IR, so not in
  `pool_content`). Crossroads of Destiny moved to `cards/fixtures.rs` but stays
  in the default pack, because Rust integration tests and `manabot/verify` build
  games from it. D1 keeps the `tla-pool-v1` key; `set_cards` is the set filter.
  Undefined TLA cards are no longer listed as unsupported pool rows.
- Not rewritten (dated evidence): the Academy curated pack `local_identity` values
  keep pre-rename keys.
- Full TLA completion remains the eight-PR plan; no merge or new Task authorized.

Main's MTG-193 notes at the sync live at `6d069d6e5:scratch/questions.md`.

## T2 (87e012) — 2026-10-29 operator choices

- When printed flashback and Iroh's grant both apply, the engine uses the
  grant's cost instead of offering both. True dominance holds for every TLA
  card; a non-dominated alternative would need a cost choice.
- Learning schema 8 extended append-only (Plant, Rogue, `flashback`) following
  `1e0d3c998`. The first new opcode needs learning schema 9; not yet taken.
- Slice 1 avoided a new opcode: landcycling is `search_basic_land` with
  `destination: hand` (searching any card of the subtype, not only basics),
  graveyard self-return is `return_to_hand` targeting `source`. If reviewers
  prefer a distinct `search_library` opcode, that is schema 9.
- Card abilities get their own action variant (`ActivateCardAbility`) sharing
  the `PriorityActivateAbility` type, rather than widening `ActivateAbility`
  to an `ActivationSource`; internal payment/target state uses the shared
  `ActivationSource`.
- A discard-cost ability refers to its card's graveyard object after costs
  (Team Avatar's damage source), standing in for the hand object's LKI.
- World stays w16 while T2 only adds cards and flashback costs; bump when a
  slice changes existing rules meaning (one unreleased world per Task).
