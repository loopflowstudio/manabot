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

## T3 (6875f0) assumptions — 2026-10-29

- Support at this base still reads `qualifications.json`; the four new lands get
  entries there so the world supports them. Jack Heart's steer says tests never
  gate support; those entries go away when T1's gate removal is synced.
- Per the same steer, only this batch's new tests went to `managym/tests/cards/`;
  moving T1's existing per-card tests is left to T1/PR #394 to avoid conflicts.
- Conditional entry reads the card owner's battlefield (cards enter under owner
  control in every current path).
- Learning schema 8 is unreleased (new in T1), so new vocabulary is appended in place.
- `semantic_conformance_tests` fails at this base (WORLD_VERSION w16, evidence w15);
  left to T1's world release rather than regenerated here.
- Zookeeper's creature-type count excludes a fixed noncreature-subtype list rather
  than a creature-type registry (none exists); revisit if a pool card breaks it.
