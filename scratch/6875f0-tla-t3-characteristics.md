# T3 — characteristics, counters, replacements, attachments (Task 6875f0)

Brief: the Task directive (49 cards, obligations in `docs/rules/tla/allocation.jsonl`,
`planned_pr == "T3"`). Stacked on T1 (892c47, PR #394; last synced at head `01e8f9975`)
and carrying T2 at PR #398's old head `16dfd3dbb`. Every merge needs Jack Heart's review.
Card tests live in `managym/tests/cards/tla.rs` (steer, Jack Heart 2026-10-29).

## Done (49 of 49)

Per-item detail for items 1–22 is at
`d0c3e3a8e:scratch/6875f0-tla-t3-characteristics.md`; per-slice code in
`git log 86ebe9ccf..`.

1–11. Conditional entry, static grants, counters by kind (`CounterKind`, stun
   in `untap`, exhaust), 7b base P/T, Equipment and Aura attachment with SBAs
   704.5m/n, Vehicles and crew, state-dependent mana, `ScopeView` layered
   characteristics, gained land types and red-mana replacement, X costs.
12–16. Excess damage and fight, kept red mana, tutor and library-top casting,
   granted abilities from attachments (`activated_abilities_for`,
   `AbilityGrant`), Honest Work's `set_characteristics` (one
   `has_lost_abilities` reader), one tap authority (`permanent_can_tap`),
   shared-creature-type targets.
17. T2 sync onto PR #398 head `16dfd3dbb`: one X announcement, exhaust gate,
   `ZoneEntry`; world w17. 18. Glider Staff, Crashing Wave.
19–20. Fire Nation Palace; observation-only fields for counters by kind,
   attachment, set name/types, granted land, firebending, stack X and spell
   grants (PyO3, protocol schemas, Academy types), then Academy and LLM text.
21–22. Card tests in `tests/cards/<set>.rs` with shared `cards/support.rs`.

`semantic_conformance_tests` fails on w15 receipts (world-release evidence,
as on T1).
One intermittent `structured_offers` failure is recorded in `scratch/questions.md`.

## Accepted model (concept reviews, 2026-10-29)

A permanent's characteristics are what `Game` derives from them (`ScopeView`
layers, `has_lost_abilities`, `permanent_name`, `permanent_has_subtype`,
`activated_abilities_for`). `Permanent` and `Card` fields are inputs, never
answers. Observations stay complete (acceptance; Jack Heart's 2026-10-21
shared-decision-surface direction), so item 19 added counters by kind,
`attached_to`, effective name and creature types, granted land, stack X, spell
keyword grants and granted firebending, following T2's `CardData.face_down`
precedent. Frozen widths are unchanged.

Item 20 (`a1d67b416`) gives people and LLM bots the same fields as labels only:
`permanent-notes.ts` (counters by kind, "attached to X" / "X attached" across
seats via `tablePermanents`, set creature types, granted land, firebending) feeds
`Card`'s accessible name and margin chips, and the ordered stack shows "X = n".
`text_observation.py` appends the same fields only when they are present, so
existing text is byte-identical. `_characteristics` in the projection omits on
`None`, not falsiness. Academy checks use nvm node 24 (Homebrew node 25.8 lacks llhttp). TLA card
names left in `tests/rules/` are CR-rule witnesses or deck filler, not
single-card tests (concept review after item 22).

## Delete — do not maintain

Qualification entries, `Permanent::local_power`/`local_toughness`,
`Permanent::can_tap`/`producible_mana`,
`StaticScope::Equipped`/`YouControlWithCounters`, play.rs `MAX_X`,
`Game::equipped_creature` and `Permanent::can_attack`
are gone; do not restore them.

## Accepted: the policy input shows everything (Jack Heart, 2026-10-29)

Jack Heart, at demo: "clearly the observation (and the tokenized representation)
need to show everything." This supersedes item 19's "no policy tensor column"
choice and the open PR-review question (concept review page
`scratch/concept-review.html`, options 1a and 2b).

Contract: the `Observation` is complete, and so is the policy input. New engine
state reaches the native and Python policy encoders in the change that adds it.
A frozen version never silently falls behind; older versions stay byte-frozen
for their checkpoints.

Jack Heart's later 2026-10-29 steers widen it: every mechanic is observable in
all three encodings (policy tensor, learning program tokens, LLM text), with an
exhaustive guard, a directed test per mechanic and viewer-safety tests; cards
become engine-type definitions (no authoring layer) on T1's sync, each with a
Scryfall id.

23–25. Policy input v5 (2026-10-29), the new default; v1–v4 byte-frozen.
   Layout constants live in `observation_encoder.rs` (Python mirrors them;
   WORLDS.md w18 lists widths); `ContinuationFrame` gains `x` and
   `granted_keywords`. The model joins attachment and blocking columns
   (`PERMANENT_LINK_INDEXES`), not numeric features. Item 24 added the
   state the `Observation` lacked (attacking, blocking, abilities lost, used
   abilities, kept red mana, no maximum hand size); item 25 shows it in
   Academy (`permanent-notes.ts`, PlayerArea) and the LLM text with the
   same words, and adds directed LLM-text tests (Foretell, foretold casts,
   Hakoda's library-top casts).

26. Sync onto T1's one card representation (`01e8f9975`, 2026-10-29; Jack
   Heart's steer). Every T2+T3 card and authoring `Op` is now an engine
   `CardDefinition`/`Ability`/named `Effect` value: set files regenerated with
   T1's `scratch/conversion/generate.py` (extended for `ManaUse`, restricted
   mana, `{X}` and all `state` modules) from T3's lowering; all 254
   definitions Debug-identical to it. 93 new Scryfall ids fetched. World w18
   (T1's w17 is the representation). v1–v4 stay 28 wide; T1's v4 `attacking`
   column moves to v5. Token closure walks serialized `token_name`s.
   `SetCharacteristics.name` became `named` (tokens never carry a `name`
   field). Tests now use T1's `tests/support/{game,play,inspect}.rs` (added:
   `five_color_unfunded_setup`, `move_to_hand`, `cast_card`, `castable`,
   `play_land`, `can_activate*`, `activate_card`, `has_action`,
   `offered_targets`, `counters_of`); `observability.rs` joins T1's token
   guards (106 `Effect`, 16 `TriggerCondition` witnesses, `Permanent`/`Player`
   fields at v5 columns) with T3's v5 tensor guard.

## Remaining

- **Sync onto T2's new head (PR #398 `d218e1dc2`, observed 2026-10-29).** T2
  synced onto the same T1 head (`01e8f9975`) and independently converted its
  cards, then shipped its own w18 and policy input v5: player 60, card 55,
  permanent 32 (v1–v4 keep 40 / 41 / 29), new terms at the old validity column,
  validity last, native buffers bound by player width. This branch's v5 is
  44 / 42 / 50 with v1–v4 at 28 and validity at column 31. Both carry stack X,
  granted firebending, used exhaust and no maximum hand size. A trial merge
  conflicts in 31 files (v5 encoders, observation, cardsets lea/stx/tla,
  `play.rs`, `observability.rs`, `tests/cards/tla.rs`, WORLDS.md, AGENTS.md,
  protocol schemas, text observation, PlayerArea). Since T2 lands before T3,
  the merge must take T2's card conversion and v5 layout and re-add only
  T3's columns; CLAUDE.md's v5 width line, WORLDS.md and the observability guard
  follow. **Decision needed (Jack Heart):** extend T2's unfrozen v5 and w18 in
  place (simpler; no checkpoint exists for either), or give T3 v6 and w19 so T2's
  v5 freezes when it lands. Not yet decided; the permanent-row width
  disagreement (T1/T2 widened v1–v4 to 29 for `attacking`; this branch kept 28)
  is part of it.
- World release items (w15 conformance receipts) and browser proof.

Check (item 26, debug): `cargo test --no-fail-fast` passes except the w15
`semantic_conformance_tests` pair; clippy `--all-targets -D warnings` clean.
pytest: model+semantic+verify parity 374, CLAUDE.md world suite 66, Academy
play/protocol 66, env+verify 182 pass.
Compress (after 26): one keyword-column table (`keyword_columns`) in
`observability.rs`; `tla.rs` uses the shared support helpers. `cargo test --test
cards` 378 pass, clippy clean.
