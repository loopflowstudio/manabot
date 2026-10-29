# T3 — characteristics, counters, replacements, attachments (Task 6875f0)

Brief: the Task directive (49 cards, obligations in `docs/rules/tla/allocation.jsonl`,
`planned_pr == "T3"`). Stacked on T1 (892c47, PR #394; last synced at head `e403b8a1f`,
merge `e5f26e23b`) and synced onto T2 (87e012, PR #398 head `16dfd3dbb`). Every merge needs Jack Heart's review.
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

Schema 8 (unreleased) is appended in place after T2's entries.
`semantic_conformance_tests` fails on w15 receipts, and the CLAUDE.md world
suite fails on `manabot/sim/registration.py` (w15 literal), frozen
`schema_version` asserts and w13 pins. All are world-release evidence.
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

23. Policy input v5 (2026-10-29), the new default; v1–v4 byte-frozen.
   - Layout (`observation_encoder.rs` owns it; `observation.py` mirrors):
     player 42 (turn/20 at 40, seat at 41); card 42 (face down at 40);
     permanent 42 (stun/fire/double-strike counters 27–29, attachment join
     30, characteristics set 31, granted land 32, firebending 33, temporary
     Plains/Island/Swamp/Mountain/Forest/Construct 34–39, other temporary
     subtypes 40); relation 48 (`ContinuationFrame.x` at 33,
     `granted_keywords` 34–47). Validity stays last (relation: 31).
   - `ContinuationFrame` gains `x` (stack, announcement, payment and
     resolution frames) and `granted_keywords` (stack spells), serde-omitted
     at defaults; `KeywordData::flags()` gives field order.
   - Model: the attachment column is a join (excluded from numeric features,
     like the definition); zero-initialized `attachment_to_host` /
     `host_to_attachment` exchange pre-attention embeddings both ways, v5 only.
   - LLM text adds "face down" and stack "granted keywords" (present-only).
   - Tests: `managym/tests/cards/observability.rs` (exhaustive guard; directed
     pairs v4 cannot tell apart; viewer safety); Python native = reference
     parity, attachment join, text guard over every pyclass field.

24. Missing observation state (2026-10-29): `PermanentData.attacking`,
   `blocking` (the attacker from `CombatState.attacker_to_blockers`),
   `abilities_lost` (`has_lost_abilities`), `used_abilities` (indices in
   `activated_abilities_for` whose `activation_used`), `PlayerData.
   keeps_red_mana` and `no_maximum_hand_size`; serde-omitted at defaults,
   PyO3 and `.pyi`. V5 edited in place: permanent 50 (attacking 41, blocking
   join 42, abilities lost 43, used-ability flags 44–47 for indices 0–3,
   later-index count 48), player 44 (kept red mana 42, no maximum hand size
   43). The model's link join covers both permanent links
   (`PERMANENT_LINK_INDEXES`, `link_to_target`/`link_from_target`). Text and
   both guards list them; directed Rust tests for attacking, blocking A vs B,
   lost abilities, spent exhaust and player flags.


25. Academy and LLM text for item 24 (2026-10-29): `LegacyPermanentView`
   gains `attacking`, `blocking`, `abilities_lost`, `used_abilities`;
   `LegacyPlayerView` gains `keeps_red_mana`, `no_maximum_hand_size` (Rust,
   Pydantic, six schemas regenerated, `types.ts`, projection omits at
   defaults). `permanent-notes.ts` labels "attacking", "blocking X", "has lost
   all abilities", "used-up abilities 1, 3" (the LLM text's words);
   PlayerArea states kept red mana beside the mana pool and "No maximum hand
   size" in the player bar. Directed LLM-text tests for the Foretell offer,
   casting a foretold card on a later turn and casting an Ally from Hakoda's
   library top; a live projection test for attack and block.

## Remaining

- After T1 (PR #394) publishes its authoring-layer removal: sync, convert
  this batch's cards to engine-type definitions (no new authoring `Op`s), add
  the learning-token half of the guard over `Effect`/`Ability`, and Scryfall
  ids. T1's plan now lists all three as its Remaining 1–3 (owner: T1). Still
  unpublished (2026-10-29): origin at `e403b8a1f`; T1's local `f9732c12f` also
  files card tests by set, so the sync reconciles it with items 21–22.
- World release items (registration literal stops at w15, w15 conformance
  receipts, frozen `schema_version`/digest asserts, w13 pins) and browser proof.

Check (items 23–24, debug, before the e403b8a1f merge): `cargo test --no-fail-fast`
passes except `semantic_conformance_tests` (w15 receipts); pytest 690 pass, 19 fail + 5 error, all
world-release evidence or failing at `752c75ef7~1` too. The merge touched only
`world.rs`: debug `cargo test --lib world::` 2 pass.
Compress (after item 25): encoder card/permanent rows via `chunks_exact_mut`;
the model's numeric permanent columns computed once. Debug `cargo test --lib
agent::` 31 and `observability` 18 pass; pytest tests/model 176 pass. Native
parity against a rebuilt `.so` is left to gate.
Compress rerun (no code change since): scratch trimmed of superseded "observations
don't show" notes and the stale support-map pointer.
Check (item 25): vitest 115 pass, `npm run check` 0 errors (nvm node 24); pytest
play modes, experience/study/advice/replay/server protocol 89 pass; text
observation 29 pass; debug `experience_protocol_tests` 5 pass.
