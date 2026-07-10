# Card-Conformance Audit (2026-07-09)

Full audit of every card registered in `managym/src/cardsets/` against
Scryfall (Oracle-Cards bulk data, fetched 2026-07-09), at three levels:

- **Level 1 — shell**: name, mana cost, colors, type line (supertypes,
  types, subtypes verbatim), printed P/T, engine-supported keywords, and
  `text_box` vs oracle text. All mismatches fixed.
- **Level 2 — semantics**: does the registered implementation do what the
  oracle text says? Classified CONFORMANT / SIMPLIFIED-SANCTIONED (listed
  in the wave doc's deviation section) / DEVIATION (documented) / STUB.
- **Level 3 — tripwire**: a committed Scryfall fixture + a cargo test that
  keeps every current and future registration shell-conformant.

## Totals

| Category | Count |
|---|---|
| Real cards registered (in fixture) | 55 |
| Level-2 CONFORMANT | 45 |
| Level-2 SIMPLIFIED-SANCTIONED | 9 |
| Level-2 DEVIATION (documented, wave doc) | 1 (Ancestral Recall) |
| Level-2 STUB | 0 |
| Level-1 shells fixed by this audit | 20 registrations (details below) |
| Token definitions (`is_token`, excluded from fixture) | 2 (Ally, Clue) |
| Invented cards (`// not a real card`) | 1 (Crossroads of Destiny) |

## The worst finds (all fixed)

1. **Glider Kids** — registered `{1}{U}` 2/1 Human Ally with no flying.
   Real card: **{2}{W} 2/3 Human Pilot Ally with Flying** (ETB scry 1 was
   the only correct part). Wrong cost, wrong color, wrong P/T, missing
   keyword, missing subtype. Fixed; flying now trace-tested
   (`glider_kids_flies_over_ground_blockers`). Note: the card is white —
   it had been sitting in UR test decks; test decks got Plains.
2. **"Grey Ogre" does not exist** — the Alpha card is **Gray Ogre**.
   Renamed across the engine, tests, python configs, gui server, bench
   script, and wave docs (frontend test mocks left for the concurrent
   frontend wave).
3. **Waterfall Aerialist** — registered `{2}{U}` 2/4 Djinn. Real card:
   **{3}{U} 3/1 Djinn Wizard**. The ward trace tests were rewritten (a
   resolved Bolt now kills the 3/1 instead of marking damage on a 2/4).
4. **Ancestral Recall** — oracle is "**Target player** draws three
   cards."; the engine self-draws with no target. Unfixable without new
   target-player-draw machinery — documented as a deviation in the wave
   doc; text box now carries the real oracle text.
5. **Raging Goblin** — missing its **Berserker** subtype.
6. Thirteen `register_creature` cards had **empty text boxes** (oracle
   prints keyword lines); Shivan Dragon's text box was missing entirely
   (Flying + firebreathing); Man-o'-War / Accumulate Wisdom texts were
   stale vs current oracle wording ("enters", reworded Wisdom).

## Level-3 tripwire (permanent)

- `managym/tests/fixtures/scryfall_cards.json` — committed snapshot
  (name, mana_cost, type_line, power, toughness, keywords, oracle_text,
  colors) for all 55 real registered cards.
- `managym/tests/conformance_tests.rs` —
  `every_registered_real_card_matches_scryfall` iterates
  `CardRegistry::definitions()` (new accessor) and checks each
  non-token, non-invented registration against the fixture: mana cost
  (canonicalized pip multiset; Scryfall's color-wheel pip order is
  cosmetic), colors, type line, printed P/T (`"*"` ⇔ `power_cda`),
  supported keyword flags (flying, reach, haste, flash, vigilance,
  trample, first/double strike, deathtouch, lifelink, defender, menace,
  hexproof) plus ward/kicker presence, and `text_box` vs `oracle_text`.
  **A registration without a fixture entry fails** with instructions;
  stale fixture entries also fail. `not_real_cards_are_registered`
  keeps the invented-card allowlist honest.
- Text normalization (documented in the test): reminder text `(...)` is
  stripped from both sides (registrations may include or omit it),
  except when the oracle text is *only* reminder text (basic lands),
  where just the parens are dropped; lines are trimmed, blank lines
  dropped, spaces collapsed; ability-paragraph newlines preserved.
- Sanctioned shell exception, explicit in the test: Suki's hybrid
  `{2}{G/W}{G/W}` accepted against the engine's `{2}{G}{W}`.
- `scripts/refresh_card_fixture.py` (uv-run) re-fetches via Scryfall's
  **bulk-data** endpoint (one ~180 MB download, no per-card calls — the
  per-card API rate-limited immediately), parses the cardsets for
  registered names, skips `is_token: true` blocks and
  `// not a real card` markers, and rewrites the fixture. Exits nonzero
  if a registered name isn't on Scryfall.

Semantic conformance is not auto-testable; its proxy is the per-card
trace-test suite (`managym/tests/rules/`: stage2_cards, stage3_cards,
tla_cards, cr_702_keywords, interaction_spells). The one behavior fixed
by this audit that lacked a trace test — Glider Kids' flying — got one.

## Per-card table

L1 = shell after this audit (FIXED = this audit changed the shell).
L2 = semantic classification. Sanctioned deviations reference the wave
doc list (`wave/rules/01-two-deck-slice.md`, Stage-3 RESULT + additions).

| Card | Set file | L1 shell | L2 semantics | Notes |
|---|---|---|---|---|
| Plains / Island / Swamp / Mountain / Forest | alpha (basics) | OK | CONFORMANT | oracle text is all-reminder; normalization keeps content |
| Llanowar Elves | alpha | OK | CONFORMANT | `{T}: Add {G}` mana ability |
| Gray Ogre | alpha | FIXED (name was "Grey Ogre") | CONFORMANT | vanilla |
| Lightning Bolt | alpha | OK | CONFORMANT | "any target" = creature/player; engine has no planeswalkers/battles |
| Ancestral Recall | alpha | FIXED (text box → oracle) | DEVIATION (documented) | engine self-draws; no target-player draw machinery; wave-doc listed |
| Counterspell | alpha | OK | CONFORMANT | |
| Wind Drake | alpha | FIXED (empty text box) | CONFORMANT | |
| Giant Spider | alpha | FIXED (empty text box) | CONFORMANT | reach |
| Raging Goblin | alpha | FIXED (+Berserker subtype, text) | CONFORMANT | |
| Serra Angel | alpha | FIXED (empty text box) | CONFORMANT | |
| Typhoid Rats | alpha | FIXED (empty text box) | CONFORMANT | |
| War Mammoth | alpha | FIXED (empty text box) | CONFORMANT | |
| Wall of Stone | alpha | FIXED (empty text box) | CONFORMANT | |
| Boggart Brute | alpha | FIXED (empty text box) | CONFORMANT | |
| Youthful Knight | alpha | FIXED (empty text box) | CONFORMANT | |
| Fencing Ace | alpha | FIXED (empty text box) | CONFORMANT | |
| Healer's Hawk | alpha | FIXED (empty text box) | CONFORMANT | |
| Craw Wurm | alpha | OK | CONFORMANT | vanilla |
| Shivan Dragon | alpha | FIXED (text box added) | CONFORMANT | flying + `{R}: +1/+0` |
| Pyroclasm | ice_age | OK | CONFORMANT | |
| Man-o'-War | visions | FIXED (text → "enters") | CONFORMANT | ETB bounce, may target itself |
| Pop Quiz | strixhaven | OK | SIMPLIFIED-SANCTIONED | learn = discard→draw only (no sideboard, 1v1) |
| Igneous Inspiration | strixhaven | OK | SIMPLIFIED-SANCTIONED | learn, as above |
| Divide by Zero | strixhaven | OK | SIMPLIFIED-SANCTIONED | learn, as above; spell-or-permanent bounce conformant |
| Waterfall Aerialist | strixhaven | FIXED ({2}{U} 2/4 Djinn → {3}{U} 3/1 Djinn Wizard) | SIMPLIFIED-SANCTIONED | ward triggers on spells only (wave doc) |
| Kyoshi Warriors | tla | OK | CONFORMANT | ETB 1/1 white Ally token |
| Avatar Enthusiasts | tla | OK | CONFORMANT | other-Ally-ETB counter |
| Invasion Reinforcements | tla | OK | CONFORMANT | flash + ETB token |
| Jeong Jeong's Deserters | tla | OK | CONFORMANT | ETB counter on target creature |
| South Pole Voyager | tla | OK | CONFORMANT | this-or-another-Ally ETB; 2nd-resolution draw |
| Tiger-Seal | tla | OK | CONFORMANT | upkeep tap; 2nd-draw untap |
| Otter-Penguin | tla | OK | CONFORMANT | 2nd-draw +1/+2 & unblockable |
| Forecasting Fortune Teller | tla | OK | CONFORMANT | ETB Clue token |
| Glider Kids | tla | FIXED ({1}{U} 2/1 no-fly Human Ally → {2}{W} 2/3 flying Human Pilot Ally) | CONFORMANT | flying trace test added |
| Firebending Lesson | tla | OK | CONFORMANT | kicker: 5 instead of 2, same target |
| It'll Quench Ya! | tla | OK | CONFORMANT | counter unless {2} |
| Accumulate Wisdom | tla | FIXED (text → current oracle) | SIMPLIFIED-SANCTIONED | bottoms rest in random order ("any order" on oracle; no reorder sub-decision, wave doc) |
| Water Tribe Rallier | tla | FIXED (reminder text verbatim) | CONFORMANT | waterbend activated ability; oracle itself says "random order" |
| Allies at Last | tla | OK | CONFORMANT | affinity via `cost_reduction_per`; power damage to last target |
| Badgermole Cub | tla | OK | CONFORMANT | earthbend 1 ETB; triggered mana ability is controller-scoped (verified) |
| Fire Nation Cadets | tla | OK | SIMPLIFIED-SANCTIONED | firebending = on-stack conditional attack trigger (wave doc) |
| First-Time Flyer | tla | OK | CONFORMANT | conditional +1/+1 static |
| Dragonfly Swarm | tla | OK | SIMPLIFIED-SANCTIONED | ward spells-only; CDA printed off-battlefield (wave doc) |
| Compassionate Healer | tla | OK | CONFORMANT | becomes-tapped trigger |
| Earth Kingdom Jailer | tla | OK | CONFORMANT | up-to-one optional target (Decline verified via `Effect::target_optional`); CR 603.6e pragmatics documented |
| White Lotus Reinforcements | tla | OK | CONFORMANT | anthem (verified controller-scoped) |
| Earth King's Lieutenant | tla | OK | CONFORMANT | `PutCountersOnEachMatching` verified controller-scoped |
| Suki, Kyoshi Warrior | tla | OK (hybrid cost allowlisted) | SIMPLIFIED-SANCTIONED | {G/W} as {G}{W}; legend rule unenforced (wave doc) |
| Yip Yip! | tla | OK | CONFORMANT | +2/+2; Ally rider grants flying |
| Fancy Footwork | tla | OK | CONFORMANT | per-target legality (CR 608.2b) |
| Enter the Avatar State | tla | OK | SIMPLIFIED-SANCTIONED | keywords granted, Avatar type not added (wave doc) |
| Crossroads of Destiny | tla | n/a — `// not a real card` | n/a | invented modal-machinery proof card; excluded from fixture |
| Ally (token) | tla | n/a — `is_token` | CONFORMANT | matches TLA's 1/1 white Ally token |
| Clue (token) | tla | n/a — `is_token` | CONFORMANT | `{2}, Sacrifice: draw` |

## Engine-wide simplifications touching many cards (pre-existing, wave doc)

- Scry keeps relative order; look/scry bottom moves are silent reorders.
- Ward fires on spells only (no ability targeting an opponent's
  permanent exists in M1).
- No hybrid mana, no legend rule, no planeswalker/battle targets.

## Validation

- `cargo test` (managym): all green, including the 2 new conformance
  tests and the new Glider Kids flying trace test.
- `cargo clippy --all-targets`: clean.
- `pytest`: 217 passed (cp312 extension rebuilt after the cardset
  changes).
