# MTG-431 — Foundations stack copies, choices, costs and procedure tail

All 56 of the 56 scoped Foundations rows have directed implementation evidence.
Slices 1–10 are complete. Demonic Pact and Maze’s End already had working
definitions.

**Next step: Gate / Integrated validation** — run the full debug suite, rebuild
the extension and Academy fixtures, run current-world admission and pinned-pilot
checks, Engine Budget, stress states, and the cross-card synergies below.

Remaining passes (56 implemented, 0 pending): none.

Sorting (checked against `managym/state/` and the engine, 2026-11-24, base `cf73d9778f`):

- **25 compose from today's types** (23 in Slice 9, plus Bolt Bend and Teach
  by Example in Slice 5).
- **28 wait on a shared primitive** (Slices 1–8; Curator and Biogenic Upgrade
  each need a new decision shape, the rest are small).
- **Abyssal Harvester** builds on MTG-420's copiable-snapshot authority. PR #593
  merged 2026-11-25T03:17Z (`45ddf3bca7`) and this branch merged it at
  `72c35dc5bc`, so Slice 10 is unblocked. Its ledger row still names
  `blocked_by: MTG-418` (graveyard arrivals from any zone); Slice 10's
  per-turn graveyard record is that query, so clear the field when delivered.

No card here waits on authority owned by MTG-430 (planeswalkers) or MTG-432
(characteristics, replacements and combat constraints). Fiery Annihilation
uses the existing `Modification::ExileIfWouldDie`. Gnarlid Colony uses the
existing `EntersWithPlusOneCounters`.

## First printings (set file, Scryfall id)

grn: Thousand-Year Storm `270a0863-7d07-43f0-925d-a8ce0383a1cb`, Affectionate Indrik `b4c8ddc1-d95c-499f-b1d1-f608f8f07b02` ·
war: Bolt Bend `39b35408-3728-4e1b-9f58-b0775df914d6`, Finale of Revelation `6630c34a-1a97-4e31-9d2c-1150b0aa903e` ·
stx: Teach by Example `aa7fbb9b-50a8-4d18-a667-fe965468ca16`, Deadly Brew `87d33e48-90fc-4aac-b09a-68050bc053b5` ·
eld: Run Away Together `aeffc3c0-567c-442f-ba06-b7d9617c5789`, Heraldic Banner `2e349af5-3f25-46d3-908e-83b2f6028b95`, Charming Prince `dcb94950-3f3e-4876-84f8-d5e4d9cfecee`, Wishclaw Talisman `07c17b01-ee5d-491a-8403-b3f819b778c4` ·
som: Genesis Wave `c920236f-c3d7-421c-b021-103996da790e` ·
neo: Secluded Courtyard `0539b1a5-8704-476f-ba1f-2fe01190e157`, Uncharted Haven `1d4ad89a-3a00-4bf4-a357-4a8a089d4a82` ·
rna: Biogenic Upgrade `0dd73fb2-453f-40b9-8beb-dfa99e6a706e`, Gate Colossus `99767e2f-a558-4d63-b9b6-923d15b433e1` ·
znr: Felidar Retreat `45340647-4d3e-4be1-b0e6-e40cc56a438b`, Gnarlid Colony `7327289d-eed8-44b1-8495-7172e2b49d5f` ·
tsp: Mystical Teachings `a057e5f3-b6bd-4995-b990-714201f36989` · isd: Tribute to Hunger `f77e0f88-2285-4b59-9165-9948c75d77a3` ·
scg: Dragon Mage `7687a201-0ecc-4739-86e3-3b4090d345a8` · xln: Sorcerous Spyglass `85506a24-8d60-475c-9f43-65994caca7d4` ·
bro: Demolition Field `d9c88546-13c9-4d7e-a618-cb2ccd1dbc0f` · m12: Adaptive Automaton `79e42ead-df6e-4181-ae2b-a2abfc3f1d7c` ·
m20: Drakuseth `d09af78f-efde-4107-8406-cb12fd11c686` · m21: Primal Might `1cd8cee8-7ea0-4037-8a3b-39334dc064fb` ·
m19: Diamond Mare `ca600b3f-2c70-489b-b218-6e3245b90114` · psal: Frenzied Goblin `f786de73-98ff-4426-abca-dc789f6848ce` ·
dom: Grow from the Ashes `51d4d1c2-671c-498c-a232-7d076e3dc3bb` · mor: Ballyrush Banneret `a029814e-d84d-43e5-b483-e918871b3333` ·
rtr: Desecration Demon `8242fade-754c-4404-b3fb-f3cccf84b3b6` · one: Ovika `b298cf34-7aa5-4f97-a86c-7f28d2113b87` ·
dtk: Savage Ventmaw `690008d1-d1fe-49ad-810c-84be57cecc6c` · blb: Three Tree Mascot `aaced75b-6e07-457c-8ea2-f74d99710d15` ·
zen: Gatekeeper of Malakir `71db3698-a45c-4eaf-87e6-30502c0c10f4` · mh1: Springbloom Druid `6161d2ed-7cff-4c90-9e74-1d179a6c1498` ·
frf: Flamewake Phoenix `fefd5848-9fe1-4129-a5d7-e51606bf76ef` · mbs: Slagstorm `9e318b03-2aad-462b-a2a9-8b6bdf0e93d6` ·
fdn: Divine Resilience `f3a08245-a535-4d24-b8c0-78759bb9c4b0`, Squad Rallier `65e1ee86-6f08-4aa0-bf63-ae12028ef080`, Cephalid Inkmage `b7e47680-18c7-4ffb-aac4-c5db6e7095ba`, Curator of Destinies `9ff79da7-c3f7-4541-87a0-503544c699b5`, Inspiration from Beyond `b636fe95-664f-4fb1-aab9-28856edeccd6`, Gutless Plunderer `909d7778-c7f8-4fa4-89f2-8b32e86e96e4`, Soul-Shackled Zombie `deea5690-6eb2-4353-b917-cbbf840e4e71`, Fiery Annihilation `54fe00aa-d284-48f9-b5a2-1bd4c5fa8e58`, Strongbox Raider `b2223eb8-59f9-489b-a3f3-b6496218cb79`, Sylvan Scavenging `c35b683c-d3b2-46a1-876a-81b34e8ba2fc`, Zimone `20ccbfdd-ddae-440c-9bc0-38b15a56fdd1`, Arcane Epiphany `06431793-5dfe-4cbf-990b-4bcc960d1f31`, Arbiter of Woe `b2496c4a-df03-4583-bd76-f98ed5cb61ee`, Vampire Gourmand `917514c0-9cd5-4b97-85b9-c4f753560ad4`, Koma, World-Eater `c3b92caa-3401-4b11-9515-152f3e057c05`, Perforating Artist `72980409-53f0-43c1-965e-06f22e7bb608`, Abyssal Harvester `f2e0f538-5825-47e9-883c-3ec6fd5b25ea`

Check each id and append a `managym/tests/fixtures/scryfall_cards.json` row as
`CLAUDE.md` describes. Frenzied Goblin's earliest paper printing is the
Salvat promo; `psal.rs` exists, so follow the repository's convention.

Vocabulary edits that are table data, not primitives:

- Subtypes: the complete Slice 1 vocabulary includes Kithkin and Nightmare.
- Counter kind: `CounterKind::Wish` (append; `record_vocabulary.rs`).
- Tokens in `cards/src/tokens.rs`: `token.raccoon_3_3_g` (Sylvan Scavenging),
  `token.cat_beast_2_2_w` (Felidar Retreat) and `token.phyrexian_goblin_1_1_r`
  (Ovika), each with its first token-printing Scryfall id.

## Finished slices (detail in git and the ledger)

1. Entry choices on one battlefield incarnation (Banner, Haven, Mare, Automaton, Courtyard, Spyglass) — `47e0e34eda`, `9e1bd7c4a8`, `1635633a40`.
2. Curator's opponent-chosen piles — `49417f76ad`, `cb84b1c0fe`.
3. Biogenic Upgrade's cast-time counter shares — `49417f76ad`, `cb84b1c0fe`.
4. `TargetRelation` and whole-assignment `Decision::Retarget` (Drakuseth, Fiery Annihilation, Run Away Together, Divine Resilience, Charming Prince, Affectionate Indrik) — `4e481dc0a1`.
5. Stack copies and retargeting (Thousand-Year Storm, Bolt Bend, Teach by Example) — `d73b18db4a`; stress `bench_states storm-copies --sizes 8`.
6. `LibraryTop`/`Unselected` selectors, `SelectCards.max_mana_value`, `Effect::GrantPlay` (Gutless Plunderer, Strongbox Raider, Genesis Wave) — `460fe18685`.
7. Last-known values and shared sacrifice (Tribute to Hunger, Deadly Brew) — `30fd60f499`.
8. Kicked entries, composite ward, `ManaExpiry`, Changeling, `CardPredicate.keyword`, `CounterKind::Wish` (Gnarlid Colony, Gatekeeper, Ovika, Ventmaw, Mascot, Teachings, Wishclaw, Ballyrush) — `460fe18685`.
9. 23 compose-only cards (`token.raccoon_3_3_g`, `token.cat_beast_2_2_w`, `Effect::double_counters`, graveyard trigger `intervening_if` and `source_present` fixes for Flamewake Phoenix and Gate Colossus) — `1eb342e57b`.
10. Abyssal Harvester (`TurnState::graveyard_arrivals`, `CardPredicate.put_into_graveyard_this_turn`, `Except::CreatedThisWay`, `CardData.put_into_graveyard_this_turn`, v11 `card::PUT_INTO_GRAVEYARD_THIS_TURN` column 56) — `1eb342e57b`; `blocked_by` cleared on its ledger row.

Known defect for the gate:
`resolution_tests::retained_creation_precedes_allocation_and_preserves_copy_entry_terms`
fails (`resolution_tests.rs:3866`, a token not on the battlefield) at
`e0baeed8e8`, before Slice 4, and the same way after it. The full debug run
must fix or explain it.

## Integrated validation

- Rust: `CARGO_BUILD_JOBS=2 cargo nextest run --profile ci --test-threads 2 --manifest-path managym/Cargo.toml`
  in debug before landing; `cargo fmt --all`.
- Rebuild the extension and refresh Academy fixtures
  (`scripts/refresh_academy_fixtures.py`).
- Current-world admission:
  `uv run --extra dev --extra artifacts --extra reports pytest tests/semantic/test_elemental_world.py tests/eval/test_evaluation_models.py tests/eval/test_mirror_admission.py tests/training/test_monitor_evaluation.py tests/model/test_choice_separability.py -q`,
  plus `tests/model/test_pinned_pilot.py`.
- Stress states through `bench_states`; Engine Budget stays within 10%:
  8–16 Courtyards with distinct mana restrictions and eight differently
  chosen Banners or Automatons (payment and layers); `storm-copies --sizes 8`;
  Genesis Wave X = 16; eight Ventmaw-style pools; Changelings under eight
  chosen-type anthems.
- Synergies to exercise together:
  - Thousand-Year Storm with Teach by Example (a Teach copy is not cast, so
    it never raises Storm's count).
  - Bolt Bend redirecting a Drakuseth trigger's single-target copy is not
    legal; it is legal on Fiery Annihilation only if that spell has one
    target.
  - Three Tree Mascot with Adaptive Automaton, Secluded Courtyard and
    Ballyrush Banneret.
  - Sorcerous Spyglass naming Three Tree Mascot leaves its mana ability
    working; naming Wishclaw Talisman stops it.
  - Genesis Wave putting Heraldic Banner and Uncharted Haven onto the
    battlefield simultaneously: each asks its entry choice.
  - Deadly Brew against Desecration Demon.
  - Gate Colossus with Maze's End.
- Ledger: update each delivered row with `status: implemented`, `key`,
  `evidence` (test path::name) and `result`, as High Fae Trickster's row does.

## Ledger status

All 56 scoped rows are `implemented` (0 pending), including Abyssal Harvester (`blocked_by` cleared).
