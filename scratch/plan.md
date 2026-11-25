# MTG-431 — Foundations stack copies, choices, costs and procedure tail

All 56 of the 56 scoped Foundations rows have directed implementation evidence.
Slices 1–15 are complete. Demonic Pact and Maze’s End already had working
definitions.

Every one of the 56 scoped ledger rows is `implemented`; Abyssal Harvester's
`blocked_by` is cleared. Slice 11's four concept defects, Slice 12's three
behavior-review defects, and Performance Slices 13–15 are built and verified.

**Next: Gate / Integrated validation** (below).

## Engine performance review at `7aed946d0d` (Apple M4 Max, 16 cores, contended host load ~6–23)

### Supported gains (uninstrumented release vs base `3be5a0cfb2`, C/R/R/C via `/usr/bin/time -l`)

- Identical counts and digests to both checkout base `3be5a0cfb2` and CI reference `7d3ac1eb1c` (`managym/perf/baseline.json`):
  - `bench_steps` (512 games, 118,559 steps, digest `10601622128491155944`): instructions retired `45.879B, 45.857B` (mean **45.868B**) vs base `46.754B, 46.734B` (mean **46.744B**, **-1.87%**); cycles `10.822B, 10.783B` (mean **10.802B**) vs base `11.003B, 10.965B` (mean **10.984B**, **-1.66%**); wall `2.70–2.75s` (43,092–43,895/s) vs base `2.74–2.79s`.
  - `bench_search` (seeds 1–3 digests `13638875039089371439`, `5466143068363283977`, `9998925852575961750`, 0 cap hits, identical `score_digest_blake3` and `action_counts`): instructions retired `977.25B, 977.43B` (mean **977.34B**) vs base `1,027.71B, 1,027.67B` (mean **1,027.69B**, **-4.90%**); cycles `226.44B, 228.74B` (mean **227.59B**) vs base `236.86B, 236.50B` (mean **236.68B**, **-3.84%**); wall `58.52s` / `68.02s` (contended spike) vs base `60.20s` / `59.94s`.
- Paired `/usr/bin/sample` profiles (`cand-search` 10,666 vs `base-search` 10,652 samples) confirm Slice 13's gain with no unprofiled regression on curated decks: `layers::Context::finish_object` `6.58% → 0.76%` (**-5.82pp**), `CreatureTypes::among` `4.48% → 0.68%` (**-3.80pp**), `_platform_memcmp` `3.38% → 0.68%` (**-2.70pp**), `layers::Context::evaluate` `32.84% → 28.38%` (**-4.46pp**), `Game::layer_board` `38.88% → 34.60%` (**-4.29pp**). `dbda3a656e` (`lf compress`) preserved and slightly improved retired instructions vs `a17915a437` (`45.920B → 45.868B` steps; `977.74B → 977.34B` search).
- Debug invariants verified: 300/300 debug `nextest` tests across layers, statics, mana, payment, changeling, courtyard, automaton, banner, mascot, spyglass, retarget, storm, genesis_wave and ventmaw pass (including the uncached layer oracle and payment direct/cache/scan assertions); `bench_states` self-tests pass in both release (`2/2`) and debug (`2/2`), and `entry-choices changeling-anthems --sizes 4,8,16` preserves exact digest `ffc4ac840ad0742b34d0a53609c610b799d6660e25102555709373f53893f446`.
- Ruled out / flat large-board shapes: `storm-copies` (`n=4,8`), `genesis-wave` (`n=4,8,16`), `ventmaw-pools` (`n=4,8`) scale linearly (`step:* ≤ 0.12 ms`, `search ≤ 5.6 ms`); `characteristics` on `entry-choices` and `changeling-anthems` stays ≤ `0.22 ms` at `size=16`.

### Negative results and open bottlenecks (touched large-board `bench_states`)

- Earlier `a17915a437` notes recorded only contended wall times (`1,098 ms` / `230 ms`) for `bench_states`, with no retired instructions or post-Slice-14 CPU profiles. Fresh load-invariant counters, `/usr/bin/time -l` retired instructions and direct-PID `/usr/bin/sample` profiles at `7aed946d0d` show **superlinear payment-search growth and credible profile-backed gains remain on both touched shapes**:
  - `changeling-anthems --sizes 4,8,16`: **80.55B instructions retired** (`19.44B` cycles — nearly 2× all 512 games of `bench_steps`). `cast-choice step:choice` grows superlinearly (`plans 12 → 20 → 36`, `calls 118 → 354 → 1,210` [3.00× / 3.42×; **10.3×** over 4× size], `states 86 → 262 → 902`, `0.10 → 1.26 → 7.39–9.69 ms`); `cast-choice search` grows `plans 2,552 → 7,738 → 19,310`, `calls 21,392 → 125,464 → 616,291` (**28.8×**), `states 15,251 → 91,935 → 457,360` (**30.0×**; `3.1–5.5s`). Profile (`size=16`, 4,005 samples): **78.3%** inclusive in `Game::mana_ability_actions` → `Game::activation_witness` (77.6%, running full plans for all 16 interchangeable `Three Tree Mascot`s, each mutating `Arc::make_mut(&mut model.sources)` and missing `PaymentCache`), **25.0%** inclusive (`20.5%` self) in `Sip13Rounds` hashing `Arc<PaymentSearchState>` (`17.15%` in `Obligation::hash_slice`), **18.8%** inclusive (`14.0%` self in `PaymentResources::cmp_source`) in `PaymentSearch::canonical` (4 `partition_point` binary searches per comparison), **15.4%** self `PaymentModel::witness`, **4.7%** self `PaymentModel::uses`.
  - `entry-choices --sizes 4,8,16`: **29.93B instructions retired** (`7.08B` cycles). `priority step:activate` grows `calls 155 → 384 → 944` (**6.1×**), `states 78 → 239 → 711` (**9.1×**); `cast-choice search` grows `plans 2,843 → 9,391 → 21,280`, `calls 13,409 → 57,356 → 186,044` (**13.9×**), `states 6,382 → 29,495 → 120,859` (**18.9×**; `0.43–1.79s`). Profile (`size=16`, 1,010 samples): **55.7%** inclusive in `Game::mana_ability_actions` → `Game::activation_witness` (53.0%, testing all 16 distinct-type `Secluded Courtyard`s separately even when their restrictions evaluate identically on active `SpendingContext`s), **~20.4%** self in hashing (`14.3%` `Sip13Rounds`, `2.3%` `DefaultHasher`, `2.2%` `ManaRestriction` hash, `1.7%` `PaymentResources::hash`), **11.7%** self `PaymentModel::witness`, **8.8%** self `cost_coverable`, **5.2%** self `uses`, **~5.2%** self in `PaymentCache` scan + `Vec<ManaActivation>` clone/eq.

### Completed performance Slice 15 (`managym/src/flow/mana.rs`)

15. **Purpose-projected equivalence in `mana_ability_actions` and `PaymentSearch::groups`, O(1) `PaymentResources` used-ability bitmasks, and fast non-cryptographic `visited` hashing** — complete.
    - *Implemented:* (a) `Game::mana_ability_actions` deduplicates feasibility checks per purpose-projected source equivalence class (same `untapped`, `waterbend`, `keyword`, convoker colors under convoke, tap-candidate membership, and `PurposeProjection`-projected `fixed`/`abilities` preserving `SpendEffect`/`ManaExpiry`), reusing one `PaymentModel`, preserving exact action order, and verifying every deduplicated source in debug builds via `debug_assert_eq!`; (b) `PaymentSearch::groups_for` and `CanonicalState` project `ManaRestriction`s onto reachable `SpendingContext`s (`PurposeMask` + interned `ProjectedChoices`) so identically-usable distinct-restriction Courtyards collapse into one canonical group and memo key; (c) `PaymentResources` uses O(1) per-source `u64` bitmasks (`used_masks`, `used_overflow` fallback for `>= 64` abilities) in `is_used`/`cmp_source`/`canonicalize`, `PaymentSearch::visited` / `PurposeSupply::allows` / `PurposeProjection` use a deterministic `FastHasher`, and `PaymentCache` fast-paths `Arc::ptr_eq` on `sources`.
    - *Verified:* Debug tests pass with all debug oracles active (`flow::mana::` 58/58 including 3 new tests, `bench_states` self-tests 2/2, `fdn::` cards 192/192). Uninstrumented release `bench_steps` (`46.07B`–`46.24B` vs base `46.744B`, **-1.1% to -1.4%**, digest `10601622128491155944`) and `bench_search` (`981.12B` vs base `1,027.69B`, **-4.53%**, identical seed digests, `score_digest_blake3`, and 0 cap hits) preserve exact digests and beat base. On `bench_states --sizes 4,8,16`: `changeling-anthems` drops **`80.55B` → `20.65B` retired instructions (-74.4%, 3.90×)** (`cast-choice step:choice` `plans 12→20→36` → **`4→4→4`**, `calls 118→354→1,210` → **`40→72→136`** linear, `states 86→262→902` → **`29→53→101`**; `cast-choice search` `plans 19,310` → **`5,074`**, `calls 616,291` → **`192,077`**, `states 457,360` → **`142,189`**); `entry-choices` drops **`29.93B` → `17.91B` retired instructions (-40.2%, 1.67×)** (`cast-choice step:choice` constant at `15 plans / 25 calls / 5 states`; `priority step:activate` `calls 944` → **`552`**, `states 711` → **`420`**; `cast-choice search` `plans 21,280` → **`9,800`** linear, `calls 186,044` → **`102,999`**, `states 120,859` → **`68,722`**). (`bench_states` `digest` moves because `payment.witness_digest` hashes fewer deduplicated `model.plan` calls while legal actions, choices, and state digests stay identical.)
    - *Deferred to gate/CI:* Ubuntu `x86_64` CI Engine Budget run (`engine-budget.yml`) and full workspace debug `nextest` / Python admission checks.

Behavior review at `6af8463ff9`: no open defects. The 16 debug failures found
at `1278923dc6` (15 frozen semantic-kernel traces, Azula fixture) and the two
LLM-text field failures are fixed; after the compress commits,
`tla::fire_lord_azula*` (2 passed) and `tests/semantic/test_text_observation.py`
(103 passed) still pass. Remaining nits, not blocking: Academy shows Retarget
as "Choose a target for …" and `KeepPermanents` as "Choose from the revealed
cards".

## Coverage decomposition at `47e21050c2`: no further slices

Review criterion: the representative cards compose from small orthogonal
primitives, with no card-specific engine logic, multi-axis special variant or
duplicate state. The batch meets it, so no refactoring slice is planned.

- Stack copies: Thousand-Year Storm is `Repeat { count:
  SpellsCastBeforeTriggering, effect: CopyTriggeringObject }`; Teach by Example
  is a delayed trigger over the same copy verb; Bolt Bend is the existing
  `ChangeTarget`. One copy owner (`CopiableSpell`), one retarget owner
  (`flow/retarget.rs`).
- Targets: `TargetRelation` on the clause, `ObjectSelector::ClauseTarget(n)`,
  `Simultaneously` and `Fight { first, second }` carry Drakuseth, Fiery
  Annihilation, Run Away Together, Indrik and Primal Might.
- Library and selection: `LibraryTop`, `Unselected`, `SelectCards.max_mana_value`
  and `GrantPlay { what, until }` carry Genesis Wave, Plunderer and Raider and
  replaced `ExileTopCardsToPlay` and `GrantGraveyardCastThisTurn`.
- Entry choices: one `Permanent.chosen: EntryChoice` read by predicates, mana
  output and the type layer. A second chosen-type land (cast-only) is
  `creature_purposes: [Cast]` with no engine change.
- Costs and values: `PaymentCost::All`, `ManaExpiry`, `CardPredicate.keyword`,
  `EachPlayerSacrifices { predicate }`.
- The engine diff names no card (checked by grep over added lines).

Single-consumer variants left as they are, because splitting them now would
guess at a second card. Split each when that card arrives:

- `Effect::SeparatePiles { look }` (Curator). The shape then is `SelectCards`
  over `LibraryTop` with `Selected`/`Unselected` as the piles and a `ForPlayer`
  pile choice.
- `TargetRequirement.up_to_if_kicked` (Divine Resilience).
- `ReplacementEffect::AsEntersChoose.look_at_opponent_hand` (Spyglass).
- `PrintedCharacteristics.division` (Biogenic Upgrade); it moves onto the
  divided clause when a spell divides among one clause of several.

## Follow-ups outside MTG-431 (separate Tasks)

- Retire `Effect::TargetCreaturesDealPowerDamageToLastTarget` (Felling Blow,
  Bite Down, Rocky Rebuke, Allies at Last) onto
  `DealDamage { by }` over `ClauseTarget`. Not done here: Allies at Last needs
  a per-source amount read, two TLA definitions change their learning tokens
  (pinned-pilot rollover), and `agent/heuristic/knowledge.rs` matches the
  variant by name.

- Semantic candidate-token encoding for `Decision::ChooseEntryValue` (Color and
  367 creature types): generalize `NameCandidateEncoder` to
  `RecordNamespace::Token` vocabularies across `managym`,
  `manabot/policy/prepare.py`, Torch and MLX.
- Retire `Effect::LookAndSelect` onto `SelectCards { from: LibraryTop }`, and
  replace `Effect::AddMana.until_end_of_combat` with `ManaExpiry` on
  definitions (pinned-pilot rollover).

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

11. Concept alignment: Retarget projects chosen `"new target"`s; one `Player.mana_pool` with `ManaExpiry` (combat pool deleted, additive `expiry` beside derived `combat`, `MATCH_STATE_HASH_VERSION` 33); group clicks skipped in decision memory; `PartitionPiles`/`ChoosePile` are `Modal`; `keep_permanents` stage (`keep_creatures` still read); Academy prompt lines — `e1f42473b0`.
12. Behavior-review defects: Azula fixture preserves firebending `EndOfCombat` mana alongside step `{R}`; `text_observation._cards_lines` renders `put_into_graveyard_this_turn` and `_TEXT_FIELDS` covers `Card.put_into_graveyard_this_turn` and `PooledManaClass.expiry`; verified all 14 frozen semantic-kernel traces match base `3be5a0cfb2` identically on Commands and event boundaries (only `state_hash` `v31`→`v33`, `pending_choice_digest`, and `content_digest` moved), re-recorded `evidence-battlefield-history-v1.json`, rebuilt the release extension, and regenerated the authored-match authority and parity receipts (all 174 commands, presentation events, and terminal board identical; event diffs are pure `card_def_id` renumbering).
13. Layer performance: per-object dirty bits (`FINISH_TYPES` / `FINISH_ACTIVATIONS` / `FINISH_STATICS`) skip unchanged `finish_object` work (uncached debug oracle still forces `FINISH_ALL`); `PredicateSource` guarded by `reads_source()` in `layers::Context` and `Game::object_matches_with_source`; O(1) `This`/`Attached` targets and single-pass `objects.iter()` in `Context::targets` — `1811194e41`.
14. Payment-search performance: allocation-free `PaymentResources::cmp_source` + bit-packed `PaymentResources::hash`, `Arc<SpendPredicates>` precomputed hash digest on `ManaRestriction::Only` (byte-identical serde) and `ManaChoices` digest on `Obligation::Output`, single-pass `optimistic_supply` and per-`SpendingContext` `PurposeSupply` memo in `PaymentSearch` — `a17915a437`. Engine Budget beats CI reference `7d3ac1eb1c` (`bench_steps` -1.74% instructions, `bench_search` -4.85% instructions) with identical digests; `entry-choices` (`size=16`) `step:activate` 8.0× faster (`7.80`→`0.98 ms`) and `search` 3.43× faster (`789.64`→`230.19 ms`); `changeling-anthems` (`size=16`) `step:choice` 2.20× faster (`6.00`→`2.73 ms`) and `search` 1.89× faster (`2,080.77`→`1,098.88 ms`).
15. Payment-search equivalence, canonicalization and hashing: purpose-projected deduplication in `Game::mana_ability_actions` (debug oracle verified), purpose-projected `PaymentSearch::groups_for` + `CanonicalState` memo keys preserving `SpendEffect`/`ManaExpiry`, O(1) `PaymentResources` `u64` bitmasks (`used_masks` + `used_overflow`), deterministic `FastHasher` for search tables, and `Arc::ptr_eq` in `PaymentCache`. Cuts `changeling-anthems` retired instructions by 74.4% (`80.55B`→`20.65B`, linear `step:choice`) and `entry-choices` by 40.2% (`29.93B`→`17.91B`), with identical `bench_steps`/`bench_search` digests beating base `3be5a0cfb2`.

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
- Hash version 33 moves match-state digests: expect Engine Budget digest and
  retained-fixture changes; adopt with `scripts/engine_budget.py record --reason`.
- Pinned pilot: v11 card column 56 (`put_into_graveyard_this_turn`) is an
  appended input column; `tests/model/test_pinned_pilot.py` must pass or the
  change must be declared with `scripts/pinned_pilot.py record --reason`.
