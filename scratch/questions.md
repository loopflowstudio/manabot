# MTG-286 decisions and evidence — 2026-11-11

Sync resolution: retain this branch's active TLE notes when main deletes the
inherited release-notes file; main's release and licensing changes apply unchanged.

Jack Heart: "lets file a task for covering all of TLE".
Inherited MTG-285 release notes remain in this file at checkout base
`4c041d6cefe829c5b1b52273ba19991d663ee26b`; they do not direct this Task.
The live plan is `scratch/mtg-286-define-every-tle-card-so-tla-limited-pools.md`.

Reversible assumptions: retain the existing 1v1 development world with correct
no-commander behavior. Full Commander/multiplayer setup remains a demo question,
not a prerequisite or a reason to narrow the 242-name inventory. Retain the
shipping guide's planned-support labels and Scout's Warning example until the
complete implementation demonstrates them.

Earlier documentation-only work: the inventory matched 242 fetched names,
8 defined and 234 missing; syntax/fence/link checks and `git diff --check`
passed. No runtime or corpus proof was claimed. `lf open --host desktop --file
docs/engine.md` reported "No Desktop Files view attached; nothing opened."
Original uncommitted notes were retained before compression at
`/var/folders/m6/r3tllnrs1yq7yfbwm680tss40000gn/T/mtg286-compress-ju2m728q/`.
Realign observed `origin/main` at `4c041d6cefe829c5b1b52273ba19991d663ee26b`,
no Wave or PR, and no further Task comments beyond the claim. Runtime sources
then matched that base; this implementation pass changes them.

## Implementation evidence

The new `managym/cards/tests/fixtures/tle-inventory.jsonl` preserves all 242
names, Oracle IDs/paragraphs, TLE printing IDs and first-printing IDs/sets.
Sources: [TLE search](https://api.scryfall.com/cards/search?q=set%3Atle&unique=cards&order=name)
and Scryfall exact-name print searches ordered by release ascending, fetched
2026-11-12 UTC. Requests initially hit HTTP 429; retained successful responses
and sequential slower exact-name searches completed the fetch. Existing eight
cards keep their definition printing IDs. Noxious Gearhulk uses KLD rather than
its simultaneous MPS edition; Three Dreams records its earlier PSAL printing.
This is external source data, not a complete clause allocation or admission list.
SHA256: `334e003532fcc4d3426086da033a02356715faeb7c892279978ba40a394a327d`.

The Cabbage Merchant is implemented with counted tap costs and both printed
triggers. The first competing-cost test exposed an offered Food that was
reserved for its own activation; checking reserved tap availability before
selection fixed it. The test now proves five Foods pay the Food activation
through two Merchant activations, whereas four cannot. Source-tapped behavior,
three/four Food counts, waterbend exclusion, and Command replay/stale rejection
also pass. Observation continuation remains acting-viewer-only, consistent
with the existing privacy contract; committed permanent taps are public.

Validation (focused, debug unless stated):

- `cargo test -j 2 --manifest-path managym/Cargo.toml -p managym-cards --test cards merchant_ -- --test-threads=2` — 12 passed: seven TLE behavior tests, one new learning-token test, four existing Merchant-named tests.
- Same binary, filters `boiling_rock_rioter_exiles_with_itself_and_casts_your_exiled_allies`, `barrels_of_blasting_jelly_filters_one_mana_once_each_turn`, `committed_optional_payment_resumes_after_two_nested_mana_decisions`, `every_trigger_condition_and_ability_reaches_the_learning_tokens`, `badgermole_cub_earthbends_and_adds_green_when_a_creature_taps_for_mana` — 5 passed.
- `cargo test -j 2 --manifest-path managym/Cargo.toml -p managym --lib registered_mana_activations_consume_finite_payment_resources -- --test-threads=2` — 1 passed.
- Required root `uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — built cp313 macOS arm64 wheel; atomically replaced the matching `_managym.cpython-313-darwin.so`.
- `uv run --extra dev --extra play pytest tests/verify/test_observation_parity.py -k merchant_tap_count -q` — 4 passed (v5, v6, v8, v9); native allocation, preallocated buffers, Python reference and LLM text agree at both tap selections. Foods are created by real Unlucky Cabbage Merchant casts from the existing priority-root fixture, not injected token deck entries.
- Changed Python Ruff check/format, workspace formatting and `git diff --check` passed. Offline inventory validation matched the original 242-name list, unique IDs and 9 current literal definitions; 233 remain missing. Scratch stays within both context limits.

The new production counterexample is recorded in the plan: Heartbeat needs
actual produced types, either player's triggering sources and output identity
that `ManaChoice` currently discards. Animal Attendant needs spent-mana entry
provenance absent from current pools. The implement instruction says:
"If implementation turns up a counterexample that breaks the slice, the data
model, the deletion path or the overall design, stop dependent work and revise
the design or send it back to review." This stops dependent production work;
it does not remove all-card scope or mark the Task complete. No Graph transition,
publication, corpus fetch or corpus admission proof occurred.

Remaining at that point (clause allocation and vector/Academy proofs were
completed later; see below): contract review, missing definitions, world
integration, gate suites and the 1,000-event corpus admission proof. The artifact corpus is in the manifest, not initially local;
fetch through Doppler at that acceptance step. Frozen evidence stays unchanged.

## Compression — 2026-11-12 UTC

Pre-edit uncommitted notes retained at
`/var/folders/m6/r3tllnrs1yq7yfbwm680tss40000gn/T/mtg286-compress-xijd7yuo/`.
The inventory table now points to the complete JSONL instead of duplicating
242 names; baseline evidence and all external identity bytes are retained.
The plan describes the surviving resource model and shared selected-tap witness;
all-card scope, production-design blockers and acceptance obligations remain.

Validation of compression:

- `cargo test -j 2 --manifest-path managym/Cargo.toml -p managym-cards --test cards merchant_ -- --test-threads=2` — 12 passed in debug after the final code edits; the same binary's `boiling_rock_rioter` and `committed_optional_payment_resumes_after_two_nested_mana_decisions` filters each passed.
- `cargo test -j 2 --manifest-path managym/Cargo.toml -p managym --lib registered_mana_activations_consume_finite_payment_resources -- --test-threads=2` — 1 passed.
- `cargo fmt --manifest-path managym/Cargo.toml --check`, changed-Python Ruff and `git diff --check` — passed. Offline comparison verified all 242 names against the historical table and the unchanged JSONL SHA256.
- Root `uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — passed; matching cp313 macOS arm64 extension installed atomically. `uv run --extra dev --extra play pytest tests/verify/test_observation_parity.py -k merchant_tap_count -q` — 4 passed (v5/v6/v8/v9). Full affected suites remain with gate.

## Realignment — 2026-11-12 UTC

`lf task status MTG-286 --json` returned no Wave or PR and no direction beyond
the existing claim comment. Fetched `origin/main` at
`c405da851f98ccbb0697f98205451a47cbbad9b8` (MIT licensing/release work), already
an ancestor of checkout HEAD `afdab8dca00cbc3ff6cc6fe6d557f724ad4100b3`.
The incoming feedback's "runtime sources remain unchanged, 234 definitions are
missing" describes the earlier pass; the current Merchant implementation and
recorded focused results supersede it.

Offline inventory comparison still found 242 unique names, 9 literal definitions
and 233 absent; the JSONL SHA256 remains unchanged. Source inspection confirms
the recorded parity test uses scalar allocated/preallocated buffers, not vector
execution. The plan now separates those proofs, makes production/spending and
observation design obligations explicit, and retains 1v1/no-commander as the
reversible assumption with expansion left for demo. Existing focused test results
were reused; no runtime suite or corpus admission was rerun during realignment.

## Clause allocation and integration — 2026-11-12 UTC

The allocation binds all 242 names and 411 Oracle paragraphs to native rule
obligations; it includes vanilla cards and retains first-printing owners.
`check_tle_coverage.py --allocation-only` checks that binding, while its default
mode checks the installed catalog and fails for missing support. Neither is an
admission authority or a printed-behavior proof. Five offline tests cover the
intact allocation and rejection of missing/duplicate clauses, wrong owners and
changed source bytes. The inventory SHA256 remains unchanged.

Merchant's first WebSocket fixture with exactly two Foods skipped the second
tap automatically. Three Foods preserve real choices at both taps with default
forced skipping on. The final fixture casts ordinary cards, reconnects at both
unpaid counts and the color choice, compares exact fork/Command execution and
native VectorEnv replay with Python encoding, and rejects stale/duplicate
Commands without mutation. It covers v5/v6/v8/v9; it is not browser visual proof.

Eight ordinary definitions were added with six directed debug tests. Their
printing shells were fetched by the pinned IDs through
`https://api.scryfall.com/cards/collection` on 2026-11-12 UTC; IDs, sets and every
Oracle paragraph matched the pinned inventory. The retained shells are in
`managym/tests/fixtures/scryfall_cards.json`. Names: Capital Guard, Explosive Shot,
Kindly Customer, Koala-Sheep, Kyoshi Warrior Guard, Warship Scout, Wolf Cove
Villager and Zuko's Offense. No new mechanic was needed; Sheep was missing from
the shared subtype category table and is now a creature type.

The production proposal now includes additional allocation findings: Toph,
Greatest Earthbender requires actual total cast spending; Uncle's Musings requires
its distinct colors. Citadel damage and Henge life gain must run inside the mana
ability, once. `scratch/tle-production-design.md` names the single production
occurrence, pool equivalence, explicit spending and cast receipt owners plus
predecessors to delete. Production-event timing and the exact v10 encoding/join
grammar remain unresolved review items, not runtime support. No version/default
or frozen evidence changed. The no-commander assumption and planned-support
shipping documentation remain in place.

Focused validation:

- `cargo test -j 2 --manifest-path managym/Cargo.toml -p managym-cards --test cards tle_baseline -- --test-threads=2` — 6 passed in debug, covering all eight additions.
- `cargo test -j 2 --manifest-path managym/Cargo.toml -p managym --test conformance_tests every_registered_real_card_matches_scryfall -- --test-threads=2` — 1 passed.
- Root `uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels` — passed; matching cp313 macOS arm64 extension replaced atomically after final Rust edits.
- `uv run --extra dev --extra play pytest tests/academy/test_merchant_payment_recovery.py tests/infra/test_tle_allocation.py tests/verify/test_observation_parity.py -k 'merchant_payment or tle_allocation or merchant_tap_count' -q` — 13 passed against the rebuilt extension.
- `uv run python scripts/check_tle_coverage.py` — exit 1, exactly 225 missing definitions; the 17 present names pass identity, eligibility, key uniqueness and learning-row checks. This is an unmet outcome, not a passing coverage result.
- `cargo fmt --manifest-path managym/Cargo.toml --all -- --check` — passed. Root-only Cargo formatting omitted child-crate files; AGENTS.md now names `--all`.

Remaining: review/design of the new contracts and other allocated rule extensions,
225 definitions, token identity closure, current-world integration/receipts,
full gate suites and the 1,000-event corpus admission proof. No corpus fetch,
Graph transition, publication or Task completion occurred.

## Compression — second pass, 2026-11-12 UTC

Code-only cuts: `flow/mana.rs` imports hoisted and four still-accurate search
comments restored; the counted tap-payment instruction no longer overwrites its
stage text with a duplicate count (LLM text names it from the continuation
relation, "N tap selections still required"); TLE tests share `create_foods`;
the parity test's text-observation import moved to the module header.
Debug `merchant_ tle_baseline boiling_rock_rioter committed_optional_payment`
21 passed; lib `registered_mana_activations continuation` 10 passed; rebuilt
cp313 extension, focused Academy/allocation/parity pytest 13 passed.

## Realignment — third pass, 2026-11-12 UTC

`origin/main` had no commits beyond HEAD; the Task had no comments beyond the
claim. `uv run python scripts/check_tle_coverage.py` still reports exactly 225
missing definitions against the installed extension. The incoming direction's
allocation and Merchant vector/Academy items were already complete; the plan
already records them. No runtime suite was rerun.

## Fourth pass — definitions and three rule slices, 2026-11-12 UTC

Open question (does not block other slices): Heartbeat of Spring, Animal
Attendant, Toph, Greatest Earthbender, Uncle's Musings, Tarnished Citadel and
The Great Henge stay unregistered until an authored review settles the
production-event timing and the v10 encoding table in
`scratch/tle-production-design.md`.

Assumptions taken (reversible): Scout's Warning is a payload-free player record
`Modification::NextCreatureCastFlash`; every such record on the caster ends when
any creature card is cast (each Warning's "next creature card" is that card).
Its observation is the existing player effect fact (`Seen::Fact(Effect)`), like
`HasAbilitiesOfSource`; v1–v4 see only the offered off-timing cast. -1/-1
counters follow the existing open counter-kind fact vocabulary; v4 sees their
P/T change only. Eladamri's Call moves the found card to hand by the existing
search/return path (same as the TLA "reveal ... put into your hand" searches).

Results (debug unless stated):
- `cargo test -p managym-cards --test cards tle_baseline` — 18 passed; new reprint/rule
  filters `sol_ring diresight consider fervor run_amok heroic_intervention
  dramatic_reversal entomb eladamri` 9 passed; `black_sun each_counter_kind` 3
  passed; `scout_s_warning modification` 4 passed; `observability scryfall
  definition subtype` 66 passed; `conformance_tests` 2 passed; lib `counter sba
  layer` 14 passed and `vocabulary flash effect` 9 passed.
- Rebuilt cp313 extension (atomic replace). `check_tle_coverage.py` — exit 1,
  190 missing (52 TLE names defined), no other issues.
- `pytest tests/verify/test_observation_parity.py` — 33 passed, including the new
  real-cast Scout's Warning + -1/-1 fact parity at v5/v6/v8/v9;
  `tests/semantic/test_text_observation.py` 61 passed.
- The shipping guide's Scout's Warning example now prints `[[]]`; its
  planned-support wording is retained until gate/demo.

Player experience counters (traced from `no_maximum_hand_size`: PlayerData,
PyO3, experience projection, six protocol schemas, encoder facts, LLM text,
Academy UI) serve only non-bonus cards; outside the narrowed scope below.

## Fifth pass — bonus-sheet scope, 2026-11-12 UTC

Steer (Jack Heart, relayed by comment af74fb09): "Ok, we should start with only
the TLE cards that are bonus sheet TLA cards" and "no need to remove anything
already included though". The Task is retitled "Define the 61 TLE bonus-sheet
cards so TLA Limited pools admit". Acceptance stays: all 1,000 corpus events admit.
`managym/cards/tests/fixtures/tle-bonus-sheet.jsonl` pins the 61 booster printings
(Scryfall `set:tle`, `booster: true`, collector numbers 1-61; same TLE printing
ids as the inventory); `check_tle_coverage.py --bonus-sheet` checks that scope.

Twelve non-bonus TLE definitions written before the steer arrived (Earthbending
Student, Bumi, Suki, Momo, Inspired Insurgent, Mechanical Glider, Eel-Hounds,
Ruthless Waterbender, Loyal Fire Sage, Dai Li Censor, The Art of Tea, Frantic
Confrontation; 30 debug tests passed) were not committed; they are parked on local
branch `mtg-286-nonbonus-batch3-parked` for the later non-bonus Task.

Added bonus cards on existing rules: Meteorite, Sunbaked Canyon, Treetop Village,
Cruel Tutor, Captain Sisay, Prosperity, Insurrection, Intruder Alarm, Elemental
Bond, Beastmaster Ascension, Valakut, Humble Defector (13 debug tests passed).
Two rule touches: `Move` to the library a card is already in, with a stated end,
reorders it without a zone change (Cruel Tutor's "then shuffle and put that card
on top"; the proposal check correctly rejects same-zone moves); the definitions
guard now admits a source question in a P/T buff's own condition (layers ask it of
the source, `layers.rs` node conditions), with `CantBeBlockedIf` as the retained
negative. Cruel Tutor's caster knows the top card; no observation tracks library
order knowledge (existing gap, not new information revealed).
Then Visions of Beyond (new `Condition::AnyGraveyardAtLeast`) and Standstill
(new `PlayerSelector::TriggeringPlayersOpponents`), each with guard tokens,
witnesses, descriptions and directed tests. Debug: full cards target 632 passed;
`rules_tests` 169 passed (2 ignored); lib `tokens vocabulary description` 8 passed;
rebuilt extension; text-observation + allocation pytest 67 passed.
`check_tle_coverage.py --bonus-sheet` — 21 of 61 defined, 40 missing, no other issues.

Remaining bonus names need these rule families (allocation ledger has detail):
production/spending review (Heartbeat, Great Henge, Tarnished Citadel); copy entry
and token copies (Clone, Clone Legion, Sakashima, Rhys); cascade (Volcanic
Torrent); end-turn (Sundial); phasing + life lock (Teferi's Protection); combat
skip/requirements (Empty City Ruse, Taunting Challenge); counter/zone-history
triggers (Bloodchief, Dark Depths, Brought Back); cast-time costs (Force of
Negation, Shattering Spree replicate, Cityscape Leveler unearth, Mystic Remora
cumulative upkeep); scoped counts/costs (Blasphemous Act, Dockside, Return of the
Wildspeaker, Training Grounds, Agent of Treachery); new tokens (Join the Dance
Human, Release to Memory Spirit, Lita's Zeppelin, Koma's Coil, Rhys's Elf Warrior,
Marit Lage); damage source/LKI (Warstorm Surge, Noxious Gearhulk, Searing Blood);
uncounterable + color predicates (Rending Volley, Koma); others per allocation.

## Realignment — sixth pass, 2026-11-12 UTC

`origin/main` has no commits beyond HEAD. The incoming direction's counts
(17 defined/225 missing) and its experience-counter slice predate the bonus-sheet
steer; current `check_tle_coverage.py --bonus-sheet` is 21/61 with 40 missing
(66/242 on the full ledger). For this Task the production open question blocks
only Heartbeat of Spring, The Great Henge and Tarnished Citadel; Toph, Uncle's
Musings and Animal Attendant are non-bonus. Plan rewritten to the 61-card scope.

## Seventh pass — 13 bonus cards, 2026-11-12 UTC

Defined: Empty City Ruse, Join the Dance, Dockside Extortionist, Blasphemous Act, Agent of
Treachery, Return of the Wildspeaker, Warstorm Surge, Noxious Gearhulk, Searing
Blood, Rending Volley, Rites of Flourishing, Fevered Visions, Drannith
Magistrate, Imprisoned in the Moon. Shells fetched by first-printing id through
`https://api.scryfall.com/cards/collection` on 2026-11-12 UTC; Human token id
`7d13a93a-a43d-4cf5-8300-8341f3b7f1b1` from Join the Dance's `all_parts`.

Found and taken (reversible):
- Searing Blood's delayed damage was silently dropped: the damage proposal
  rejected the resolved spell's stack incarnation (no battlefield LKI).
  `proposed_event.rs::object_ref_is_known` now accepts an earlier incarnation of
  a real card (CR 603.7d source of a delayed trigger). Rules/lib/cards suites pass.
- Step-start triggers now carry the active player as the trigger's player
  (`PlayerSelector::TriggeringPlayer`), for Rites and Fevered Visions.
- `Destroy` now binds its results like `Move`, so `Condition::Moved` reads
  "destroyed this way" (Gearhulk); no earlier card read `Moved` after `Destroy`.
- Imprisoned in the Moon uses the existing `SetTypes { [Land], [] }` modeling
  (the rules witness in `cr_613_continuous_effects.rs`), which also drops land
  subtypes; whether a land keeps its land types under it is unverified.
- Drannith Magistrate shares the hand-only gate (`zone_cast_cost`); like the
  existing until-next-turn restriction it does not stop free casts from exile or
  library during resolution (`ExileUntilNonlandCastFree`, `CastLibraryTopFree`).
  Cascade will need that gate. The player `hand_only` column still shows only the
  temporary restriction; the static is seen through card tokens and actions.

Open (token identity; blocks two cards): Release to Memory makes a plain 1/1
colorless Spirit, but the content pack registers tokens by unique name and
`Spirit` is the TLA token with block restrictions. Lita's Zeppelin has no
token printing on Scryfall (searches `name:Zeppelin include:extras`, `set:tj22`),
so no `scryfall_id` exists for it. Both wait for a token-identity decision.

Not run before commits: the `/verify` skill (it verifies a merged change and
finishes the Task, so it does not fit mid-implementation checkpoints).

- Empty City Ruse's record applies while its player is active and ends at
  their cleanup; created at sorcery speed it is always "their next turn". A
  copy or flash grant during that player's own turn would wrongly apply at once.
  The Academy note was added but no frontend check ran (gate).

Results (debug unless stated): full cards target 646 passed; `rules_tests` 169
passed (2 ignored); `conformance_tests` 2 passed; lib 192 passed; fmt applied.
Rebuilt cp313 extension (atomic replace); `check_tle_coverage.py --bonus-sheet`
35 of 61, 26 missing, no other issues; pytest text observation, observation
parity, TLE allocation and policy catalog 101 passed.

Compress pass (2026-11-12): `power_toughness_or_lki` shares the
TriggeringPower/TargetToughness read; `damage_source` keeps only the
`Triggering` LKI fallback (its one consumer, Warstorm Surge) and its doc no
longer splits `select`'s; `by: None` literals became `Effect::deal_damage`.
Debug: cards 646 passed, `rules_tests` 169 passed, resolution lib 49 passed;
rebuilt cp313 extension; `--bonus-sheet` still 35 of 61, 26 missing.
