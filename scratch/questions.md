# MTG-286 decisions and evidence — 2026-11-12

The live plan is `scratch/mtg-286-define-every-tle-card-so-tla-limited-pools.md`.
Full earlier notes (passes one to seven, compressions, realignments) are at
`git show 2d40ac781:scratch/questions.md`; this file keeps what is still live.

## People's words

Jack Heart: "lets file a task for covering all of TLE". Steer (Task comment
af74fb09): "Ok, we should start with only the TLE cards that are bonus sheet
TLA cards", "no need to remove anything already included though", "All 41 names
missing from the 1,000-event corpus are in this list", and "Keep everything
already built (Cabbage Merchant payment model, the 9 non-bonus TLE cards, the
242-name ledger); don't remove or revert it, but don't extend non-bonus TLE
cards further. The other ~180 non-bonus TLE cards are later work".

## Reversible assumptions

- 1v1 development world with correct no-commander behavior; Commander and
  multiplayer stay a demo question. Shipping guide keeps planned-support wording
  and the Scout's Warning example until gate/demo.
- Scout's Warning: payload-free `NextCreatureCastFlash`; casting any creature
  card ends every such record. -1/-1 counters and other counter kinds are
  counter-kind fact rows. Eladamri's Call uses the existing search-to-hand path.
- Imprisoned in the Moon uses `SetTypes { [Land], [] }`, which also drops land
  subtypes; unverified whether a land keeps its land types.
- Empty City Ruse's record applies while its player is active and ends at their
  cleanup; a copy or flash grant during that player's own turn would apply at once.
- Eighth pass (this file's newest section) adds three more, listed there.

## External sources

- `managym/cards/tests/fixtures/tle-inventory.jsonl`: 242 names from the
  [TLE search](https://api.scryfall.com/cards/search?q=set%3Atle&unique=cards&order=name)
  plus exact-name print searches, fetched 2026-11-12 UTC (initial HTTP 429s,
  retried slowly). SHA256 `334e003532fcc4d3426086da033a02356715faeb7c892279978ba40a394a327d`.
  Noxious Gearhulk uses KLD (not MPS); Three Dreams records its earlier PSAL printing.
- `tle-bonus-sheet.jsonl`: Scryfall `set:tle`, `booster: true`, collector 1–61.
- Card shells in `managym/tests/fixtures/scryfall_cards.json` are fetched by pinned
  first-printing id through `https://api.scryfall.com/cards/collection`.
  Token ids come from the creating card's `all_parts`: Human
  `7d13a93a-a43d-4cf5-8300-8341f3b7f1b1` (Join the Dance), Koma's Coil
  `54a1c6a9-3531-4432-9157-e4400dbc89fd`, Marit Lage
  `8f5c3863-c876-48a8-9bc8-be20cd61074c` (fetched 2026-11-11 local).
- Lita's Zeppelin has no token printing on Scryfall (`name:Zeppelin include:extras`,
  `set:tj22`), so it has no `scryfall_id`.

## Open questions (each blocks only the named cards)

- Token identity: Release to Memory needs a plain colorless 1/1 Spirit, but the
  content pack registers tokens by unique name and `token.spirit` is the TLA
  Spirit. Lita's Zeppelin has no printing id. Decide name-vs-key resolution and
  the id of an unprinted token.
- Production review (`scratch/tle-production-design.md`): event timing and the
  v10 encoding table; blocks Heartbeat of Spring, The Great Henge, Tarnished
  Citadel (and non-bonus Toph, Uncle's Musings, Animal Attendant).
- New observed history v9 cannot hold (no spare player/card columns; a new
  `FactKind` is a layout decision): Bloodchief Ascension needs per-player life
  lost this turn; Brought Back needs per-card "put into a graveyard from the
  battlefield this turn". Choose a new fact kind on v9 or wait for a freeze.
- Cityscape Leveler's Powerstone mana "can't be spent to cast nonartifact
  spells": `ManaUse` restricts only by the Lesson/Shrine/Ally subtype bits and
  the pool's restricted classes are observed by those bits. Needs a restriction
  shape (artifact spells plus any ability) and its observation.

## Known gaps

- Training Grounds' reduction is applied when the activation is priced, before
  an announced X: an `{X}` waterbend activation is not reduced after X is chosen.
- Taunting Challenge maximizes requirements one blocker at a time; with menace
  or a "can't be blocked except by two" attacker, a lone required block may be
  removed later rather than redistributed.
- Koma's "Choose one" is two activated abilities sharing the sacrifice cost (the
  mode is chosen at activation, CR 700.2a); a rule counting abilities would see two.
- No observation tracks a player's knowledge of library order (Cruel Tutor).
- The player `hand_only` column shows only the temporary restriction; Drannith
  Magistrate's static is seen through card tokens and offered actions.

## Not run before commits

The `/verify` skill verifies a merged change and finishes the Task, so it does
not fit mid-implementation checkpoints. Full suites, corpus admission through
Doppler R2 credentials and browser checks belong to gate.

## Eighth pass — six bonus cards, 2026-11-12 UTC

Defined: Koma, Cosmos Serpent (khm), Training Grounds (roe), Taunting Challenge
(ptk), Volcanic Torrent (cmr), Mystic Remora (ice), Dark Depths (csp).
New rules: `Modification::CantActivateAbilities` (offers, activation, payment
model and producible mana all skip the permanent) and `MustBeBlockedByAllAble`
(a blocker able to block such an attacker is offered only those blocks);
`StaticAbility::ActivatedAbilitiesCostLess { predicate, generic }` with a one-mana
floor, applied in activation finalization; `TriggerCondition::YouCastThis`
(the spell is the source, its caster controls the inline trigger);
`Effect::Cascade` with `ResolutionCastOffer::Cascade { card, exiled }` (others
and a declined hit go to the bottom in RNG order; an uncastable hit bottoms at
once); `EffectValue::SpellsCastThisTurn`; `Effect::CumulativeUpkeep { each }`;
counter kinds `Age`, `Ice`; `ReplacementEffect::EntersWithCounters`;
`TriggerCondition::NoCounters { kind }`, a CR 603.8 state trigger checked in
stabilization over the sources indexed with counter changes, not retriggered
while its occurrence waits or is on the stack; Koma's Coil and Marit Lage tokens.

Behavior change taken: every `CastDuringResolution` offer other than `Hand`
(exile or library) is now refused while the caster `casts_only_from_hand`
(Drannith Magistrate), closing the gap for `ExileUntilNonlandCastFree` and
`CastLibraryTopFree` too.

Assumptions taken (reversible): Koma's modes as two abilities (above); Mystic
Remora's draw runs for the caster's opponent (`TriggeringPlayersOpponents`, 1v1)
and "you may draw" is a two-mode `Modal`; Dark Depths' trigger creates Marit
Lage only if the sacrifice happened (current Oracle "If you do").

Results (debug): cards target 656 passed; `rules_tests` 169 passed (2 ignored);
lib 192 passed; `conformance_tests` 2 passed; fmt applied. Rebuilt cp313
extension (atomic replace); `check_tle_coverage.py --bonus-sheet` 41 of 61, 20
missing, no other issues; pytest observation parity, text observation, TLE
allocation, policy catalog and record vocabulary 112 passed. The Academy
permanent-note phrases were added without a frontend check (gate).
`cargo clippy --workspace --all-targets`: four warnings, all in code this pass
did not change (`DelayedEvent`, `PowerCda`, `StackObject` variant sizes; a manual
checked division in `flow/mana.rs`); CI does not run clippy. `git diff --check` clean.

## Eighth-pass compress, 2026-11-12 UTC

Shared `next_trigger_id` for the four trigger enqueue sites; `YouCastThis` and
`NoCounters` never fire from `trigger_condition_fires` (their own scans own them);
loop form of `activated_cost_reduction`; dropped `counter_description` arms equal
to `kind.name()`; moved a misplaced doc comment. Debug `cargo nextest run
--workspace`: 1131 passed, 1 failed — `semantic_conformance_tests`
`checked_semantic_conformance_evidence_reproduces`: all 14 traces drift because
the pack `content_digest` moved (evidence `0798b2c7…`, regenerated `cd7a91d1…`)
as the branch added definitions; the earlier passes did not run this target.
Gate regenerates `conformance/semantic-kernel-w17-v2` evidence. Extension not
rebuilt (no behavior change).

## Ninth pass — six bonus cards, 2026-11-12 UTC

Defined: Clone Legion (dtk), Rhys the Redeemed (new `shm.rs`), Three Dreams
(new `psal.rs`), Sundial of the Infinite (m12), Mirrorwing Dragon (emn), Force
of Negation (mh1); Elf Warrior token `49180301-f72f-48ca-8be1-df295e89cdd7`
(Rhys's `all_parts`). Card data fetched 2026-11-12 UTC by pinned id from
`https://api.scryfall.com/cards/collection`.

Assumptions taken (reversible):
- Force of Negation's alternative reuses `CastChoices::kicked` (no v9
  column): kicked means the alternative is used, as it already means the mana
  for `PayLifeOrMana`. The LLM relation text still says "kicker included" for
  it; native offer labels and decision context name the alternative.
- `EndTheTurn` moves stack spells (and copies) to exile, drops abilities and
  pending triggers, ends combat, empties mana, and jumps to cleanup; the game
  loop then starts cleanup without priority (`tick_priority` and `turn_tick`
  return when a resolution reset the step).
- Mirrorwing: legality of each new target is slot 0's retarget candidates
  from the caster's view; every target slot of a copy becomes its creature.
- Three Dreams' set file is `psal.rs` (Scryfall set `psal`, Salvat 2005).

Results (debug): cards target 663 passed; lib + `rules_tests` +
`conformance_tests` 363 passed (2 skipped); fmt applied. Rebuilt cp313
extension (atomic replace); `check_tle_coverage.py --bonus-sheet` 47 of 61,
14 missing; `--allocation-only` no issues. Pytest observation parity, text
observation, TLE allocation, policy catalog, record vocabulary, Academy
optional/merchant payment and the new
`tests/academy/test_alternative_cost_recovery.py`: 116 passed, 6 failed, none
from this pass: three `test_decision_surface.py` precision demos ("content/setup
identity differs from this runtime", the pinned demo identity moved with the
added definitions, like the conformance evidence; gate) and
`test_merchant_payment_recovery` for v5/v6/v8 (main supports only v9).

Ninth-pass compress: `Game::uses_exile_alternative` names the kicked-means-
alternative rule once (play.rs); distinct-name search finish is one match; the
stale "distinct land" error reads "distinct-name search candidate". Debug: cards
663, lib 192, `rules_tests` 169 passed. No behavior change, so the extension was
not rebuilt.
