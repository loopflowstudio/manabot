# T6 (MTG-225): Sagas, faces, face-aware copies and turn scheduling — plan

Status (2026-11-02): all 16 cards and the one-catalog refactor are merged here; not published. Jack Heart's delivery order (2026-11-02): MTG-238 #414 goes first; T6 integrates it afterward. Jack Heart's steer (2026-11-02, comment
b222da8f) adds the content-pack simplification to T6; see
[t6-direction.md](t6-direction.md). Brief: Task MTG-225 (16 cards, allocation
`planned_pr: T6`); direction in [t6-direction.md](t6-direction.md). One
cohesive PR, then stop for Jack Heart's review.

Base: T6 started on main `4c8324a59` (T3 + T4) while T5 #405 synced onto
main, then merged T5's synced head `836042c07` (2026-11-02, four small
conflicts), so it is stacked on T5 with T5's copy effects available.

## Model

- **Faces (CR 712).** A transforming double-faced card is one definition
  function whose `CardDefinition.back_face: Option<Box<CardDefinition>>`
  holds the back face. The content pack registers the back face as its own
  row right after its front, so a transformed permanent's observation row,
  learning tokens and text name the back face. Back-face rows are not deck
  cards (pool reason, deck admission). The front's tokens carry a definition
  reference to the back row (like `token_name`).
- **Face state on the card.** `Card` keeps its front row while the back face
  is up (`Card::transform` swaps `definition_id`/`definition`); leaving the
  battlefield turns it front face up (CR 712.8a, 400.7). Transforming keeps
  the same permanent (counters, damage, attacking; CR 712.9). "Return it
  transformed" enters with the back face up (`ZoneEntry.transformed`,
  CR 712.14). A card with no back face (a token copy) ignores transform
  instructions (CR 701.28c).
- **Sagas (CR 714).** `CounterKind::Lore`. Chapter abilities are
  `TriggerCondition::Chapter { n }` triggered abilities ("I, II" = two
  abilities). A Saga enters with one lore counter (714.3a); after the draw
  step the active player adds one to each Saga they control (714.3b, turn-based
  action as the precombat main phase begins). Adding lore counters triggers
  each chapter whose number was crossed (714.2b). SBA 714.4: sacrifice a Saga
  with lore ≥ its final chapter unless one of its chapter abilities has
  triggered and not left the stack.
- **Turn scheduling.** Extra turns are a queue on `TurnState` (CR 500.7,
  most recent first); extra combat phases are an inserted combat phase
  (CR 500.8) with an optional attack restriction (Bumi).

## Cards

| Card | State |
| --- | --- |
| Leaves from the Vine | supported; directed tests |
| The Cave of Two Lovers | supported; directed tests |
| The Legend of Kuruk // Avatar Kuruk | supported; transform reentry, Spirits, extra turn |
| The Legend of Kyoshi // Avatar Kyoshi | supported; directed test |
| The Legend of Roku // Avatar Roku | supported; directed test |
| The Legend of Yangchen // Avatar Yangchen | supported; decision code 23 (choose to exile) |
| The Rise of Sozin // Fire Lord Sozin | supported; name choice as a resolution Modal |
| Aang, Swift Savior // Aang and La | supported; airbends spells, transforms in place |
| Aang, at the Crossroads // Aang, Destined Savior | supported; next-upkeep delayed transform |
| Avatar Aang // Aang, Master of Elements | supported; bending events and four-kind history |
| Bender's Waterskin | supported |
| Bumi, Unleashed | supported; additional land-creature combat |
| Koh, the Face Stealer | supported; ability inheritance as an effect record |
| Ember Island Production | supported; copies the face that's up, transformed |
| Fire Lord Azula | supported; spells cast while attacking are copied |
| Foggy Swamp Visions | supported |

## Delete — do not maintain

- `Effect::ExileTopCardPlayThisTurn`: replaced by `ExileTopCardsToPlay {
  count, until: PlayWindow }`. Done.

## Remaining

- After #414 lands: integrate it here (managym-state/managym-cards crate move,
  Keywords API). #414 owns Phoenix Fleet Airship's conversion to a conditional
  TypeGrant and already incorporates main's T5.
- Gate: re-record w17 evidence (list below); Academy UI checks have not run.
- Decide (Jack Heart): the one-catalog encoder cost, below. Until then
  `test_etude_job.py::test_worker_runs_real_learner_and_reports_admitted_policy`
  exceeds its 30 s limit.
- Not done: a retained checkpoint bound to an authored pack no longer loads.

## Checks

- 2026-11-02: debug `cargo test --tests` green except Scryfall fixture (fixed) and
  semantic-kernel w17 evidence (content digest moved; re-record at gate).
- 2026-11-02 (one catalog + face fixes merged, HEAD): debug `cargo test --lib
  --tests --no-fail-fast` fails only
  `checked_semantic_conformance_evidence_reproduces`; `--test cards` 523 passed;
  clippy clean. pytest semantic, policy catalog, academy, heuristic, parity,
  env, authored matchup, mirror admission, training decks, package boundaries:
  22 Academy failures, all retained-identity or digest drift awaiting re-record
  (authored match, choice navigation, decision surface, learn evidence/setup,
  public commitment, trigger order); 4 more were a stale `permanent_dim` pin,
  fixed. tests/training as a whole not run here.
- 2026-11-02 (all 16 cards, T5 gate head `2df025c20` merged): debug
  `cargo test --tests --no-fail-fast` fails only
  `checked_semantic_conformance_evidence_reproduces` (w17 content digest moved;
  re-record at gate); Python text/parity/protocol 62 passed.
- 2026-11-02 (after T5 merge, extension rebuilt): debug lib/cards/conformance/
  rules targets green; pytest tests/semantic tests/env + parity 276 passed;
  experience protocol 14 passed. Academy UI checks not run (no node_modules).

## One catalog (steer b222da8f)

Branch `jack/mtg-225-t6-one-catalog` (from `2c1ba169f`). Every match runs on
`default_content_pack()`, built once from `cardsets::catalog()` plus Crossroads
of Destiny, name order, back faces right after their fronts. Removed:
`ContentPack::from_definitions`/`pack_key`, the `CardRegistry` alias,
`ContentPackManifest.pack_key`, `content_pack_for_match`/`admits`/`select`/`exact`,
the same-pack error, `pool_content` and the `pools()` map, `Game::new_with_content`,
`created_tokens` and pack token closure. `PlayerConfig.world: Option<String>`
became `submitted: bool` (serialized `"world": "world"`); Python keeps
`PlayerConfig.world` accepting only `None`/`WORLD_SETUP` and rejects pack keys.
Authored setups are `AuthoredSetup` decklists named by preset or `pack_key`.
`WorldIdentity` reads the catalog directly; world queries are infallible. Ward
lowering stays in `register_card`, documented as the catalog's one normalization.

Proof: catalog digest `6741f471…9827`, length 345 and the id→name manifest
hash `f67e67ee…b603` are identical on `2c1ba169f` and on the branch;
`content_pool("world")` and `world_identity()` give the same digest;
`CARD_DEFINITION_COUNT` = 345 = catalog length.

Measured cost, for Jack Heart to decide: authored vocabulary names now bind the
345-row world catalog. `SemanticCardEncoder("ur-lessons-vs-gw-allies", 64)`
goes from 31 definitions / 3,598 expanded tokens / 4.9 ms per `definitions()` to
345 / 51,909 / 45.3 ms (`elemental-world-v2` was 100 / 14,097 / 20.5 ms). The
Etude worker fixture regime (`ataraxos_move_scalar`, hidden 8, 2 updates) takes
5.07 s at `2c1ba169f` and 31.29 s on this branch, 26 s of it in `backward`.
`test_etude_job.py::test_worker_runs_real_learner_and_reports_admitted_policy`
therefore exceeds its 30 s budget and fails; left failing, not retuned.

Re-record needed (not done; frozen evidence). Failure sets compared by running
the same tests at `2c1ba169f`:
- `conformance/semantic-kernel-w17-v1/evidence.json` and `manifest.json`
  (`checked_semantic_conformance_evidence_reproduces`): already failing at
  `2c1ba169f` with 12 cases drifted (evidence digest `92b2f506…`). The branch
  adds `temur-163` and `temur-164`, and changes the content digest of the
  elemental, sacrifice, temur and elemental-world cases, which ran on authored
  packs.
- `academy/fixtures/choice-navigation-w17.json`: position `CHOOSE_TARGET` is the
  one new Academy failure (state digest); `DECLARE_BLOCKER`, `DISCARD`, `LEARN`,
  `PAY_OR_NOT` already failed.
- Already failing at `2c1ba169f`, re-record at the gate; authored matches among
  them also move here: `conformance/authored-match-w17-v1`,
  `authored-match-parity-w17-v1`, `public-commitment-parity-w17-v1`,
  `academy/fixtures/{learn-demo,precision-demo,precision-mana-demo,trigger-order}-w17.json`.
- Not evidence, same failure at `2c1ba169f`: `test_ur_lessons_increment` (stale
  `permanent_dim` 37 vs 38), `tests/sim/test_rollout.py` selects-valid-actions
  (`fact validity must be binary`), `test_search_runtime` seed-197 ABIs,
  `test_conditional_search` frozen fixture, `tests/belief/test_state.py` receipt.

Fixed because the refactor exposed it (pre-existing for the world catalog at
`2c1ba169f`): `BoundSemanticPack.pad_catalog` raised "definition reference is
outside program spans" on T6's `back_face` reference, which sits in the front's
characteristics span; 7 projection tests that used authored packs hit it. It has
no production caller. It now also returns
`characteristics_ref_{source_positions,target_rows,mask}`.

Checks weakened because their subject no longer exists (review these):
- A setup binding no longer has authored admission behind it, so a checkpoint
  whose recorded `setups` were edited still loads; execution still compares the
  actual decks to the binding. Removed the `sideboard` case of
  `test_ordinary_loader_rejects_incompatible_binding_before_weights`.
- Removed "explicit pack" and partial-binding assertions from
  `test_mirror_admission.py`, `test_matchups.py`, `test_elemental_world.py`; a
  mirror of authored decks now runs without any selection.
- `checkpoint_world` recognises the Elemental roster by deck membership (any
  pair of its decks binds all ten), where it used to read the explicit pack; a
  hand-built pair of those decks now gets the roster binding too.

Compatibility for Jack Heart: setup-bound checkpoints of authored packs no
longer load (`checkpoint world/setup/input binding differs`, pinned by
`test_retained_manifests_load_only_when_they_name_no_authored_pack`); their
pack-sized encoder weights could not fit a world-sized encoder either, though no
retained checkpoint file was tried. Manifests recording `pack_key: None` load
through `current_binding_form`. `CURRENT_PACK_KEYS` stays as the names a recipe
may give its vocabulary, all binding the world catalog; collapsing them to
`world` would mean rewriting regimes and `execution.py`'s Elemental check.
`MatchConfig.world` keeps a frozen regime's authored label and runs it like
`None`. Curated manifests keep `semantic_pack_key`, resolved as a setup name.
Kept, allowed by the steer: `Arc<ContentPack>` on game state and public
`register_card`, which content-sensitivity tests use to extend a copy.

## One catalog (steer b222da8f)

Merged from `jack/mtg-225-t6-one-catalog` (`3493282a4`). Every match runs on one
catalog; authored setups are decklists named by preset or `pack_key`;
`PlayerConfig.world` is `submitted: bool` natively and Python accepts only
`None`/`WORLD_SETUP`. World digest, length and id-to-name manifest were identical
before and after at `2c1ba169f` (`6741f471...`, 345).

Cost for Jack Heart to decide: authored vocabulary names bind the whole catalog.
UR/GW encoder 31 to 345 definitions; `definitions()` 4.9 to 45.3 ms; Etude worker
fixture 5.07 to 31.29 s, 26 s of it in `backward`. Options: accept it, or encode
only the definitions a batch uses.

Consequences: authored-pack checkpoints no longer load (pinned by a test);
edited `setups` in a checkpoint binding now load, with execution still comparing
decks; explicit-pack assertions were removed from mirror admission, matchup and
Elemental tests; `CURRENT_PACK_KEYS` stays as names that all bind the catalog.

Re-record at gate, none rewritten: semantic-kernel w17 evidence and manifest;
authored-match and parity, public-commitment parity; Academy choice-navigation,
learn, precision, precision-mana and trigger-order fixtures.
