# T6 (MTG-225): Sagas, faces, face-aware copies and turn scheduling — plan

Status (2026-11-03 publication gate): MTG-238 is integrated at `373bb08ca`
against main `44388ad42`. The permitted w17 evidence refresh, native rebuild,
model/learner/admission checks and Academy verification are complete. Publication
is **blocked by unimplemented accepted Sozin behavior**, described below. No PR,
merge or auto-merge; one cohesive PR remains the review boundary.

The 2026-11-03 publication steer supersedes the earlier wait for #414. `lf sync`
confirmed the current base without additional changes. Historical implementation
and earlier blocked-publication notes below remain dated evidence.

## Publication gate — 2026-11-03

The accepted brief requires Fire Lord Sozin's optional X payment followed by a
reflexive multi-graveyard-target trigger. The definition at
`managym/cards/src/tla.rs:9331` instead collects targets on the initial trigger
and resolves `PayTargetsManaValue` afterward. This exposes targets before payment,
fixes X to their total, and omits the separate priority window after payment.
The existing test at `managym/cards/tests/cards/tla.rs:13933` explicitly expects
the incorrect ordering; a temporary assertion that the first choice must not be
`ChooseTarget` fails in debug. The original test bytes were restored. Reproduction:
`.runs/t6-publication/sozin-payment-order.log` (1 test, failed as expected).
The earlier “same outcomes” assumption in questions.md is not accepted behavior.

The Rise of Sozin's chapter II has a separate missing strategic choice:
`Effect::ExileNamed` (`managym/src/flow/resolution.rs:1553`) uses `.take(count)`
and moves all matches immediately, rather than allowing zero through four chosen
cards from the searched zones. Its existing test only witnesses automatic exile
of four copies. This contradicts the printed text and shared complete-action
contract. Neither behavior is an integration conflict or a fixture discrepancy;
completing their shared payment/trigger/search decision paths is implementation
work required before the publication boundary holds. Do not hide the gaps with
UI behavior or weaken the accepted obligations.

Completed publication preparation:

- Fixed Academy's Koh label to read the projected `ContinuousEffect.source`,
  rather than the native-only `source_name`; the original UI check failed here.
- Built the release cp312 wheel and atomically replaced the root native extension.
- Refreshed only mutable w17 semantic conformance, authored-match, both parity
  receipts, Academy demo/navigation/trigger fixtures and the four current belief
  v3 pins. Historical receipts and pins are unchanged. Retrieved the required
  RUL-9 artifact by its manifest digest (cache hit); no regeneration of it.
- Retained 142 authored Commands, actor/revision/offer counts, presentation events
  and terminal player state. Semantic event definition references move from the
  authored vocabulary to the one catalog (including definition-kind source IDs
  on CARD_REVEALED). Academy recorded Commands/choice prefixes are unchanged.
- No engine source changed in this gate. The catalog content digest is
  `144f167a79bbf89a1a9eec758d8c5006416fb6150b540802ee52ddb9a7ab74ec`.

Verification (logs in `.runs/t6-publication/`; successful unchanged checks reused):

- `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2
  -- --test-threads=2`: initial run passed all binaries before stale conformance;
  after refreshing conformance, ran that binary, remaining study binaries,
  managym-cards/managym-state suites, engine doctests and bench_search tests.
  **913 existing tests passed, 2 ignored**, including all 528 card tests and
  observability. This does not establish the two missing Sozin obligations.
- `cargo fmt --check`; all-target/all-feature clippy with pinned PyO3 Python:
  passed. Ruff on all 39 changed Python files, plus refreshed belief pins: passed.
- Python model suite, elemental-world, evaluation models, mirror admission,
  monitor evaluation, active training, Etude learner job and observation parity:
  passed within an invocation reporting 264 passed, 1 skipped and 10 stale
  Academy-fixture failures. The latter were repaired by the producer refresh.
- Focused Academy navigation/Learn/trigger tests plus changed current-content,
  learning projection, text observation, matchups and authored-matchup checks:
  **102 passed** after refresh.
- Authored parity CLI: 142 Commands, 143 checkpoints, 858 viewer checks, no
  divergence; stale-object and stale-revision rejection retained. Public parity
  CLI: 142 Commands, 69 commitments, zero gaps. Parity/belief tests initially
  17 passed / 3 stale pins or receipts; after refresh the five relevant tests
  passed. Authored presentation assertions stayed unchanged.
- Academy `npm ci --ignore-scripts`, `npm run check`, `npm test`, `npm run build`:
  passed, **119 unit tests**. Isolated Playwright release navigation/Learn/trigger
  cases on API 18125 / frontend 15185: **10 passed**. Test-owned servers exited.
- Engine Budget remains CI-owned and unmeasured here; no PR exists to supply a
  candidate run. Its potential digest/reference update remains publication work.

Compatibility caveat remains: old authored-pack checkpoint weights/bindings are
not migrated by the one-catalog change. No retained checkpoint file was loaded;
manifest rejection and current checkpoint paths are covered by the model suite.

Next: finish the required Sozin semantics and directed negative/choice tests,
revisit the other documented implementation approximations against acceptance,
refresh only receipts affected by those edits, then publish for Jack Heart's
review. Do not replay the completed merge or unrelated successful checks.

## Model

- **Faces (CR 712).** A transforming double-faced card is one definition
  function whose `CardDefinition.back_face: Option<Box<CardDefinition>>`
  holds the back face. The content pack registers the back face as its own
  row right after its front, so a transformed permanent's observation row,
  learning tokens and text name the back face. Back-face rows are not deck
  cards (pool reason, deck admission). The front's tokens carry a definition
  reference to the back row (like `token_name`).
- **Face state on the card.** `Card` keeps its front row while the back face
  is up (`Card::turn_back_face_up`/`turn_front_face_up` swap
  `definition_id`/`definition`); leaving the
  battlefield turns it front face up (CR 712.8a, 400.7). Transforming keeps
  the same permanent (counters, damage, attacking; CR 712.9). "Return it
  transformed" enters with the back face up (`ZoneEntry.transformed`,
  CR 712.14). A card with no back face ignores transform instructions
  (CR 701.28c). A token copy of a double-faced permanent has both faces and is
  transformed if the original is, so it can transform (CR 707.8a; the earlier
  "cannot transform" proposal is superseded, see questions.md).
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
- Compress pass (2026-11-02), done: admission's runtime-content argument and
  its unreachable token/missing-definition arms (the pool marks them
  ineligible), `learning_schema_path` and its pack-key argument, `default_order`,
  `scratch/tools/`. `Game::inherited_definition` returns the catalog row.
- Candidate, not taken: `BoundSemanticPack.pad_catalog` has no production
  caller (tests only); deleting it means reworking 7 projection tests.

## Remaining

- Integrate #414 after it lands: managym-state/managym-cards crate move and
  Keywords API. #414 owns Phoenix Fleet Airship's conditional TypeGrant, rebinds
  the same mutable w17 receipts (match-state hash contract 12) and allocates
  object ids for earthbend records. Re-recording before this integration would
  be repeated by it.
- Re-record w17 evidence (list below), rebuild the extension from the integrated
  source and run the full debug Rust, pytest and Academy UI checks. Academy UI
  has not run on this branch.
- Engine Budget (MTG-237 #413, merged and integrated at `a591093ee`; examples
  build in release). T6's rules and catalog changes probably move the
  search/step digests in `managym/perf/baseline.json`, which would fail the
  check. If so, adopt the PR's own run with `scripts/engine_budget.py record
  --reason` after #414 is integrated. Not measured locally, because the
  benchmarks are left to CI.
- MTG-235 #412 (effect vocabulary, 66 files) is still open and overlaps T6. No
  order between it and T6 is recorded.

## Learner cost and vocabulary labels (headless concept review, 2026-11-02) — done

Both directions are implemented (`321efd76d`, `5e414d0c5`); neither changes
rules, observations or frozen evidence.

1. **Learning encodes only referenced definitions.** With autograd on,
   `ActorCritic._referenced_definitions` collects every catalog row a batch can
   index (visible cards, permanents, known hands, decklists, public history,
   relations) as unique rows (`SemanticCardEncoder.referenced_rows`), and
   `definitions(rows)` runs the GRU over those programs only, scattering them
   into the full-size table (other rows zero); `rows=None` selects every row
   through the same path. Without autograd, and in `PolicyInference`'s cached table, the
   full catalog is still encoded: the subset GRU differs from the full one by
   ~1e-8 rounding, and unprepared no-grad inference must stay bitwise equal to
   serving (`test_scope_rejects_autograd_closed_and_changed_policy`). Proof:
   `test_training_encodes_only_referenced_definitions_without_changing_learning`
   (outputs and parameter gradients within 1e-6/1e-5; unreferenced rows are
   poisoned with NaN and every source gets a definition no other source names,
   so dropping any source fails it). The worker test runs in 5.6 s (30 s timeout
   with the change disabled), with its regime and limit unchanged.
2. **`CURRENT_PACK_KEYS` is now `FROZEN_VOCABULARY_LABELS`.** Same nine values,
   byte-for-byte. Documented in `learning.py`, `ModelConfig.world` and the
   manabot README: one vocabulary, fresh recipes write `world`, older labels
   are accepted on read. Setup names (`AuthoredMatchup(pack_key=...)`,
   `AuthoredSetup`) are unchanged. `execution.py` still requires the
   `elemental-world-v2` label for the `elemental-uniform` curriculum (a frozen
   regime check, left as it is).

## Headless concept review at `eb127ca9a` (2026-11-02) — no change

Checked the remaining two-kind deck concept: `PlayerConfig.submitted` (admitted
against the world) versus authored setups run as given. Collapsing it so every
deck is admitted fails on a counterexample: every preset in `setups.json` has
decks with cards the world marks incomplete (Accumulate Wisdom, Lightning
Strike, Water Tribe Rallier, Cat-Gator, Suki, Kyoshi Warrior; 19 decks over all
9 presets fail `check_decks`). So `submitted` stays. Before T6, authored
packs admitted these cards by pack membership. Now they run with no admission
at all, which is already listed under "Checks weakened" below. The rules model
(faces on `Card`, Sagas, the turn-scheduling stacks), the learner-cost repair
and the frozen names (`pack_key`, `PlayerConfig.world`) match accepted
decisions. No open product choice. What's left is integrating #414 (still open
on 2026-11-02) and the gate.

## Decisions for Jack Heart's review

- **Retained authored-pack checkpoints.** Setup-bound checkpoints of authored
  packs no longer load (Compatibility, below); their token-embedding shapes
  were sized to pack vocabularies. Losing this follows from the accepted
  one-catalog decision and is reversible (the files remain; a reader can be
  added later). Question for Jack Heart: accept, or add a reader that maps
  their vocabulary onto the catalog? Not blocking implementation.
- **Copy of a transformed permanent.** Follows CR 707.8a rather than the earlier
  "cannot transform" proposal; a house rule would need to be requested.
- **Back-face colors** follow Scryfall face data (questions.md); check them
  against the printed cards.

## Check

2026-11-02 at `5e414d0c5`: `pytest tests/model/test_inference.py tests/model/test_policy_catalog.py tests/semantic/test_current_content.py` 39 passed, worker test 5.4 s; Rust last checked at the compress pass (debug `cargo test --lib --tests` fails only `checked_semantic_conformance_evidence_reproduces`, `--test cards` 526 passed, clippy clean, digest `02ac498d…2634`, length 355); broad pytest and Academy UI not run since the main sync (gate, after #414).

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

`GameState.content` is gone: readers use `default_content_pack()`, now
`&'static ContentPack`. Digest binding is proved by
`GameState::deterministic_hash_over(&ContentPack)`. Clone cost is proved by giving
cards large copy-on-write definitions (allocation receipt v2; the retained W2-208
receipt is v1). `Card.definition` stays `Arc`: that copy-on-write seam needs it,
and the direction only says "may". The conformance metamorphic relation
`equal_content_pack_reallocation_preserves_trace` no longer has anything to
reallocate, so it is removed; w17's `evidence.json` still lists it until the
pending re-record.

Proof at the refactor: catalog digest `6741f471…9827`, length 345 and the
id→name manifest hash `f67e67ee…b603` are identical on `2c1ba169f` and on the
branch; `content_pool("world")` and `world_identity()` give the same digest. The
later card merges moved the digest as intended: on HEAD it is `02ac498d…2634`,
and `CARD_DEFINITION_COUNT` = 355 = catalog length (330 eligible, 25 not deck
entries, including 8 back faces).

Measured cost before the learner repair above: authored vocabulary names now bind the
345-row world catalog. `SemanticCardEncoder("ur-lessons-vs-gw-allies", 64)`
goes from 31 definitions / 3,598 expanded tokens / 4.9 ms per `definitions()` to
345 / 51,909 / 45.3 ms (`elemental-world-v2` was 100 / 14,097 / 20.5 ms). The
Etude worker fixture regime (`ataraxos_move_scalar`, hidden 8, 2 updates) takes
5.07 s at `2c1ba169f` and 31.29 s on this branch, 26 s of it in `backward`.
`test_etude_job.py::test_worker_runs_real_learner_and_reports_admitted_policy`
exceeded its 30 s budget; with referenced-row encoding it runs in 5.4 s,
regime and limit unchanged.

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
through `current_binding_form`. `FROZEN_VOCABULARY_LABELS` are the names older recipes
gave their vocabulary, all binding the world catalog and accepted on read;
fresh recipes write `world`.
`MatchConfig.world` keeps a frozen regime's authored label and runs it like
`None`. Curated manifests keep `semantic_pack_key`, resolved as a setup name.
Public `register_card` stays for the content-sensitivity tests that extend a copy.

## Proposed Wave memory

Wave memory stays read-only in this pass. Candidates once T6 lands:
- One catalog (Jack Heart, 2026-11-02): authored setups are decklists; the "Shipped
  substrate" line on versioned packs and its per-pack identity need updating.
- A vocabulary bound to the whole catalog has a real learner cost (31 → 345
  definitions took the worker fixture from 5 s to 31 s, mostly in `backward`);
  encoding only the rows a batch references restores it. Keep inference on the
  full cached table: the subset GRU is not bitwise equal to the full one.

## Publish attempt (headless, 2026-11-02) — blocked

`pr-publish` ran at `eb127ca9a` and stopped without publishing. Jack Heart's
delivery order (steer c371253c) has MTG-238 #414 merging first and T6
integrating it afterward. #414 is still open: no review, mergeable, Engine Budget
failing. Until then T6 cannot be verified. Debug `cargo test` still fails
`checked_semantic_conformance_evidence_reproduces`, and the w17 re-record waits
on #414. HEAD already contains origin/main `a1a02a76f`, so `lf sync` has nothing
to bring in. Next: once #414 lands, integrate it, re-record, gate, then publish.
