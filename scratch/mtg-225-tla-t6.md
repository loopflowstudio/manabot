# T6 (MTG-225): Sagas, faces, face-aware copies and turn scheduling

Status (2026-11-03, implementation repair): all 16 cards and the accepted one-catalog change are implemented. MTG-238 is integrated at `373bb08ca` against main `44388ad42`. Jack Heart's 2026-11-03 steer (`b77a8b6d-6066-48d4-be08-3007994aeb33`) requires repairing the publication findings and other recorded approximations, then publishing one cohesive PR for review. No auto-merge or landing is authorized.

The accepted outcome is correct play for humans and bots through the shared native rules authority, with directed debug tests for every obligation, complete viewer-safe choices and policy/text observations. `scratch/t6-direction.md` retains Jack Heart's 2026-11-02 scope decisions. Full earlier implementation, integration and publication history is retained at `a7ba7432c9d03188ec9affb50412dd2f3197ea61:scratch/mtg-225-tla-t6.md` and `:scratch/questions.md`.

## Publication findings and repairs

The prior gate demonstrated that Fire Lord Sozin selected targets before payment and that chapter II automatically exiled four matches. Those were acceptance failures, not equivalent outcomes. Its negative reproduction is `.runs/t6-publication/sozin-payment-order.log`.

- **Fire Lord Sozin:** `OptionalManaPaymentX` chooses an affordable X (including zero), then uses ordinary `PaymentProgress`. Declining creates no trigger. Paying creates a distinct `WhenYouDo` occurrence carrying X and the damaged opponent; targets are chosen afterward, with distinct creature cards from that opponent's graveyard and total mana value at most X. Retargeting uses the same budget. The stack gives a separate priority window and rechecks targets at resolution.
- **The Rise of Sozin II:** naming includes every actual card name in the catalog, including back faces and excluding tokens/fixtures. `SearchNamed` is a resumable private decision over the opponent's graveyard, hand and library, permitting zero through four exact eligible cards and then shuffling. The searcher can inspect nonmatching cards too; the owner does not gain library visibility. Physical visibility expires on completion, while inspected hand definition knowledge persists. Academy carries these native-authorized cards through the existing chooser inspection surface, preserving owner/zone and replay/Study opponent-hand redaction; the text renderer recognizes the native search permission. Both ordinary determinization and possible-world materialization preserve inspected cards while searching.
- **Transform guards:** a permanent's transform count and the activation/trigger snapshot replace face equality. An intervening round trip prevents a further transform. Departed sources resolve their captured face through LKI.
- **Koh:** inherited activations retain their lending definition before costs, so source departure or a changed inheritance record cannot erase/change the ability. Inherited departure triggers read LKI abilities. Retargeting uses the same retained activation owner.
- **Foggy Swamp Visions:** the delayed sacrifice waits for its controller's next end step, even if cast on another player's turn. The existing any-next-end-step rule remains for Fire Navy Trebuchet.
- **Scheduling and observation:** pending extra-turn order/resume and extra-combat order/restrictions enter deterministic identity (hash contract 13) and public v5 continuation frames (11/12). Transform counts and combat restrictions use fact kinds 9/10; captured inheritance joins the existing definition rows. Decision codes 24/25 describe payment-X/search. Delayed-event code 7 distinguishes the controller's next end step. Native/Python/text paths share these facts; v1–v4 widths and exclusions remain frozen.
- **Action capacity:** fresh Python/native encoders allow 512 actions so the whole catalog name menu fits. Saved configurations lacking `max_actions` retain 64 through `from_checkpoint`; explicit saved values are unchanged. Over-capacity inputs still fail rather than truncate.

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
| The Legend of Yangchen // Avatar Yangchen | supported; decision code 26 (choose to exile) |
| The Rise of Sozin // Fire Lord Sozin | supported; any card name, private optional search, paid-X reflexive trigger |
| Aang, Swift Savior // Aang and La | supported; airbends spells, transforms in place |
| Aang, at the Crossroads // Aang, Destined Savior | supported; next-upkeep delayed transform |
| Avatar Aang // Aang, Master of Elements | supported; bending events and four-kind history |
| Bender's Waterskin | supported |
| Bumi, Unleashed | supported; additional land-creature combat |
| Koh, the Face Stealer | supported; effect-record inheritance retained through activation/departure |
| Ember Island Production | supported; copies the face that's up, transformed |
| Fire Lord Azula | supported; spells cast while attacking are copied |
| Foggy Swamp Visions | supported |

## One catalog and learning (accepted 2026-11-02)

One process-wide catalog owns stable name-ordered `CardDefId` rows and content identity. Back faces immediately follow their fronts. Authored presets are decklists over it. Every match uses `default_content_pack()`; `WorldIdentity` reads the same catalog. Ward lowering remains the catalog's documented normalization. `Card.definition` retains Arc for the copy-on-write content-sensitivity seam; the direction permitted, but did not require, a static reference.

With autograd, `ActorCritic._referenced_definitions` gathers all rows a batch can index (cards, permanents, known hands, decklists, history, relations), encodes only those programs and scatters them into the catalog table. No-grad/serving still uses the full cached table because subset GRU rounding is not bitwise equal. Its behavioral/gradient proof and worker budget remain unchanged; this repair does not modify that optimization. `FROZEN_VOCABULARY_LABELS` preserves all nine historical labels, each bound to the catalog; fresh recipes use `world`.

The refactor alone preserved content bytes and row ordering at `2c1ba169f`; later card/rules changes deliberately changed the digest. Old authored-pack checkpoint weights/bindings are not migrated; manifest rejection and current load paths are tested, but no retained checkpoint binary has been loaded. No strength or historical compatibility claim follows.

Authored presets retain their bypass of submitted-deck construction checks. Main `c823c9c8f` removes bug-based admission exclusions: every defined non-token, non-fixture front face is playable. `PlayerConfig.submitted` still selects construction validation. Review-visible consequences remain: edited setup metadata alone no longer fails checkpoint loading (execution checks actual decks), authored mirrors need no pack selection, and any pair of Elemental decks receives the roster binding by membership.

## Delete — do not maintain

Completed in this repair:

- `PayTargetsManaValue` and the targets-before-payment tests; retain paid-X choice, normal mana payment, reflexive trigger, negative target checks and response window.
- `ExileNamed` automatic `.take(count)` exile; retain naming, private zone permission, optional exact selection and shuffle through `SearchNamed`.
- Face-equality transform guard and unused `EffectFrame.source_face`; retain captured face identity for resolving printed abilities and use a transform-count witness for intervening transforms.
- Live-record-only inherited activation resolution; retain the lending definition on the existing stack object and frame. Departure triggers use the existing LKI snapshot.
- Foggy Swamp Visions' any-player-next-end-step approximation; retain both timing rules in the common delayed event.

Already removed by the accepted catalog/effect cut: `ContentPack::from_definitions`/pack key, `CardRegistry`, per-match content selection and same-pack error, `pools()` map, `Game::new_with_content`, per-pack token closure, authored admission plumbing, and `Effect::ExileTopCardPlayThisTurn`. Their required behavior lives on the one catalog, submitted-deck admission and generic `ExileTopCardsToPlay`. There is no parallel path. `BoundSemanticPack.pad_catalog` stays because projection tests use it; it is not a deletion target.

The surviving activation lookup is `stacked_activated_ability`: resolution and retargeting share printed, inherited and attached-grant selection. Retargeting a printed trigger also reads `source_face`, borrowing its effects instead of cloning them or consulting the current face. The Aang/Redirect Lightning regression failed before that cut (`.runs/t6-publication/compress-retarget-before.log`) and passes afterward. Private-search card enumeration is an iterator over the known owner, shared by offers and visibility; colored mana reduction subtracts matching symbols and sends only the remainder to generic. No remaining deletion targets were identified in these paths.

## Verification and remaining work

2026-11-03 check result: focused debug `cargo test` filters for Sozin, Koh, transform round trips, Foggy timing, observability, scheduling hashes and decision codes passed **69 cases**; focused `uv run --extra dev --extra play pytest` passed **15 cases** across transform/version parity, field/text/schema coverage, the real Sozin text/Academy/policy path and Bumi schedule-fact parity (the last two runs resolve the earlier fixture failure and reuse 13 unaffected passes). `cargo clippy --lib --features python -- -D warnings`, workspace fmt and changed-Python Ruff passed. Release cp312 rebuilt and installed atomically; all six protocol schemas regenerated. Logs: `.runs/t6-publication/*-repair.log`, `repair-{clippy,wheel}.log`; broader gate/receipt refresh remains deferred to gate.

Compression check: `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2` with state filter `mana::tests` and card filters `sozin`, `koh`, `avatar_aang`, `trusty_boomerang`, `aang_swift_savior`, `redirect_lightning` — **28 passed**; `uv run --extra dev --extra play pytest tests/semantic/test_text_observation.py::test_sozin_search_plays_through_text_policy_and_academy_projection -q` — **1 passed**; library/Python Clippy, workspace fmt and diff checks passed. Release cp312 rebuilt/installed atomically and fresh import succeeded (`w17`, 355 rows). Logs: `.runs/t6-publication/compress-*.log`; full suites and affected receipts remain owned by gate. `lf context` confirms memory and scratch below both configured limits.

Gate owns the affected full suites and automated acceptance after this repair: debug engine/card/state tests and all-target clippy, Python model/training/current-world admission and observation checks, protocol/UI checks, and the current-world producer refresh. The new rules/content/hash inputs invalidate the previous affected receipt checks; unrelated successful checks can be reused.

Only mutable w17 receipts may be rebound: semantic-kernel, authored-match, authored live parity, public commitment parity, Academy demos/navigation/trigger-order, and current belief-v3 pins. Preserve main’s accepted 135 authored Commands (superseding the earlier 142 after Rallier’s corrected Suki filter), choice prefixes/offer ids, presentation and intended state outcomes. Historical receipts, retained checkpoints and frozen worlds remain untouched. Schema exports follow their native/Pydantic authorities, independently of receipt generation.

Before this repair, gate at `a7ba7432c` passed 913 existing Rust tests (2 ignored), model/admission/training/parity checks, 102 focused Python tests, Academy's 119 unit tests and 10 Playwright cases. That established integration, not the omitted Sozin behavior; it is not final-source evidence. Exact command/results and limits remain in the git reference above.

Engine Budget is CI-owned. If rules/catalog changes move its digests, or the reference no longer builds, adoption requires the PR candidate's own CI artifact and an explicit reason; no local performance result is claimed. Publication is one cohesive T6 PR for Jack Heart's review, not a merge.

## Proposed Wave memory

Wave memory is unchanged. Candidates after accepted delivery: one catalog replaces per-pack identity; referenced-row learning restores the enlarged vocabulary's worker cost while full-table serving preserves bitwise inference; private search permissions must cover observations, both samplers and remembered hand facts; schedule/transform history affecting future play belongs in deterministic identity and shared observation.
