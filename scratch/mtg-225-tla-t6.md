# T6 (MTG-225): Sagas, faces, face-aware copies and turn scheduling

Status (2026-11-03, reconciled after sync): all 16 cards and the accepted one-catalog change are implemented. Repairs are in `854d5e282`, compression in `ecba733ca`, and the combined head is `64b79d895` on main `c823c9c8f`, including MTG-235/238/239. [Draft PR #417](https://github.com/loopflowstudio/manabot/pull/417) is open at that head. Jack Heart's 2026-11-03 repair steer (`b77a8b6d-6066-48d4-be08-3007994aeb33`) retains one cohesive PR for review. Broad acceptance is pending; no auto-merge or landing is authorized.

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

## Completed simplification

Completed delete — do not maintain: live audit references to removed
`Game::new_with_content` in `reviews.json`, `index.jsonl` and
`scripts/audit_rules.py` now point to the surviving `Game::construct` without
changing rule dispositions, coverage claims, source hashes or historical reports.

Completed in this repair:

- `PayTargetsManaValue` and the targets-before-payment tests; retain paid-X choice, normal mana payment, reflexive trigger, negative target checks and response window.
- `ExileNamed` automatic `.take(count)` exile; retain naming, private zone permission, optional exact selection and shuffle through `SearchNamed`.
- Face-equality transform guard and unused `EffectFrame.source_face`; retain captured face identity for resolving printed abilities and use a transform-count witness for intervening transforms.
- Live-record-only inherited activation resolution; retain the lending definition on the existing stack object and frame. Departure triggers use the existing LKI snapshot.
- Foggy Swamp Visions' any-player-next-end-step approximation; retain both timing rules in the common delayed event.

Already removed by the accepted catalog/effect cut: `ContentPack::from_definitions`/pack key, `CardRegistry`, per-match content selection and same-pack error, `pools()` map, `Game::new_with_content`, per-pack token closure, authored admission plumbing, and `Effect::ExileTopCardPlayThisTurn`. Their required behavior lives on the one catalog, submitted-deck admission and generic `ExileTopCardsToPlay`. There is no parallel path. `BoundSemanticPack.pad_catalog` stays because projection tests use it; it is not a deletion target.

The surviving activation lookup is `stacked_activated_ability`: resolution and retargeting share printed, inherited and attached-grant selection. Retargeting a printed trigger also reads `source_face`, borrowing its effects instead of cloning them or consulting the current face. The Aang/Redirect Lightning regression failed before that cut (`.runs/t6-publication/compress-retarget-before.log`) and passes afterward. Private-search card enumeration is an iterator over the known owner, shared by offers and visibility; colored mana reduction subtracts matching symbols and sends only the remainder to generic. No remaining deletion targets were identified in these paths.

## Verification and remaining work

The repair evidence at `854d5e282` covers 69 focused debug Rust cases and 15 Python cases for payment/search, transform/version parity, inherited abilities, delayed timing, schedule identity and observation/text/Academy projection. Compression at `ecba733ca` adds 28 focused Rust passes and the Sozin text/policy/Academy pass, plus library/Python Clippy, formatting and changed-Python Ruff. These are focused proofs, not full-suite acceptance. Logs remain in `.runs/t6-publication/*-repair.log` and `compress-*.log`; the release cp312 extension was rebuilt and installed atomically.

The sync at `64b79d895` already regenerated the mutable w17 semantic kernel (14 cases, 3,473 Commands, 15 Phase rows), authored match and both parity receipts, Academy demos/navigation/trigger order, and current belief-v3 pins. Live/headless parity retained main's corrected 135 Commands with no divergence; public parity retained 66 commitments with no gaps. Focused sync evidence adds eight card/observability, five engine/admission and four Python passes. Logs: `.runs/t6-publication/sync-*.log`. Decision code 23 remains main's library-bottom ordering, T6 payment/search use 24/25, and choose-to-exile uses 26; loyalty/defense retain ordinals 7/8 and lore appends at 9. This supersedes the earlier 142-Command tape and the claim that producer refresh is still wholly outstanding.

Reconciliation inspected the shared payment/reflexive-trigger/search paths, retained activation lookup, catalog authority and directed tests against these logs. No new bounded implementation mismatch was identified. Before repair, gate at `a7ba7432c` passed 913 Rust tests (two ignored), model/admission/training/parity checks, 102 focused Python tests, 119 Academy unit tests and ten Playwright cases. That evidence established the earlier integration, not omitted Sozin behavior or the final combined source; exact commands and limits remain at the historical git reference above.

CI run `41744841447` passed 935 debug Rust tests (two ignored), protocol and semantic conformance, Academy UI, clean-machine play, 27 long integration tests and Engine Budget. Its four stale audit/test failures are repaired below. Main integration steps after its failed first invocation and the remaining Academy steps still need the repaired-head run. Successful unaffected checks remain reusable. Producer regeneration is only needed again if validation identifies a mismatch or its inputs change. Schema exports follow native/Pydantic authorities independently of receipt generation.

Only mutable w17 receipts may be rebound. Main's accepted 135 authored Commands, ordering/navigation prefixes, offer ids, presentation and intended outcomes remain the baseline. Historical receipts, retained checkpoints and frozen worlds remain untouched. PR validation copy includes the completed sync evidence and distinguishes passing checks from pending repaired-head CI. No further product decision is identified; Jack Heart's PR review remains required.

Engine Budget is CI-owned. Changed benchmark digests, an accepted cost or an incompatible reference require the PR candidate's own artifact and an explicit adoption reason. No local performance or playing-strength result is claimed.

CI run `41744841447` found stale audit anchors to deleted `Game::new_with_content`
and a fresh-checkpoint test expecting the historical 64-action default. Both are
reconciled with the surviving constructor and accepted fresh capacity of 512.
The main integration job also exposed a frozen architecture fixture using fresh defaults and Foggy Swamp Visions expecting battlefield-removal polarity. The fixture now uses `ObservationConfig.from_checkpoint` with its recorded v5 version, preserving the original hash; the heuristic test expects its existing unrated graveyard-copy behavior. No production behavior changes.
The raw pinned rules source was recovered with its original SHA-256; regeneration
changes only authorities/anchors across 331 rows, preserving all 3,322 dispositions,
source spans and hashes. Historical reports and receipts remain unchanged.

Check result (2026-11-03): `uv run --python 3.12 --extra dev pytest tests/infra/ tests/academy/test_clean_machine_verifier.py deploy/play/tests/` — 103 passed; `uv run --python 3.12 --extra dev --extra notebook --extra artifacts --extra reports pytest tests/model/test_world.py` — 17 passed; `uv run --python 3.12 --extra dev --extra notebook --extra artifacts --extra reports pytest tests/model/test_capacity.py tests/sim/test_heuristic.py -q` — 41 passed; pinned-source inventory verification, changed-file Ruff and `git diff --check` passed. Logs: `.runs/t6-publication/audit-*.log`; remaining combined-head checks stay with gate/CI.

## Wave memory

The code Graph's realign step records the accepted one-catalog direction and reusable payment/search, retained-state and sync-ordinal lessons in `.lf/wave/engine/MEMORY.md`. T6 remains pending review; focused evidence does not establish final acceptance. Engine has no immediate child memory directories in this checkout.
