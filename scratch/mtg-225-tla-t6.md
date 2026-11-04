# T6 (MTG-225): Sagas, faces, face-aware copies and turn scheduling

Status (2026-11-03): accepted T6 behavior and the frozen-pilot isolation repair
are implemented locally. HEAD `920bbb52ae524fab1e5fd58f7958b24f9395e1f5`
incorporates main `fb5ad2c1372f9c967b550f0b8ae0f84883f75563` and its v7
shared-effect projection. Gate owns combined validation, mutable receipt
assessment, republication and final-head CI. Existing PR #417 remains one
cohesive delivery; stop for Jack Heart's review. No merge or auto-merge authorized.

## Accepted direction

Jack Heart accepted all 16 T6 cards plus the one-catalog simplification on
2026-11-02. T5 and MTG-238 precedence are fulfilled. His 2026-11-03 repair
steer requires correct choices, not merely the same final board. He selected
1a for Academy's ordered public schedule ("1a1a"), then required composable
delayed selector capture, and selected 1a for independent event/frequency/expiry.
All are implemented; no product question remains. Details and attribution:
[t6-direction.md](t6-direction.md), [questions.md](questions.md), and the preserved
[scratch concept review](concept-review.html). HTML examples are illustrations,
not browser captures. T7's independent direction stays in `t7-handoff.md`.

The outcome is correct play for humans and bots through native rules authority,
directed debug tests for every obligation, complete viewer-safe choices and
native/Python/text/token observations. New input state preserves frozen versions;
only mutable current-world evidence may be refreshed. Existing review edits and
other contributions remain preserved.

## Publication findings and repairs

The prior gate demonstrated that Fire Lord Sozin selected targets before payment and that chapter II automatically exiled four matches. Those were acceptance failures, not equivalent outcomes. Its negative reproduction is `.runs/t6-publication/sozin-payment-order.log`.

- **Fire Lord Sozin:** `OptionalManaPaymentX` chooses an affordable X (including zero), then uses ordinary `PaymentProgress`. Declining creates no trigger. Paying creates a distinct `WhenYouDo` occurrence carrying X and the damaged opponent; targets are chosen afterward, with distinct creature cards from that opponent's graveyard and total mana value at most X. Retargeting uses the same budget. The stack gives a separate priority window and rechecks targets at resolution.
- **The Rise of Sozin II:** naming includes every actual card name in the catalog, including back faces and excluding tokens/fixtures. `SearchNamed` is a resumable private decision over the opponent's graveyard, hand and library, permitting zero through four exact eligible cards and then shuffling. The searcher can inspect nonmatching cards too; the owner does not gain library visibility. Physical visibility expires on completion, while inspected hand definition knowledge persists. Academy carries these native-authorized cards through the existing chooser inspection surface, preserving owner/zone and replay/Study opponent-hand redaction; the text renderer recognizes the native search permission. Both ordinary determinization and possible-world materialization preserve inspected cards while searching.
- **Transform guards:** a permanent's transform count and the activation/trigger snapshot replace face equality. An intervening round trip prevents a further transform. Departed sources resolve their captured face through LKI.
- **Koh:** inherited activations retain their lending definition before costs, so source departure or a changed inheritance record cannot erase/change the ability. Inherited departure triggers read LKI abilities. Retargeting uses the same retained activation owner.
- **Foggy Swamp Visions:** the delayed sacrifice waits for its controller's next end step, even if cast on another player's turn. The existing any-next-end-step rule remains for Fire Navy Trebuchet.
- **Scheduling and observation:** pending extra-turn order/resume and extra-combat order/restrictions enter deterministic identity (now hash contract 15) and public v5/v6 continuation frames (11/12). Transform counts and combat restrictions use fact kinds 9/10; captured inheritance joins the existing definition rows. Decision codes 24/25 describe payment-X/search. Delayed-event code 7 distinguishes the controller's next end step. Native/Python/text paths share these facts; v1–v4 widths and exclusions remain frozen.
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

## Public schedule and delayed abilities

`TurnState.public_schedule` projects ordered extra turns and native `resume_with`,
current combat (including ordinary unrestricted combat), and ordered pending
combats with rules-derived attacker restrictions. `Schedule.svelte` renders it;
`PlayerArea.svelte` no longer invents order from counts. Numeric protocol fields
remain for retained captures and policy. No duplicate “is extra” engine flag is
needed. Historical absence says “Schedule not recorded.” Experience, replay,
Study and advice schemas and Academy usage docs carry this contract.

`CreateDelayedTrigger` resolves an optional `ObjectSelector` at creation. Exact
`ObjectRef`s travel through the watcher, pending occurrence, stack and suspended
resolution as `Remembered`, separately from triggering-event context. Each
`ForEachTarget` iteration captures its own copy; a later incarnation cannot
replace it. Original controller and source/LKI behavior persist after departure.
Visions watches its controller's end step; Trebuchet watches any end step.

Event predicates, `DelayedFrequency::{Once, EachMatch}` and
`DelayedExpiry::{ThisTurn, Never}` compose independently. `Once` consumes the
watcher on the match even if the occurrence is countered or does nothing.
Cleanup expires waiting records, never queued/stacked occurrences. Exact-object
`Leaves` watchers retire on any departure independently of frequency/expiry;
earthbend explicitly uses once/no expiry. All six authored creation sites keep
their existing semantics. This is not a printed-trigger migration or second DSL.

Remembered references use v5+ relation kind 8; optional frequency uses kind 9,
with absent historical frequency unknown. `this_turn` remains derived expiry.
Scheduling uses frame kinds 11/12; transform/combat facts use kinds 9/10. Exact
captures introduced hash contract 14, independent lifetime fields contract 15.
`managym/README.md` owns the authoring contract. Native, Python, text and
catalog-derived tokens carry these facts without changing v1–v4 exclusions.

## Integration and compatibility

Main's library-bottom ordering keeps decision code 23; T6 payment/search use
24/25 and choose-to-exile 26. Loyalty/defense retain ordinals 7/8; lore appends 9.
Current w17 inherits main's corrected 135-Command Rallier tape and explicit
Accumulate Wisdom ordering, not the older 142-Command tape.

Main v6 (`3b122de90`, integrated at `bf7631c6a`) supplies configurable facts:
fresh 512, v5 fixed 128. T6 fresh actions are independently 512; saved omitted
version/actions retain v1/64 through `ObservationConfig.from_checkpoint`.
Explicit saved values remain authoritative; timestamp rank normalization is /128.
Main v7 (`fb5ad2c13`, integrated at `920bbb52a`) adds shared effects once while
fresh inputs stay v6. T6 schedule/delayed parity now includes v7. The shared
native effect constructor uses the catalog directly. No upstream experiment work
or allocation is inherited.

Frozen pilot scheduling/analysis fixtures explicitly supply the protocol's
compatible world identity, including the fresh-process real analysis CLI. They
never run games, training or allocation. Unpatched `frozen`, run declaration and
evaluation separately reject the actual changed runtime. Changed-input tests
reach observation admission using temporary copies; frozen protocol/regime bytes
stay intact. The production guard pins v6/256 facts/**64 actions**, not fresh
512-action defaults. `AGENTS.md` records this isolation rule.

## Delete — do not maintain

Completed cuts; no outstanding deletion target:

- Authored pack multiplicity: `ContentPack::from_definitions(pack_key)`, per-game
  content/pack selection, same-pack admission, `CardRegistry`, `pools()` and
  `WORLD_POOL` storage. One catalog and authored decklists survive; the public
  `"world"` name and frozen vocabulary/evidence labels remain.
- Fused delayed variants `NextEndStepForCreatedToken`, `TargetDiesThisTurn`,
  `DiesThisTurn`, `CreatureDiesThisTurn`, `YouNextCastThisTurn`, `NextEndStep` and
  `NextUpkeep`; inferred runtime boolean expiry and event-specific consumption.
  Selector capture, event matchers and independent lifetime fields replace them.
- Count-derived Academy scheduling; empty `TOKENS_AWAITING_PRODUCER` exception;
  duplicate `settle_until_modal` and Sozin catalog-to-mode logic; `jeong_configs`
  and its obsolete pack-selection-only test. Shared helpers, actual offered
  names and Academy's deck/sideboard/catalog coverage survive.
- Synthetic pilot dependence on live-world compatibility and implicit fresh
  action capacity. Runtime admission and historical scheduling/analysis remain.

Surviving code uses `TargetRequirement::one` and existing token/exile constructors.
Authored setups build one card-key index for all presets. Retargeting names the
other targets of its clause once for both distinctness and mana-value budget.
Python deck checking returns diagnosed issues directly, without a never-failing
`Result` wrapper. The pilot subprocess fixture uses ordinary multiline Python.
These reductions add no mechanics, identity or observation contract changes.

## Evidence, limitations and remaining gate

The previous local publication gate at `d21fae20d` passed 948 debug Rust checks
(two ignored), Clippy, 474 combined Python checks (one skipped) and the repaired
54-test text module, 118 Academy checks, 41 launcher/verifier checks, 122 UI
checks and isolated Playwright (25 passed, two skipped). A stale text assertion
was corrected to event-only timing when historical frequency is unknown.
Logs: `.runs/t6-publication/final-*.log`; full commands/results remain at
`d21fae20d3ef0e65fa7814124785f08034d6741d:scratch/mtg-225-tla-t6.md`.
These precede the v7 sync and do not certify the current combined head.

That gate refreshed only mutable w17 evidence: kernel 14 cases/3,473 Commands;
authored tape 135 Commands; live/headless/replay parity 816 viewer checks;
public parity 66 commitments. Navigation preserves offer ids and labels.
Frozen worlds, advice, experiments and checkpoints were untouched. Current
content digest was `d44e9319815ff8724c2ed901a64976942be88b4a8c278a5170f0fc4db6b572d6`.
Gate must assess any receipt refresh needed after v7 and the latest source edits.

Contrary evidence: final-head CI at that publication had **27 pilot failures,
278 passes, one skip**; a world-admission failure was reproduced locally.
The [headless demo note](t6-demo.md) records the failure, five passing recovery/
parity checks, seven rendering checks and bounded repair direction. It explains
why implementation iterated despite the earlier local gate. The repaired full
pilot module passed 33 checks; v7 sync verification passed four native
shared-effect checks, 16 parity/text checks and one exact discovery check.
Logs: `.runs/t6-demo/{pilot-isolation,sync-v7-native,sync-v7-consumers,sync-v7-discovery}.log`.
The extension was rebuilt and atomically installed (`sync-v7-extension.log`).

Real WebSocket tests cover Bumi's pending-to-current restriction with zero
pending count, both closed-deck viewers, reconnect and stale rejection;
Fatal Fissure's waiting/expired watcher, exact capture/clone identity; and
Sozin's one selection, remaining private search budget, duplicate/stale rejection
and early stop retaining exile. Native tests cover opposite schedules/resume,
unrestricted combat after restricted combat, once/repeat, expiry, countering,
queued/stacked survival, source departure, exact reentry and hash/fork identity.
They prove retained sessions, not process-crash durability. Svelte checks prove
rendered schedule content, not browser geometry; there is no separate Bumi
browser walkthrough. No strength, retained-checkpoint binary compatibility or
new performance result is claimed. CI owns the same-runner Engine Budget;
the reference and 10% tolerance are unchanged.

Local sync `lf sync --manual` / `--continue` completed at `920bbb52a` with no
unresolved paths. GitHub's `CONFLICTING` response still names published
`d21fae20d`; gate must recheck the authoritative base and mergeability before
republication, then inspect final-head CI and stop for Jack Heart's review.

Compression verification: `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test
--manifest-path managym/Cargo.toml -j 2` (setup, Sozin, Redirect Lightning and
Jeong copy filters) — 11 debug checks passed; `uv run` pytest pilot,
decklist-import and current-content modules — 60 passed. Release `maturin build`
passed and the cp312 extension was installed atomically; rustfmt, Ruff and
`git diff --check` passed. Logs: `.runs/t6-demo/compress-*.log`. `lf context`
confirms both scratch and memory under their token/byte limits. Gate retains
affected suites, mutable receipt assessment and publication; no merge authorized.

## Retained history

Full pre-compression local plan (including uncommitted pilot/v7 repair evidence)
is preserved at `.runs/t6-demo/pre-compress-plan.md`. Earlier accepted review,
integration and negative reproductions are in the pinned plan at `d21fae20d`
and its precise historical links (`a7ba7432c`, `bf7631c6a`, `7c790f78e`).
`.runs/t6-publication/sozin-payment-order.log` preserves the original negative
payment reproduction. Concept-review HTML remains unchanged. Engine memory is
read-only in this step and within budget; realign owns reconciliation.
