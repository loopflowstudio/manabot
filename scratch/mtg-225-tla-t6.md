# T6 (MTG-225): Sagas, faces, face-aware copies and turn scheduling

Status (2026-11-03): all 16 cards and the accepted one-catalog change are implemented. Rules repairs are in `854d5e282`, compression in `ecba733ca`, and sync in `64b79d895` on main `c823c9c8f`, including MTG-235/238/239. Audit and checkpoint-fixture repairs through `fc9eb76b9`, plus fixture simplification in `a82a963aa`, are published in [PR #417](https://github.com/loopflowstudio/manabot/pull/417), open at `a82a963aa`. Jack Heart's 2026-11-03 repair steer (`b77a8b6d-6066-48d4-be08-3007994aeb33`) retains one cohesive PR for review. Broad acceptance is pending; no auto-merge or landing is authorized.

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

## Schedule projection implementation — 2026-11-03

Implemented locally: native `TurnState.public_schedule` supplies typed `ScheduleView`
through the live decision surface. Historical capture serialization, policy
inputs and rules identity remain unchanged. Current combat is shown for every
combat phase, including unrestricted combat; it requires no duplicate engine
flag. The native retained `resume_with` is shown when present, never guessed.

Deleted Academy's count-derived extra-turn/additional-combat sentences in
`PlayerArea.svelte`. Preserve the numeric protocol fields for retained captures
and policy consumers; the current UI reads the ordered native schedule instead.

`Schedule.svelte` renders ordered extra turns and native resume, current combat
and ordered pending combats with rules-derived attacker restrictions. Missing
historical data says “Schedule not recorded.” Schemas for Experience, replay,
Study and advice are regenerated; frozen capture bytes are untouched. Academy
usage guidance is updated. The release cp312 extension was installed atomically.

The real WebSocket proof in `tests/academy/test_schedule_recovery.py` plays Bumi,
observes pending then current land-creature restriction with zero pending count,
checks both closed-deck viewers, reconnects to the same frame and rejects a stale
Command without changing state/events. Native progression covers opposite turn
queues/resume and unrestricted combat after restricted combat; Svelte server
rendering covers ordered lists, restrictions, seat labels and historical unknowns.
No process-crash durability or browser-layout proof is claimed.

Check: focused debug `public_schedule` (2) and Bumi (1), Python schedule recovery
(1) and experience protocol (14), Vitest GameBoard (7) and protocol (5) all passed;
Svelte check, changed-file Ruff, formatting, CI test collection and diff checks
passed. Gate owns affected combined suites, browser layout/interaction checks and
final-head CI; publication remains the selected Graph's existing PR #417 step.
No merge or auto-merge is authorized. The supplied interactive review edits and
concurrently added `scratch/t7-handoff.md` are preserved.

## Delayed-trigger composition — accepted correction, 2026-11-03

Jack Heart's review rejects `NextEndStepForCreatedToken`: timing must not own
object selection. `CreateDelayedTrigger` captures an optional `ObjectSelector`
at creation; its effects use `ObjectSelector::Remembered`. Exact `ObjectRef`s
travel with the delayed trigger, pending occurrence, stack ability and suspended
resolution, separately from the triggering event's context. No selector is
re-evaluated against a later incarnation. End-step timing retains the existing
controller/any-player distinction. The adjacent target-death timing becomes
selector-based too: the death event watches the captured subjects, through the
existing native exact-object departure watcher.
Ordinary death/spell/upkeep behavior and expiry are preserved.

Observation projects remembered objects separately in v5 relation rows (kind 8),
with native/Python/text parity and catalog-derived tokens. Frozen v1–v4 widths
and exclusions stay unchanged. Hash contract is now 14; only mutable
current-world receipts may be refreshed. Each `ForEachTarget` iteration captures
its own copy. Directed tests cover source departure/control change, separate
cast-event context, hidden-hand projection, reentry both before firing and while
resolution is suspended, exact hash sensitivity and deterministic choice forks.
The concept page records the accepted correction and implemented schedule.

Check: `cargo test` filters `tla::` (430), `observability::` (55),
`resolution_tests` (13) plus the added privacy case pass in debug; rebuilt cp312
extension + focused tensor/text and both Academy recovery tests (4), all-target /
all-feature Clippy, changed-file Ruff and diff checks pass. Logs:
`.runs/t6-publication/delayed-*.log`. Gate owns the now-required mutable w17
receipt refresh, full affected suites and final-head CI before PR #417
publication; historical evidence is unchanged and no merge is authorized.

## Delete — do not maintain

Completed in this cut: removed `DelayedEvent::NextEndStepForCreatedToken`,
`DelayedEvent::TargetDiesThisTurn` and end-step `context` plumbing. Both cards
use generic timing + capture + ordinary sacrifice. Observability witnesses use
the surviving path; timing/death/upkeep behavior tests remain. The shared
inline-trigger constructor returns its created occurrence, removing three
separate last-trigger lookups. No remaining deletion targets in these paths.

Live audit references to removed
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

The compound replay fixtures now share `_attach_saved_sampler` instead of three copies of learned-sampler reload setup. Every checkpoint save reads `agent.observation_space`; configuration equality is checked after the final reload for either sampler. The fixture retains explicit 64-action compound inputs and fresh 512-action scalar inputs. Frozen defaults and production receipt validation remain unchanged. No further deletion targets were identified in this iteration.

## Verification and remaining work

Historical command/result details and contrary evidence are retained at `5c254fd5732a42da0966e1f5badf490f3307b4f3:scratch/mtg-225-tla-t6.md`, with local logs in `.runs/t6-publication/`. The original Sozin negative reproduction remains `sozin-payment-order.log`; the repair logs establish payment/search, transform, inherited-ability, delayed-timing and observation behavior. Earlier broad green suites did not establish the omitted Sozin choices. The release cp312 extension was rebuilt and installed atomically after the native repairs.

The sync at `64b79d895` refreshed mutable w17 kernel, authored/parity receipts, Academy demos/navigation/trigger order and belief-v3 pins. It retains main's corrected 135 authored Commands, explicit ordering prefixes and offer ids, with 66 public commitments and no parity gaps. Main's decision code 23 and counter ordinals 7/8 remain fixed; T6 appends payment/search at 24/25, choose-to-exile at 26 and lore at 9. Historical receipts, retained checkpoints and frozen worlds stay untouched. Regenerate current receipts only if inputs change or validation identifies a mismatch.

CI `41744841447` passed the debug Rust suite (935 tests, two ignored), protocol/semantic checks, Academy UI, clean-machine play, long integration and Engine Budget. Its failures exposed stale audit anchors, a fresh-checkpoint test expecting 64 actions, a frozen architecture fixture using fresh defaults and an incorrect Foggy Swamp Visions heuristic expectation. Subsequent fixes preserve pinned audit dispositions/source hashes, reconstruct the frozen v5 configuration through `from_checkpoint`, and leave the graveyard-copy heuristic unrated. CI `41746278124` at `454666106` then exposed two compound fixture failures: a 64-action agent was saved with a separate fresh 512-action space. `fc9eb76b9` removes that contradiction; receipt validation was correctly rejecting it. The negative reproduction and four passing scalar/compound × uniform/learned cases are retained in `compound-checkpoint-before.log` and `compound-checkpoint-repair.log` under `.runs/t6-publication/`. The later simplification in `a82a963aa` checks configuration equality after the final sampler reload. This resolves the supplied iteration feedback's implementation request without changing production defaults or receipt validation.

At reconciliation on 2026-11-03, run `41746278124` had finished with the Academy failure above and cancelled main/long integration jobs; its Rust, Python unit, protocol, semantic, Academy UI, clean-machine and Engine Budget jobs passed. Run `41747455801` at `5c254fd57` remained in progress with Academy UI cancelled. Published head `a82a963aa` has pending CI `41747957440` and release run `41747957342`. Unaffected successful checks remain reusable, but these partial/cancelled runs do not establish repaired-head acceptance. Verification still owns combined-head CI, the retained-session search recovery proof below, and refreshing PR validation copy before Jack Heart's review.

Check: `uv run --python 3.12 --extra dev --extra notebook --extra artifacts --extra reports pytest tests/sim/test_local_update_compound.py -q` — 25 passed; changed-file Ruff, formatting and `git diff --check` passed. This includes scalar/compound save, reload and payment replay with both samplers. Verification owns remaining combined-head CI, the retained-session search recovery proof below, and PR validation copy before Jack Heart's review. Engine Budget remains CI-owned: any reference adoption requires this PR's own same-runner artifact and an explicit reason; no local speed or playing-strength claim follows. No merge or auto-merge is authorized.

## Concept review — 2026-11-03

Outcome: **nothing to change and no open product choice**. This headless review
of `a82a963aa` rechecks the earlier `454666106` review and leaves Jack Heart's
accepted experience and model unchanged.
It does not establish broad acceptance, complete CI, or authorize landing.
The current review page is `scratch/concept-review.html`; the older preserved
page remains historical proposal evidence.

The consequential mismatch was treating an equivalent final board as an
equivalent decision. The repaired shared authority represents the actual
interaction: choose X or decline, complete payment, choose reflexive-trigger
targets within the retained budget, then allow responses. Paying zero and
declining stay distinct. If a target leaves during the response window, only
remaining legal targets resolve; committed payment is not recomputed or
refunded. No combined browser payment/target confirmation is implied.

For chapter II, choose a catalog name, inspect the searched zones privately,
then exile individual eligible cards or stop and shuffle. Each selection
commits through the current native decision; no draft selection or undo
lifecycle is introduced. The chooser's permission, remaining count and
resolution frame belong to native state. Academy projects that permission,
and policy/text/search consume the same authority. Physical visibility ends
with the search; learned hand-definition knowledge remains. Reconnection must
recover that actual pending decision and its remaining choices, not reopen
the name menu or restore already-exiled cards.

The remaining concepts earn their place: one catalog owns definition identity
and digest; decklists select cards; a card's face is distinct from its object
identity; transform history and captured inherited abilities survive the
changes they guard; ordered schedules retain the next turn/combat and its
restriction. These are native facts, not parallel Academy state. Existing
usage docs already describe the one-catalog contract, so no replacement docs
or skill instructions are proposed. Historical vocabulary labels and saved
observation defaults remain compatibility inputs, not additional live packs.

The intervening checkpoint-fixture repair follows the same ownership rule:
save the agent with its actual observation configuration, then restore that
configuration when loading it, including through the learned sampler. A fresh
default is not a description of an existing agent. The inspected change uses
`save_checkpoint(agent, agent.observation_space, ...)` and checks configuration
equality after the final reload; it changes neither production defaults nor
receipt validation. It supplies no migration for old authored-pack weights.
No new checkpoint lifecycle, compatibility shim, usage-doc change or implement
direction is needed. This pass inspected source and the saved evidence; it did
not rerun tests or refresh CI status.

Counterexamples to preserve are a target leaving after payment, stopping a
search before four matches, an unchosen nonmatching hand card remaining known,
and a face returning to its original side before an older transform resolves.
The inspected directed tests represent these cases. The generic revision
contract rejects stale Commands; the inspected Sozin consumer test establishes
projection and encoder parity, **not** a Sozin-specific reconnect or
process-crash durability proof. The next recovery proof should resume a
retained session after one search selection, reject its stale resubmission,
and verify the remaining offers and viewer permissions. This is verification
of the accepted contract, not a new product decision or a new persistence
design. Existing gate/CI work and Jack Heart's review remain the next
acceptance boundary; this review changes no implementation direction.

## Interactive concept review — 2026-11-03, head `01ed2fbcb`

This review, requested by Jack Heart, supersedes the earlier headless conclusion
of “nothing to change” for the presentation of scheduled turns and combats.
It does not change implementation, authorize shipping, or restart the Graph.
The existing Sozin recovery test now supplies the previously missing retained
session proof; process-crash persistence remains outside that proof.

**Accepted experience:** the human can understand the acting agent's public
observation and complete legal actions. One catalog, immediate committed search
choices, separate payment/reflexive-trigger decisions, and preserved historical
checkpoint defaults remain accepted. No alternative rules lifecycle is proposed.

**Mismatch found by source inspection:** `TurnState.extra_turns`, `resume_with`,
`additional_combats` and `combat_attackers` retain ordered scheduling and attacker
restrictions. `Game::continuation_frames` projects these publicly as kinds 11/12,
including order and a rules-derived attacker description. Academy's
`PlayerArea.svelte` instead renders per-player counts from `PlayerView`; its
turn sentence says each player takes their extra turns “after this one”. Counts
cannot distinguish opposing extra-turn orders. The additional-combat count does
not explain Bumi's land-creature restriction, and reaches zero during the last
extra combat even while that restriction remains active. This is a presentation
gap, not evidence that execution offers illegal attacks. No browser reproduction
was run in this review.

**Proposed completion of the accepted contract (not yet implemented):** show
the committed upcoming turn order and current/pending additional combats with
their public restrictions. Derive a typed, viewer-safe schedule projection from
the native authority for ExperienceFrame and Academy; do not infer order from
counts, parse card text, copy the scheduler into the client, or expose private
continuation frames. Keep turns and combats separate: they are different timing
domains, not a guessed unified future-game timeline. Describe only scheduled
events, not predictions about unresolved spells or triggers. Existing frozen
captures without this data mean unknown, not an empty schedule.

**Usage-doc draft, proposed for Academy guidance:** “Before choosing an action,
read the scheduled turns and extra combats. The list shows events already
scheduled, in their execution order; later effects can change it. A restricted
combat states which creatures may attack, and ordinary attack offers remain
authoritative. After reconnecting, the list comes from the recovered game state.”
The existing one-catalog README guidance and agent skill instructions stay
unchanged; they already require shared authority and viewer-safe recovery.

**Walkthrough and counterexamples:** Bumi schedules a land-creature-only combat;
the person sees that restriction before the next decision, enters that combat,
and still sees it after its pending count becomes zero. Reconnecting restores
the same current restriction and remaining schedule. Two positions with one
extra turn per player but opposite queue order must display different sequences.
A newly scheduled turn may change the order. A stale Command cannot mutate the
schedule. An unrestricted extra combat must not inherit an earlier restriction.

**Consequences and next proof:** native state remains the only owner; projection
and schema carry explicit public data, UI renders it read-only, and recovery uses
the existing retained-session frame. Policy tensors and frozen receipts keep
their contracts. Verify reversed queues, current versus pending restrictions,
unrestricted combat, viewer privacy and reconnect through the real consumer path.
No new persistence service or scheduling operation is needed. Source inspection
supports the gap; these proposed presentation checks have not been executed.

**Decision accepted by Jack Heart, 2026-11-03:** selected 1a ("1a1a"). Complete
the schedule projection and recovery proof within T6 before landing. The
proposed completion and usage text above are now accepted implementation
direction, not implemented behavior. Preserve the existing PR #417 and all
other accepted T6 work; return to review after publication. No merge or
auto-merge is authorized. This is a small completion of the shared-observation
promise, not a redesign of the whole turn UI. The page at
`scratch/concept-review.html` is the review surface.

## Wave memory

The code Graph's realign step records the accepted one-catalog direction and reusable payment/search, retained-state and sync-ordinal lessons in `.lf/wave/engine/MEMORY.md`. T6 remains pending review; focused evidence does not establish final acceptance. Engine has no immediate child memory directories in this checkout.

## Publication verification — 2026-11-03

`lf sync` completed cleanly against main `c823c9c8f`, checkpointing the supplied
reconciliation/concept-review edits as `d6e7ec078`. The accepted production
implementation is unchanged from `a82a963aa`; no native rebuild or receipt
refresh was needed in this pass.

The previously outstanding retained-session search proof now passes in
`tests/academy/test_named_search_recovery.py`, included in Academy Integration.
It reaches Sozin chapter II through ordinary WebSocket Commands, exiles one
card, reconnects with the same frame and remaining three-card budget, and
checks chooser-only inspection. Duplicate retry and a stale new Command leave
the state and event count unchanged. Stopping early keeps the selected card
exiled and ends inspection visibility. This proves retained live-session
recovery, not process-crash persistence.

Validation: 77 tests passed across named-search recovery, optional-payment
recovery, choice navigation, Elemental launcher, forced-step automation, CI
selection and release configuration. Ruff formatting/checks and
`git diff --check` passed. Logs: `.runs/t6-publication/search-recovery-suite.log`.
Earlier native, learner, protocol, receipt and UI results remain applicable to
their unchanged inputs. CI `41747957440` at `a82a963aa` was cancelled after
Python unit, semantic and clean-machine successes; it was not a full-head pass.
The sync started `41748514307`, still in progress during this verification;
publication will start a new run for the test/CI-wiring change. Final-head CI
remains pending and must not be described as green. Publish one ready PR for
Jack Heart's review; no auto-merge, landing or Task completion is authorized.
