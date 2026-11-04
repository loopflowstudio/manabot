# T6 (MTG-225): Sagas, faces, face-aware copies and turn scheduling

Status (implemented 2026-11-03): all 16 cards, one catalog, Academy's accepted
schedule projection, exact delayed capture and Jack Heart's accepted 1a lifetime
composition are implemented. The current worktree retains main `3b122de90`
through merge `bf7631c6a`; existing PR #417 remains the delivery boundary.
No scope choice remains. Hash contract 15 and changed definition tokens require
mutable-w17 receipt refresh and combined validation by gate. No final-head CI
acceptance, publication, auto-merge or landing is claimed by this implementation.
One cohesive PR returns to Jack Heart for review.

The accepted outcome is correct play for humans and bots through the shared native rules authority, with directed debug tests for every obligation, complete viewer-safe choices and policy/text observations. `scratch/t6-direction.md` retains Jack Heart's 2026-11-02 scope decisions. Full earlier implementation, integration and publication history is retained at `a7ba7432c9d03188ec9affb50412dd2f3197ea61:scratch/mtg-225-tla-t6.md` and `:scratch/questions.md`.

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

## Schedule projection implementation — 2026-11-03

Implemented at `614afa3cd`, retained in `bf7631c6a`: native `TurnState.public_schedule` supplies typed `ScheduleView`
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

Recorded focused evidence covers native schedule progression, Bumi, Academy
schedule recovery/protocol, Svelte rendering and schema conformance; Svelte
checking and CI collection also passed. These headless checks establish order,
restrictions and retained-session behavior, not browser geometry. Final combined
validation remains outstanding. The interactive review artifacts and T7 handoff
are preserved.

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

Observation projects remembered objects separately in v5/v6 relation rows (kind 8),
with native/Python/text parity and catalog-derived tokens. Frozen v1–v4 widths
and exclusions stay unchanged. That capture correction introduced hash contract 14; the subsequent lifetime
composition uses 15. Only mutable current-world receipts may be refreshed. Each `ForEachTarget` iteration captures
its own copy. Directed tests cover source departure/control change, separate
cast-event context, hidden-hand projection, reentry both before firing and while
resolution is suspended, exact hash sensitivity and deterministic choice forks.
The concept page records the accepted correction and implemented schedule.

Recorded debug evidence includes 430 TLA tests, 55 observability tests and 13
resolution tests, with additional exact-identity/fork and hidden-hand cases.
The rebuilt extension passed focused tensor/text and both Academy recovery
checks; Clippy and changed-file Ruff passed. Logs remain under
`.runs/t6-publication/delayed-*.log`. These results precede the v6 sync and are
reusable only where inputs are unchanged.

## Main v6 integration — 2026-11-03

Main `3b122de90` supplies configurable fact capacity through policy input v6.
The merge at `bf7631c6a` retains fresh v6/512 facts and T6's fresh 512 actions;
v5 remains fixed at 128 facts, and omitted saved inputs retain v1/64-action
semantics through `ObservationConfig.from_checkpoint`. Explicit saved values
remain authoritative. Timestamp normalization stays rank / 128 independently
of capacity. These are distinct fact/action capacities, not a single limit.
The release cp312 extension was rebuilt after the merge; the recorded sync
suite passed 21 checks (`sync-v6-tests.log`). No T6 experiment execution or
historical receipt migration follows from the upstream training work.

The merged transform/schedule fixture covers v4/v5/v6. Reconciliation extended
the real Fatal Fissure delayed-capture parity case to v6 as well: native and
Python agree on the waiting trigger's remembered object, while v4 excludes it.
No production implementation change was needed.

## Accepted review decisions

- **Jack Heart, 2026-11-02:** all 16 T6 cards and the one-catalog simplification
  belong in one PR. MTG-238 precedes T6; that integration is complete.
- **Jack Heart, 2026-11-03, repair steer:** payment-before-reflexive-targeting,
  optional exact named-card search and the other detected approximations are
  acceptance work. A matching final board cannot replace the required choices.
- **Jack Heart, 2026-11-03, option 1a ("1a1a"):** Academy must show native ordered
  extra turns/resume and current/pending combats with public restrictions.
  The former count-derived sentences are deleted. The schedule and usage text
  are implemented, superseding the review's proposed/not-yet-implemented status.
  Turns and combats remain separate; historical missing data means unknown.
- **Jack Heart, 2026-11-03, delayed-trigger feedback:** timing, selector capture
  and effects compose. Exact incarnations are captured at creation, separately
  from event context. `NextEndStepForCreatedToken` and `TargetDiesThisTurn` are
  deleted; generic timing, `Remembered` and ordinary effects replace them.

The current concept page is `scratch/concept-review.html`; its examples are
illustrations, not browser captures. The older preserved page remains proposal
evidence. Full prior review text, implementation/delete inventory and publication
chronology remain at
`bf7631c6a1d229c1405ea800bd57e493aa66c45c:scratch/mtg-225-tla-t6.md`.
No uncommitted review evidence was removed.

## Verification and remaining work

The Sozin retained-session search proof is complete in
`tests/academy/test_named_search_recovery.py`: one selection, reconnect with the
remaining budget and private inspection, duplicate/stale rejection, then early
stop without undoing the exile. Schedule recovery has its own real WebSocket
proof. Both are wired into Academy Integration. Neither establishes
process-crash durability. The earlier 77-test consumer pass is recorded in
`search-recovery-suite.log`; recovery is no longer missing implementation work.

Earlier CI failures identified stale audit anchors and contradictory checkpoint
fixtures. Those repairs preserve audit dispositions/source hashes and save the
agent's actual observation configuration, including after learned-sampler reload.
The four scalar/compound × uniform/learned cases and 25-test fixture suite passed.
Negative reproductions and detailed historical results remain in
`.runs/t6-publication/` and at
`5c254fd5732a42da0966e1f5badf490f3307b4f3:scratch/mtg-225-tla-t6.md`.
They do not certify the subsequent schedule, capture and v6 changes.

Remaining acceptance work:

1. Gate owns the mutable w17 receipt refresh affected by delayed-trigger content
   and hash contract 15: semantic kernel, authored match and parity corpora,
   Academy demos/navigation/trigger-order captures, and dependent belief pins.
   The checked-in kernel still has v13 hashes. The prior 135-Command / 66-public-
   commitment trace, explicit library ordering and duplicate-label offer ids
   are the baseline; differences require explanation, not silent replacement.
   Frozen worlds, advice, experiment evidence and retained checkpoints stay fixed.
2. Gate owns affected combined-head debug Rust, learner/model, world admission
   and monitoring, protocol/semantic and Academy suites under
   `docs/development.md#validation-by-change-area`. Focused successes are reused
   where inputs are unchanged. Browser layout/interaction proof remains with
   gate/CI; native, WebSocket and Svelte rendering evidence is not that proof.
3. At reconciliation, GitHub reports PR #417 at `bf7631c6a`, open with no
   auto-merge; Rust, Python, integration, protocol, UI and Engine Budget checks
   are in progress. Final-head acceptance and accurate PR validation copy remain
   outstanding. Engine Budget uses this PR's own same-runner artifact and an
   explicit reason for any reference adoption, never a local speed claim.
4. Publication and Jack Heart's review remain the delivery boundary. There is
   no newly identified substantial implementation gap or product choice in this
   reconciliation. Old authored-pack checkpoint compatibility remains unproven;
   no migration or retained-binary admission is claimed.

Check: `uv run --extra dev pytest tests/verify/test_observation_parity.py::test_waiting_delayed_trigger_rows_agree_and_stay_out_of_v4 -q` — 1 passed across v4/v5/v6; changed-file Ruff and `git diff --check` passed.

## Wave memory

This is the selected code Graph's realign step. Engine memory records the
accepted public schedule/captured-object contracts and v6 integration limits;
the live gate work remains here. Engine has no immediate child directories or
child memories in this checkout. Training's v6 capacity lesson was read as
upstream evidence; its experiment decisions remain outside T6.

## Interactive concept review: delayed abilities — 2026-11-03

Requested by Jack Heart after the selector-composition discussion. Source
inspection at `bf7631c6a` plus current working changes confirms the accepted
capture correction is implemented. This pass changes review artifacts only;
no implementation, publication, Graph navigation or further approval follows.
The schedule completion and all previous T6 requirements remain accepted.

**Experience:** a card author describes the event, the objects to retain, the
ordinary effects, and how long/how often the ability waits. A player receives
the actual resulting trigger and legal choices; disconnecting cannot recreate
an expired watcher or forget its captured object. Academy and policy consume
viewer-safe state from the same authority.

**Implemented and accepted:** `CreateDelayedTrigger.remember` resolves a selector
at creation; exact references survive waiting, placement, stack and suspended
resolution. `Remembered` is separate from `Triggering`. Source departure or
control change does not reassign the existing delayed ability. Foggy Swamp
Visions and Trebuchet use generic end-step timing and ordinary sacrifice.
Current directed tests include independent captures, incarnation changes before
and after firing, original controller, hidden references, hash sensitivity and
choice forks. This review inspected those tests; it did not rerun them or certify
combined CI or durable process-crash recovery.

**Pre-implementation mismatch (now removed):** `DelayedEvent::YouNextCastThisTurn` includes an event,
first occurrence and expiry in one variant; `CreatureDiesThisTurn` implies repeat
until cleanup. Resolution chooses the runtime `this_turn` flag from the variant,
and the event matcher chooses consumption by its match arm. A new combination
therefore needs another event variant even when its event and effects exist.
This is an authoring/ownership issue, not a newly demonstrated card behavior bug.

**Accepted and implemented (Jack Heart selected 1a below):** for the existing delayed-effect slice,
separate event matching from firing policy (once or each matching occurrence)
and expiry (cleanup this turn or no stated expiry). Capture remains creation-time
and effects remain ordinary selectors/verbs. Use existing vocabulary where it
has the same semantics; do not reuse continuous-effect duration blindly because
its source/object-dependent expiry means something else. Do not migrate every
printed `TriggerCondition`, build a second DSL, or add hypothetical cards.
Illustrative spellings on the page are not a fixed API. Preserve death watchers'
exact watched incarnation and their departure retirement independently of this
lifetime split. Captured subjects and event-supplied subjects remain distinct.

**Authoring contract (now in managym/README.md):** “Choose the event independently of
how often the delayed ability fires and when it stops waiting. Capture selectors
when creating the ability. The first matching event consumes a once-only watcher,
even if its occurrence is countered or its effects later do nothing. A repeating
watcher remains until its stated expiry. Expiry removes the waiting watcher,
not occurrences already on the stack.” Existing capture and viewer-safety docs
and skill instructions remain unchanged; no new skill procedure is needed.

**Counterexamples / next proof:** compare two matching events before cleanup
(once fires once, repeating fires twice), no matching event before expiry
(nothing fires next turn), countered first occurrence (once does not re-arm),
expiry with an occurrence already queued (occurrence survives), source departure,
and captured-object reentry (new incarnation is untouched). Preserve Fatal
Fissure, Trebuchet, Visions, Lesson-cast and upkeep behavior. A retained-session
reconnect/fork must retain capture, original controller, expiry and remaining
firing policy; do not infer these from UI text or replay event history. Update
canonical hashing, viewer-safe native/Python/text/tokens and mutable current-world
receipts if the representation changes; frozen evidence stays frozen.

**Historical scope choice (resolved by the decision below):** 1a completes this decomposition for existing delayed
abilities within T6 (recommended); 1b accepts the implemented selector correction
and leaves lifetime/occurrence fusion as a recorded limitation. Jack Heart subsequently selected 1a; the decision below supersedes this question. Page: `scratch/concept-review.html`.


### Historical headless concept-review outcome — 2026-11-03 (superseded)

**Open product/scope choice, for loop-or-next:** Should T6 also make firing
frequency and expiry independently composable for the existing delayed abilities
(1a), or retain the implemented event variants and record that limitation (1b)?
The saved interactive review explicitly leaves both alternatives unaccepted.
Jack Heart's accepted steer separates event timing, captured subjects and effects
and requires preserving one-shot/expiry behavior; it does not explicitly accept
the subsequent independent lifetime authoring proposal. No answer is inferred.

Source inspection confirms the remaining distinction: `state/src/ability.rs`
(`DelayedEvent`) combines next/this-turn with the event; `flow/resolution.rs`
(`CreateDelayedTrigger`) derives runtime expiry from that variant;
`flow/triggers.rs` (`delayed_triggers_for_event`) consumes watchers in particular
match arms; `flow/damage.rs` (`end_turn_modifiers`) removes this-turn watchers.
`flow/zones.rs` separately retires exact-object death watchers on departure.
This evidence supports the saved authoring proposal, not a new demonstrated
failure of existing cards. Normal play, countered occurrence, cleanup and
reconnect/fork counterexamples above remain the next proof if 1a is accepted.

The accepted schedule, capture correction, all 16 cards, one catalog, viewer
safety and publication boundary remain intact under either choice. The page is
updated to lead with this unanswered choice and retain the accepted schedule
requirements. Only review artifacts changed in this step; no new test run, code change,
CI certification, publication or navigation is claimed. Outcome: open choice.

### Decision received — Jack Heart, 2026-11-03

Jack Heart answered **1a** to the delayed-lifetime review in the primary Task
conversation. This supersedes the open-scope-choice and headless open-choice
outcomes above. Complete independent event matching, firing frequency and expiry
for the existing delayed abilities within T6. The proposed contract, authoring
doc addition, constraints and counterexamples above are now accepted direction;
this entry does not claim implementation or verification. Preserve the earlier
accepted schedule/capture work and all T6 scope. The live Graph should iterate
implementation and return the existing PR #417 for review; no landing or
auto-merge is authorized. No product choice remains from this review.

## Delayed lifetime implementation — 2026-11-03

Delete — do not maintain (completed): `DiesThisTurn`, `CreatureDiesThisTurn`,
`YouNextCastThisTurn`, `NextEndStep`/`NextUpkeep` event names; inferred
`this_turn` in resolution; event-arm-specific consumption; runtime boolean
expiry. Replace them with event-only names, `DelayedFrequency::{Once, EachMatch}`
and `DelayedExpiry::{ThisTurn, Never}` on the existing authoring/runtime owner.
Keep exact-object departure retirement, original controller, captures, and all
existing card behaviors/tests on that path. Keep observation `this_turn` as the
existing derived expiry projection; add an optional frequency code (unknown for
historical captures), encoded as a new relation row kind 9 without widening or
repurposing frozen columns. No printed-trigger migration or new card definition.

The matcher now only determines an occurrence and its event context. One shared
frequency check consumes `Once` watchers before occurrences enter placement;
cleanup reads only expiry. `Leaves` retains its separate exact-incarnation
retirement. Earthbend explicitly authors once/no expiry. All six authored
creation sites preserve their earlier semantics. No parallel runtime path remains.

Compression uses `TargetRequirement::one` throughout the branch's affected set
definitions and the existing token/exile constructors, keeping one owner for
defaults. Lifetime proofs share only their life-gain setup; event, frequency and
expiry remain explicit in each case. The relation encoder's v5/v6 fields share
one guard, and empty watcher-consumption loops need no separate condition.

Delete — do not maintain (completed): `TOKENS_AWAITING_PRODUCER` and its empty
exception path (all tokens now have creators); `settle_until_modal` (use shared
`pass_until`); the Sozin test's duplicate catalog-to-mode algorithm (choose the
actual offered name); and `jeong_configs` with its obsolete pack-selection test
that only asserted two players. Academy's Jeong setup tests still cover deck,
sideboard, asset and catalog identity in both seats. No deletion target remains.

Native debug cases cover both frequencies and both expiries against the same
upkeep/death matchers, hash sensitivity and deterministic forks; no-match expiry,
countered first occurrence, queued/stacked survival, exact death-watch retirement,
source departure/control change, hidden captures and captured reentry. Real
Academy reconnect preserves the waiting Fatal Fissure record and expired absence,
capture identity and stale rejection. Native/Python v4/v5/v6 parity covers once
and repeating cards, and text covers explicit frequency, expiry and historical
unknown frequency. Catalog-derived tokens cover both independent enum choices.
The pre-existing vector test's removed `world` assignment migrated to `submitted`.

Check: `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2` with `-p managym --lib`, `--test authored_match_tests`, and `-p managym-cards --test cards` filters `tla::`, `observability::`, `lea::`, `stx::` passed 136/2/430/56/4/19 checks; changed Koh/Sozin tests passed their focused reruns. `uv run --extra dev --extra play pytest tests/verify/test_observation_parity.py::test_waiting_delayed_trigger_rows_agree_and_stay_out_of_v4 -q` passed 2 checks. Release extension rebuilt/installed atomically; pre/post world and learning-catalog digests match; format and diff checks pass (`.runs/t6-compress/`). Earlier lifetime consumer evidence remains in `.runs/t6-publication/lifetime-*.log`. Gate owns combined affected suites, mutable w17 receipt refresh to hash 15/content identity and publication of PR #417 for Jack Heart's review; no merge or auto-merge is authorized.
