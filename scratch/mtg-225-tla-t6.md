# T6 (MTG-225): Sagas, faces, face-aware copies and turn scheduling

Status (publication gate, 2026-11-03): all 16 cards, one catalog, Academy's
accepted schedule projection, exact delayed capture and independent lifetime
composition are implemented. Sync retained main `3b122de90` without conflicts;
`16336f2a0` preserves the local review/reconciliation edits. Mutable w17 receipts
are refreshed to hash contract 15 and combined validation is complete below.
PR #417 remains the one cohesive delivery boundary. Final publication-head CI
must still be checked; no auto-merge or landing is authorized.

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

Publication gate completed locally; remaining acceptance is final publication-head
CI and Jack Heart's review. No implementation or product choice is deferred.
Old authored-pack checkpoint migration/retained-binary compatibility remains
unproven, and no process-crash durability or playing-strength claim is made.
The dated verification below supersedes the earlier pending refresh/combined
validation notes.

## Wave memory

This is the selected code Graph's realign step. Engine memory records the
accepted public schedule, captured-object and independent lifetime contracts,
including v6 integration and evidence limits. The live gate work remains here. Engine has no immediate child directories or
child memories in this checkout. Training's v6 capacity lesson was read as
upstream evidence; its experiment decisions remain outside T6.

## Accepted delayed-lifetime review — Jack Heart, 2026-11-03

Jack Heart selected **1a**: independently compose event matching, firing
frequency and expiry for existing delayed abilities within T6. The prior open
choice and deferral alternative are superseded. The small typed API may reuse
vocabulary only where its semantics match; migration of all printed triggers,
a second DSL and hypothetical cards are outside this correction.

The authoring contract is implemented in `managym/README.md`: selectors capture
exact incarnations at creation; event context stays separate. A once-only watcher
is consumed on its first match even if the occurrence is countered or its effects
do nothing. Expiry removes waiting watchers, not queued or stacked occurrences.
Original controller and exact death-watch retirement remain independent rules.
Academy and policy consume viewer-safe projections of that native state.

The accepted review page remains `scratch/concept-review.html`. The complete
review sequence, including the superseded headless open-choice outcome and the
received decision, is retained at
`7c790f78e7008701c5d1bcb7903a8b83693bd093:scratch/mtg-225-tla-t6.md`.
The implementation below and its recorded tests resolve those counterexamples;
combined validation and review remain separate acceptance work.

## Delayed lifetime implementation — 2026-11-03

Removed: `DiesThisTurn`, `CreatureDiesThisTurn`,
`YouNextCastThisTurn`, `NextEndStep`/`NextUpkeep` event names; inferred
`this_turn` in resolution; event-arm-specific consumption; runtime boolean
expiry. Event-only names, `DelayedFrequency::{Once, EachMatch}` and
`DelayedExpiry::{ThisTurn, Never}` replace them on the existing authoring/runtime
owner.
Exact-object departure retirement, original controller, captures and existing
card behaviors/tests remain on that path. Observation `this_turn` remains the
derived expiry projection; an optional frequency code (unknown for historical
captures) uses new relation row kind 9 without widening or repurposing frozen
columns. No printed-trigger migration or new card definition.

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

Removed: `TOKENS_AWAITING_PRODUCER` and its empty
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

Recorded evidence: `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2` with `-p managym --lib`, `--test authored_match_tests`, and `-p managym-cards --test cards` filters `tla::`, `observability::`, `lea::`, `stx::` passed 136/2/430/56/4/19 checks; changed Koh/Sozin tests passed their focused reruns. `uv run --extra dev --extra play pytest tests/verify/test_observation_parity.py::test_waiting_delayed_trigger_rows_agree_and_stay_out_of_v4 -q` passed 2 checks. Release extension rebuilt/installed atomically; pre/post world and learning-catalog digests match; format and diff checks pass (`.runs/t6-compress/`). Earlier lifetime consumer evidence remains in `.runs/t6-publication/lifetime-*.log`. Gate owns combined affected suites, mutable w17 receipt refresh to hash 15/content identity and publication of PR #417 for Jack Heart's review; no merge or auto-merge is authorized.

Reconciliation check: `git diff --check` — passed; existing focused logs reused, with combined validation and mutable-w17 refresh deferred to gate.

## Concept-review outcome — 2026-11-03

Reviewed the accepted experience against `7c790f78e` and the preserved local
reconciliation. **Nothing to change; no open product choice.** Jack Heart's
schedule and delayed-lifetime decisions are implemented, not proposals awaiting
another answer. No new implementation direction or scope is introduced.

The concrete interaction is Bumi's additional combat: read the committed
restriction, advance into that combat, then reconnect before choosing attackers.
The pending count becomes zero while the current restriction remains visible;
the recovered frame and native offers still govern the choice. `TurnState`
owns execution order and restrictions, `public_schedule` projects them, and
Academy renders them without a second scheduler. A recording without that
projection remains unknown. Equal counts with opposite turn orders and an empty
queue during restricted combat remain the counterexamples to count-based UI.

For delayed abilities, the important distinction is waiting versus already
triggered. Creation captures exact objects; event matching creates an occurrence;
frequency consumes the watcher; cleanup expires only waiting records. Countering
the first occurrence cannot re-arm a once-only watcher, and reconnect or a fork
cannot select a replacement incarnation. The small independent lifetime types
and existing occurrence/resolution machinery express those contracts. Merging
them back into an event name or a single generic duration would lose behavior.

The one-catalog model likewise leaves deck selection as a deck choice, not a
second world choice. Sozin's payment, reflexive targeting and private search
remain deliberate native decisions; recovery retains committed payment or exile
and the actual remaining choice. No browser cancellation or inferred authority
is added. Existing `academy/README.md`, `managym/README.md` and shared-authority
instructions already describe the accepted experience; no usage or skill edit
is needed. Prior 16-card scope and compatibility limits remain in force.

Next proof is the existing gate work above: refresh affected mutable-w17
receipts, validate the combined head and return PR #417 for Jack Heart's review.
This review inspected source, recovery tests and recorded focused logs; it did
not rerun suites, render the page, establish crash durability, or certify CI.
The following loop-or-next receives this outcome and selects navigation.

## Publication verification — 2026-11-03

`lf sync` completed without conflicts and preserved all review edits. The
accepted scope and diff were inspected against actual PR base main. No Rust or
production-rule change was needed in this pass; the installed release extension
from compression matches content digest
`d44e9319815ff8724c2ed901a64976942be88b4a8c278a5170f0fc4db6b572d6`.

Refreshed only mutable w17 kernel, authored/parity, Academy and belief receipts.
The kernel's 14 cases keep their command counts/outcomes (3,473 Commands).
The authored receipt keeps all 135 Commands, semantic and presentation events;
only state and command-receipt identities change. Live/headless/replay parity
has 816 viewer checks and no divergence; public parity retains 66 commitments,
the same command tape and zero provider gaps. Navigation adds native schedule
projection and runtime identities; every retained offer id and label replays.
Frozen worlds, advice, experiment evidence and checkpoints were not rewritten.

Evidence is in `.runs/t6-publication/final-*.log`:

- Complete debug Rust workspace: 948 passed, two ignored. Clippy all targets/
  features and rustfmt passed.
- Combined Python model, semantic, native/Python parity, belief, arena admission,
  monitoring, active training, compound checkpoint/replay and package boundaries:
  474 passed, one skipped, one stale text assertion failed. The assertion assumed
  timing implied "next" despite historical frequency being unknown; updated it
  to the event-only "controller's end step" contract. The complete text module
  then passed all 54 checks. No behavior was weakened or production code changed.
- Academy recovery/protocol/navigation/parity and CI configuration: 118 passed.
  Server protocol: three passed. Launcher/clean-machine verifier unit checks:
  41 passed.
- Svelte checking: zero errors/warnings; 122 UI unit tests passed; build passed.
  Isolated release Playwright: 25 passed, two skipped, including combat layout,
  choice navigation, deck entry and trigger-order recovery. Its servers exited.
  Schedule-specific evidence remains native progression, real WebSocket recovery
  and Svelte rendering; no separate Bumi browser walkthrough was performed.
- Changed-file Ruff and diff checks passed.

Publication updates PR #417's description to include schedule, delayed capture,
independent lifetime composition, v6 integration and these proof limits. Final-head
CI, including same-runner Engine Budget, remains separate from local evidence.
The existing performance reference and 10% tolerance are unchanged; no local
speed claim or reference adoption is made. Stop for Jack Heart's review.
