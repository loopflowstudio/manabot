# T6 decisions and assumptions — 2026-11-03

Jack Heart's accepted scope is all 16 T6 cards plus one catalog (2026-11-02). His 2026-11-03 steer requires completing the detected Sozin gaps and auditing earlier approximations, then publishing for review. No merge or auto-merge is authorized. The current plan is [mtg-225-tla-t6.md](mtg-225-tla-t6.md).

## Resolved by implementation

The earlier claims that targets-before-payment had the same outcome, searching could automatically exile four matches, a face comparison represented intervening transforms, inherited abilities could disappear when Koh left, and any next end step sufficed for Foggy Swamp Visions were incorrect. They are replaced by shared payment/reflexive-trigger/search decisions, captured transform/ability state, LKI and controller-specific delayed timing. Ordered turn/combat scheduling and restrictions now reach identity and public observation. These repairs are acceptance work, not new product choices.

Private searching reveals the searched zones only to the chooser. Temporary physical visibility ends on completion, but previously inspected hand definitions remain known. Both native samplers preserve exact visible cards during search. Academy maps native-visible opponent hand/library cards to the existing agent inspection list with original ownership and zone; replay/Study opponent.hand stays redacted. Two-player hand knowledge uses the existing definition-count authority without fabricating reveal events. Naming covers real catalog names including back faces; no unsupported outside-catalog names are invented. Fresh action capacity 512 fits this catalog; saved omitted capacity remains 64 and oversized input still fails explicitly.

## Retained design decisions and review limits

- A double-faced card is one definition function with a back face registered immediately after it; the back face has no independent key/Scryfall id and is not a deck entry. Its mana value reads its front face. Leaving the battlefield returns the card to its front face; transform retains permanent identity.
- CR 707.8a permits a copied transformed permanent to be a transforming double-faced token. The earlier cannot-transform proposal is superseded by the printed rules, not a house rule. Back-face colors use Scryfall face data (Sozin black; other back faces colorless); retained review caveat is to compare printed cards. No new external data assumption was made in this repair.
- Saga entry emits CountersPut for lore and chapter I; other entry counters retain existing behavior. Exactly-X targeting uses `max_x` with nonzero minimum; up-to-X uses zero minimum. Yangchen's optional draw is a resolution Modal inside its target clause.
- Bending events retain their domain meaning: waterbend once per paid cost, firebend once per resolved combat-mana instruction, earthbend/airbend once per resolved instruction. Four-kind turn history is shared state.
- One catalog preserves frozen vocabulary labels on read. The referenced-row learner optimization remains intact; full-table no-grad serving remains bitwise aligned. Old authored-pack checkpoint weights/bindings are not migrated and no retained binary was loaded. This is a review-visible compatibility limit, not an implicit promise to add a shim.
- Authored presets retain their bypass of submitted-deck construction checks. Main now admits every defined non-token, non-fixture front face; incomplete-card exclusions no longer exist.
- Current w17 is mutable: definition/token changes and hash contract 15 require affected receipt refresh, not a world bump. Earlier worlds, frozen advice/experiments and retained checkpoints are never regenerated to make checks green. Engine Budget uses same-runner CI evidence; no synthetic load or local speed claim.
- MTG-238 and MTG-235 are already integrated. Their historical blockers/delivery orders are no longer outstanding T6 work. The unrelated printed-P/T protocol choice remains outside T6.

## Historical evidence

Complete inherited T2–T5, decklist, MTG-235/236/237/238, integration and previous publication notes are retained at `a7ba7432c9d03188ec9affb50412dd2f3197ea61:scratch/questions.md` and `:scratch/mtg-225-tla-t6.md`. The latter preserves prior gate results and the negative Sozin reproduction. Older linked refs there remain historical evidence, not direction. No uncommitted evidence was removed: the only pre-curation scratch edit was the repair delete list, retained in the current plan.

## Main sync — 2026-11-03

Completed at `64b79d895`: T6 incorporates main `c823c9c8f`, preserving the single catalog and main's
definition-based admission (no incomplete-card exclusion), plus both sets of
card fixes. Main has published decision code 23 for library-bottom ordering
and counter ordinals 7/8 for loyalty/defense; T6's choose-to-exile is appended at
26 and lore at 9. T6 payment/search keep 24/25. This resolves the unpublished
T6 collisions without changing main's retained meanings. Refreshed current w17 receipt
Commands use main's corrected Rallier/Accumulate Wisdom traces, with derived
runtime values rebound to the combined implementation. Historical receipts
remain untouched.

Main supersedes the old 142-Command tape with 135 Commands after Rallier
correctly excludes Suki with power above three. Preserve that accepted trace
and the explicit Accumulate Wisdom ordering Commands in navigation prefixes.

## Schedule projection assumption — 2026-11-03

Current combat is displayed for both ordinary and extra combat, with the native
restriction if any. This avoids inventing a persisted “is extra” flag that the
accepted display does not need. `resume_with` is shown only when native state
retains it; the client does not predict the next normal turn from counts.
Missing historical schedule data remains unknown. These reversible presentation
choices preserve Jack Heart's accepted 1a direction.

## Main v6 sync — 2026-11-03

Merged main `3b122de9078c985f846a0f3c5a3a42ebca3b1182` at `bf7631c6a`, retaining
v6's configurable fact capacity (fresh default 512), T6's fresh 512-action
capacity, and the checkpoint reader's historical v1/64-action defaults. V5
remains fixed at 128 facts; explicit saved values remain authoritative. Preserve
both branches' parity tests, extending T6 scheduling/transform coverage to v6.
Main's MTG-213 experiment decisions remain historical context at
`3b122de9078c985f846a0f3c5a3a42ebca3b1182:scratch/questions.md`; they do not
replace T6's scope or authorize experiment work in this sync.

## Delayed lifetime projection — 2026-11-03

The small typed API uses `DelayedFrequency::{Once, EachMatch}` and
`DelayedExpiry::{ThisTurn, Never}`; no continuous-effect duration is reused.
The existing `this_turn` observation field remains a derived expiry fact. An
optional frequency code, unknown in historical captures, gets relation row
kind 9 in v5/v6; existing columns and v1–v4 exclusions remain fixed. Canonical
hashing moves to contract 15 because the waiting record serialization changed.
Gate owns the resulting mutable-w17 refresh.
