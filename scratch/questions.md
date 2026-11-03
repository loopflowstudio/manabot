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
- Authored presets retain their bypass of submitted-deck admission because some include incomplete cards. The plan records weakened assertions and actual-deck execution checks; removing this distinction is outside the accepted change.
- Current w17 is mutable: definition/token changes and hash contract 13 require affected receipt refresh, not a world bump. Earlier worlds, frozen advice/experiments and retained checkpoints are never regenerated to make checks green. Engine Budget uses same-runner CI evidence; no synthetic load or local speed claim.
- MTG-238 and MTG-235 are already integrated. Their historical blockers/delivery orders are no longer outstanding T6 work. The unrelated printed-P/T protocol choice remains outside T6.

## Historical evidence

Complete inherited T2–T5, decklist, MTG-235/236/237/238, integration and previous publication notes are retained at `a7ba7432c9d03188ec9affb50412dd2f3197ea61:scratch/questions.md` and `:scratch/mtg-225-tla-t6.md`. The latter preserves prior gate results and the negative Sozin reproduction. Older linked refs there remain historical evidence, not direction. No uncommitted evidence was removed: the only pre-curation scratch edit was the repair delete list, retained in the current plan.
