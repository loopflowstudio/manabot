# Elemental sacrifice and landfall contracts (MTG-162)

The native registry and `elemental-sacrifice-v1` compiled pack contain the 17
owned real cards selected on 2026-10-20. Five basics and the Ally, Human Soldier,
Clue, Food and Treasure tokens complete the 27-definition closure. The two
40-card validation rosters exercise that closure; final ten-deck admission
belongs to MTG-164. This is development world w8, not a strength baseline or
world freeze.

## Rules and payment ownership

`SacrificeCost` describes a selected permanent, an optional mana alternative,
and whether the source is excluded. `Payment` retains the announced spell or
activation while choices finish. Mana and the selected sacrifice commit before
responses; targets remain the originally announced targets. Self-sacrifice and
tap costs remain on the activated ability definition. Activated stack objects
retain their exact source reference after sacrifice or removal. Life payment is checked
before activation and committed with the other costs.

Chosen-color activated mana abilities do not use the stack. Their immediate
color choice preserves the payer's priority and can resume an announced cost,
waterbend payment, or a resolving demanded payment. Feasibility counts each
physical mana source once, and mandatory payments exclude colors that cannot
finish the announced cost. Sacrifice candidates cannot also be consumed as a
mana source. Existing fixed-source production remains in the same payment path.

`PermanentSacrificed` is distinct from destruction or death and matches triggers
against the pre-departure battlefield. Replacement of the graveyard destination
does not erase the sacrifice. Last-known facts preserve creature animation for
death triggers and effective keywords for damage after a source departs. A once-per-turn trigger is limited at event collection, keyed
by exact source incarnation and ability, and resets with the turn.

Basic-land search has one native operation and one private decision owner.
Count, tapped entry, conditional untap and source return are typed parameters.
Multiple selections remove each found card before the next offer and shuffle
once on completion, including failure to find. Fabled Passage counts the found
land when testing four lands after the shuffle. Merchant's optional choice is
whether to search; searching and finding nothing still returns its exact source
and shuffles. A departed or returned source is not followed into a new object.
Search candidates are derived from the current library and visible only to the
chooser; they do not become pinned library knowledge under determinization.

New choice descriptions come from native typed operations. Academy displays
those descriptions without interpreting card names or rules text. Ordinary
play retains its lazy training dependencies. Study's pre-command `ed2` address
and separately validated landmark Command are unchanged; frozen `erd1`
evidence retains its stronger recorded-command binding.

## Directed behavior

`managym/tests/rules/sacrifice_cards.rs` exercises ordinary actions and semantic
Commands, including stale rejection, replay equality and nested exact rollback.

| Cards | Directed rules |
| --- | --- |
| Bastion of Remembrance; Callous Inspector | Soldier entry, creature-death drain, self-damage and Clue; animated-land last-known creature status |
| Curious Farm Animals | Death life gain and optional activation target, with sacrifice paid even when no target is chosen |
| Cycle of Renewal | Resolution-time land sacrifice, two distinct basic selections, tapped entry and one completed search |
| Deadly Dispute | Selected additional cost before responses, two draws, usable Treasure |
| Deadly Precision | Four-mana alternative and the sacrifice alternative paid using immediate Treasure mana |
| Fabled Passage; Prismatic Vista | Distinct life and entry rules; four-land untap including the found land; Tracker triggers on both entries |
| June, Bounty Hunter | Another-creature restriction, own-turn activation, Clue and two-draw evasion |
| Lotus Cobra; Sazh's Chocobo; Tireless Provisioner | Separate landfall mana, counter and token decisions |
| Tireless Tracker | Entry Clue and sacrifice counter, including fetch-land integration |
| Mayhem Devil; Tolls of War | Own and opponent sacrifices, damage targeting, departed-source lifelink and controller/turn restrictions |
| Unlucky Cabbage Merchant | Food, optional search, failure to find, exact source return, life gain and hidden-library privacy |
| Weaponize the Monsters | Selected creature sacrifice before responses and independent damage and exact source attribution after removal |

All 17 real-card shells and Oracle text were retrieved by exact English name
from Scryfall's named-card API on 2026-10-20 and retained in
`managym/tests/fixtures/scryfall_cards.json`. Executable behavior is typed code,
not parsed Oracle text. Existing generic keyword tests cover menace and the
other reused keyword authorities.

## Compiled and consumer evidence

The source, fixtures, generated IR and coverage live under
`content/semantic/v1/sacrifice.*`. The complete compiled/native definition
comparison includes costs, triggers, target clauses, statics and token closure.
Learning schema 4 lives at `content/semantic/w8/learning_schema.json`; schemas
2 and 3 retain their original opcode vocabularies and files. New instructions
use opcodes 51–55. Frozen opcode 24 lowers into the same search operation.

The [w8 corpus](../../conformance/semantic-kernel-w8-v1/) records ten games,
including both compiled sacrifice orientations. Its inherited Phase source
comparison does not claim external-engine parity for the new cards. Separate
w8 Academy authority/parity receipts and Learn fixture preserve the older
worlds' artifacts. Match-state hash version 5 covers the additional turn and
last-known state.

The Academy parity receipt fingerprints all Rust sources; later source edits
invalidate that receipt even if the observed Command tape stays unchanged.

Gate evidence on 2026-10-21 includes 468 passing debug Rust tests with CI's
optimization setting, Clippy with denied warnings, and affected Python checks
covering compiled admission, current-world arena admission/monitoring and
Academy parity. The opcode compatibility test now pins all 55 assignments;
assignments 1–50 remain unchanged. The full learnability diagnostic passed
three capacity seeds and reward-driven PPO (65,536 transitions, evaluation
accuracy 0.59375 before and 1.0 after, versus 0.640625 without updates).
This is diagnostic learning evidence, not full-game strength certification.

The release build and 13 browser interaction tests passed locally. The prompt
matrix's behavior/accessibility checks also passed with snapshots ignored.
Normal visual comparison failed on the macOS host against the pinned Linux
reference (952 pixels at the initial priority panel); that result remains
deferred to the Ubuntu 24.04 CI Visual Reference Gate. No reference was changed,
and the snapshot-ignored run is not visual certification. Exact commands and
local log locations are recorded in the working design for delivery.

The inherited public-commitment tape divergence and frozen RUL-9 parity
provenance mismatch remain unresolved; neither frozen input is rewritten here.
