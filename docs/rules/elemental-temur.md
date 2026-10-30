# Elemental Temur contracts (MTG-163)

The six owned real cards execute through the native registry and the
`elemental-temur-v1` compiled pack. Three basic lands complete its nine-definition
closure. Two 40-card validation rosters exercise the new mechanics; final
admission of the ten selected decks belongs to MTG-164. Development world w9
changes rules, observations, and state hashes. Earlier evidence retains its
original identity and contents.

## Printed rules and ownership

The six Scryfall card records were retrieved on 2026-10-21 by set and collector
number and retained in `managym/tests/fixtures/scryfall_cards.json`: TLA 181,
226, 227, 150 and 80, and KLD 229. Registrations use typed engine operations;
Oracle text is independently checked by the existing conformance test.
[Wizards' release notes](https://magic.wizards.com/en/news/feature/avatar-the-last-airbender-release-notes)
provide the flashback, waterbend, Guide and Ran and Shaw rules clarifications.

| Card | Runtime contract |
| --- | --- |
| Great Divide Guide | Every land and Ally its controller controls gains a chosen-color tap ability, including Guide itself. Creature tap costs obey summoning sickness; the grant disappears with its source. |
| Hermitic Herbalist | Separate one-mana and two-mana activations; the latter offers every unordered pair of colors and retains the Lesson-casting restriction in the pool. |
| Iroh, Grand Lotus | Firebending 2. During its controller's turn, non-Lesson instants and sorceries in that graveyard can use printed-cost flashback, and Lessons can use {1} flashback. Ordinary spell timing, targets, additional costs and reductions still apply. |
| Prophetic Prism | Draws on entry. Its immediate mana ability pays {1} and taps before offering one chosen color; it cannot fund itself. |
| Ran and Shaw | Flying and firebending 2. Entry requires having been cast and three distinct Dragon and/or Lesson cards in its controller's graveyard, rechecked at trigger resolution. The token copies copiable characteristics without legendary status and was not cast. Its activated Dragon buff lasts until end of turn. |
| Waterbending Lesson | Draws three before offering resolution-time waterbend {2} or discarding. Waterbend may tap summoning-sick creatures and artifacts. Once payment starts, it must finish; partial payment cannot be followed by declining. |

`Mana` owns ordinary and Lesson-restricted color counts. Only the casting cost
of a Lesson permits restricted mana; that spell's later demanded payments do
not. The shared feasibility search pays activation costs before adding output,
counts each physical source once, and groups identical sources to avoid land
permutations. It accounts for immediate triggered mana through the same bonus
authority used by actual taps. Offers exclude mana activations whose outputs
cannot complete an already committed cost.

`SpellOnStack` retains casting origin and whether flashback was used. A spell
stays on the stack through finalization or countering until the shared zone
movement applies exile replacement. Losing Iroh after casting cannot remove
that requirement. Battlefield incarnations record whether they entered as a
cast spell; tokens and other entries do not inherit that fact.

Copied tokens retain the source definition's semantic identity and detach its
copiable characteristics. Counters, temporary buffs, tap state and cast status
are not copied. Canonical state hash version 6 includes token characteristics.
Native observations and Python bindings expose actual token/legendary flags
and ordinary/restricted mana colors; their training encoders agree on player
width 40 and card width 41. Learning schema 5 adds typed operations without
changing the retained w8 schema.

## Evidence and limits

`managym/tests/cards/` (one file per set) exercises restricted spending, chosen colors,
Guide grants and removal, Prism's upfront payment, flashback resolution,
countering and other stack departures, Iroh's turn and ordinary spell timing,
resolution-time waterbend, Ran and Shaw's cast requirement and graveyard
recheck, source removal, nonlegendary copy characteristics and Dragon buffs.
The shared choice contract checks stale Command rejection, deterministic
execution, clone isolation and nested exact rollback at the new decisions.

The compiled/native closure comparison includes all definition fields. The
source, fixture opcode traces, generated IR and coverage are under
`content/semantic/v1/temur.*`. Python admission and terminal play use the normal
training environment, semantic catalog and bounded observation encoder.
The w9 conformance corpus includes both Temur seat orders alongside the retained
Allies/Lessons, Elemental combat and sacrifice orientations. Its two executors
share rules primitives; this is not independent-engine parity evidence.
The inherited Phase overlap matrix is pinned source inspection, not runtime
differential evidence. Its local outcomes include historical simplifications
that predate w9 (such as Learn without a Lesson pool); its mismatch labels do
not establish current w9 defects or independent agreement.

Academy's current authority/parity receipts and Learn prefix have new w9 paths.
Earlier receipts, frozen `erd1` Study evidence, and the `ed2` pre-command address
contract retain their meanings. Release browser checks and the pinned Linux
visual reference remain delivery gates; these engine checks do not establish
visual certification, deck balance, or playing strength.
The retained navigation corpus now has a w9 fixture, verified against native
offers, projections, and state digests. Checkpoint-advice, belief-recommendation
and public-commitment evidence remains unreconciled with w9. Current Learn
browser tests do not close those broader Academy contracts.

Current coverage evidence references `waterbend_taps_do_not_trigger_badgermole_cub`.
The generated coverage artifact follows that corrected rule; historical
conformance corpora retain their original bytes. Batched semantic-policy tests
allow four float32 epsilon of absolute matrix-kernel roundoff near zero, retain
the relative error bound, and require the same decoded Command.
