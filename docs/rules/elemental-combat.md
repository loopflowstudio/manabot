# Elemental combat and Learn contracts (MTG-161)

The native registry contains all 42 owned cards selected on 2026-10-20.
The separate `elemental-combat-v1` semantic pack admits the complete pool in
development world w7. World freeze and release certification remain pending. Oracle shell conformance covers every real registration;
printed text is presentation data and never supplies executable behavior.

## Native behavioral coverage

Directed cases live in `managym/tests/rules/elemental_cards.rs`. The following
mapping identifies the behaviors asserted rather than claiming exhaustive
Magic rules coverage.

| Owned cards | Directed behavior |
| --- | --- |
| Abandon Attachments | Optional discard, exact draw count and decline |
| Azula Always Lies | All three mode combinations before payment; independent targets, including the same creature for both clauses |
| Boomerang Basics | Nonland bounce; draw only for a controlled target |
| Cat-Gator | Swamp-dependent entry damage and lifelink |
| Corrupt Court Official; Ozai's Cruelty | Target player chooses the printed number of discards |
| Dai Li Agents | Two earthbend clauses can choose the same land; attack drains by creatures with counters |
| Deserter's Disciple | Another controlled small creature; tap payment; independent ability survives source departure |
| Enthusiastic Study; Fists of Flame; Octopus Form | Targeted stats/keywords, draw count and Learn continuation |
| Epic Downfall; Power Word Kill | Mana-value and excluded-subtype legality |
| Eyetwitch; Pretending Poxbearers; Knowledge Seeker | Distinct death rewards; second-draw counter |
| Field Trip; Shared Roots | Basic/subtype legality, tapped entry, shuffle even on failure, Field Trip's Learn continuation |
| Foggy Swamp Spirit Keeper | Second-draw Spirit creation; Spirit combat restriction |
| Gather the White Lotus | Token count by Plains; arbitrary scry order and privacy |
| Giant Cindermaw; Savor | Trample/life prevention; Food, debuff and tap/sacrifice costs |
| Gnarled Professor; Professor of Symbology | Entry Learn |
| Haru, Hidden Talent; Wartime Protestors | Another Ally entry, earthbend and exact triggering Ally counters/haste |
| Iroh's Demonstration | Both modes, unavailable targeted mode, opponent-only mass damage |
| Jet, Freedom Fighter | Creature-count entry damage; up to two distinct death targets |
| Lightning Strike | Printed damage through target choice and resolution, including source-removal scenarios |
| Master Piandao | Attack look/select, public selected-card knowledge and random remainder |
| Messenger Hawk | Either hybrid color, entry Clue and second-draw payoff |
| Momo, Friendly Flier | First matching flying spell reduction, later matching spell unreduced, another flier entry buff |
| Monastery Swiftspear | Prowess resolves above the noncreature spell |
| Origin of Metalbending | Artifact destruction; counter and indestructible prevent destruction/lethal damage |
| Ostrich-Horse | Mill three; only milled lands selectable; counter on decline |
| Sokka's Haiku | Separate spell/land bindings; partial illegality preserves draw/mill and untap |
| Spectral Sailor | Draw activation survives source departure |
| Tiger-Dillo | Attack eligibility requires another four-power creature |
| Toph, the Blind Bandit | Entry earthbend and counter-dependent power |
| True Ancestry | Optional permanent-card graveyard target, Clue; exact incarnation rejection |
| Uncle Iroh | Lesson-only cost reduction |
| Zuko's Exile | Exile and Clue for the target's controller |

## Shared authority and information contracts

Actions and resolution use the same `game_object::Target` type; the older
`state::target::Target` import path is a re-export. Exact binding in
`BoundTarget` remains the Command boundary's responsibility.

Modes of a spell are chosen before targets and payment. Modes inside an ability
remain resolution-time choices. `TargetClause` declares independent clauses;
effects inside one clause share its targets. Distinctness applies within each
clause, so two clauses may legally choose the same object. Stack and suspended
frames carry the clause bindings and chosen spell mode. Graveyard targets carry
an exact `ObjectRef` and cannot follow a card through a zone change.

Scry records inspected cards and the partial top/bottom order. It commits one
library reorder after all selections. Every remaining card can be selected
next for either destination; this represents every legal final ordering.
Inspected library cards are visible to the chooser and pinned during their
possible-world materialization. Search derives candidates from the live sampled
library instead of pinning those candidates as known library positions.
Opponent-commitment materialization refreshes search offers from that library.

Directed shared checks cover ordinary Commands, deterministic replay, stale
command rejection without mutation, nested exact-fork rollback, private native
observations, scry materialization, and search materialization. Public graveyard
returns use definition-count hand knowledge rather than revealing physical hand
copies. Counter mutations journal their permanent at the event authority.

The bounded encoders use card width 40 and permanent width 27. Indestructible
is card column 38 and permanent column 25; validity follows each. The outside
marker stays at card column 37 and the permanent definition join stays at 24.
The Python/native parity test names and asserts these columns.

## Compiled admission and consumers

`content/semantic/v1/elemental.source.json` describes all 42 owned cards, five
basics and four tokens. Its generated IR executes through the shared typed
lowerer; only presentation text comes from the native registry. The Elemental
pack is a distinct catalog entry. Historical two-deck and Jeong artifacts are
unchanged. Both 46-card validation rosters select it in either seat order; they
are test setups, not final ten-deck product admission.

The compiler adds reusable operations for library search, counted discard,
dynamic quantities, modal programs, independent target clauses, graveyard
returns/mill and conditional statics. Existing opcode numbers 1–23 remain
stable. Role declarations and actual target binding are separate: inferred
binding uses the selected mode and shared effect traversal. The diagnostic
trace retains new rule steps symbolically; the live lowerer and game engine
execute their decisions and quantities.

`compiled_elemental_closure_matches_all_native_rules` compares the entire rules
structure of every definition (including token closure) with the native
registry. Together with the directed cases above and complete compiled games,
this establishes native/compiled coherence without copying native rules into
the runtime lowerer. Control-change cases cover resolution-time target
controller rewards and the captured controller of an announced trigger.

Learning schema 3 is separate at `content/semantic/w7/learning_schema.json`.
It includes typed hybrid pips, quantities, modes and target clauses. Runtime
binding verifies the exact compiled source/IR fingerprints. Complete catalog
padding succeeds without truncation. The consumer test sends a real compiled
search Command through acting-view belief tracking and a history-enabled
policy, checks nonzero history gradients, and preserves public history under
determinization. It does not claim inference quality or new strength.

[The w7 corpus](../../conformance/semantic-kernel-w7-v1/) contains eight terminal
games and 1,107 reproducible commands, four on the compiled Elemental pack.
CI and the current conformance test select this root. Its pinned Phase matrix
retains the inherited 15-row source comparison; it makes no external-engine
comparison claim for the new cards. Frozen `erd1` receipts and older corpora
retain their identities.

## Gate evidence and remaining limits — 2026-10-20

The debug Rust suite passes all 446 tests, including the 42 Elemental cases.
Clippy passes with all targets/features and warnings denied. The CPython 3.12
release extension is rebuilt and installed. The semantic suite passes 138
tests, including exact historical kata artifact reproduction. That static
rebuild tolerates only the additive empty-sideboard setup fields and verifies
the complete original IR and token schema; historical experiment runners
retain their preregistered compiler checks.

The new [w7 authority receipt](../../conformance/authored-match-w7-v1/) and
[w7 parity proof](../../conformance/authored-match-parity-w7-v1/) preserve the
older receipts. Live, headless and persisted replay agree on 156 Commands and
942 viewer checks. A separate w7 Learn browser fixture preserves the older
fixture and reaches Learn after 13 hero Commands. Browser coverage passes for
choice navigation, combat presentation and Learn, including reconnect and all
three Learn outcomes. The Learn test waits for the new game's revision before
submitting the first offer, preventing a race against the previous game.

World freeze remains blocked. The historical public-commitment tape diverges
at hero revision 41 (its next expected hero revision is 46). RUL-9's contract
already pins a different parity-file hash than the checked file at HEAD;
neither frozen input is rewritten. The public-commitment proof needs a new
world-specific corpus, and the historical RUL-9 provenance mismatch needs
resolution at its contract owner. The pinned Ubuntu visual comparison remains
with CI: the macOS run differed by 952 pixels on the first priority snapshot.
No visual reference was updated. These are unresolved checks, not passes.
