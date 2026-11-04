# Tarkir: Dragonstorm — implementation batches

Research date: 2026-11-04; engine revision `5dd11cec4100aa0e1f4a23c063dfd5778e2ba28f`.
Status: planning research, not implementation or rules certification.

All **276 input identities** were read in full, including **13 Omen pairs** and
all Saga chapters. [The ledger](tdm.jsonl) assigns each identity exactly once,
with card-specific obligations and dependencies. It includes Cori-Steel Cutter,
whose fetched Standard legality is banned; inventory membership is distinct from
format admission. TDC Commander and SPG cards are outside this set input.
The official [release notes](https://magic.wizards.com/en/news/feature/tarkir-dragonstorm-release-notes)
and [rules update](https://magic.wizards.com/en/news/announcements/tarkir-dragonstorm-update-bulletin)
establish that boundary and clarify Omen, behold, endure, harmonize and mobilize.
Scryfall input URLs and hashes are in [source-inventory.json](../source-inventory.json).

A literal-name scan of current set modules finds only the five basic lands.
That is a source inventory, not proof that all other cards require new engine
machinery. Many effects already compose from native verbs, selectors, conditions,
target clauses and delayed records. Existing first-printing definitions remain
the owner of reprints; TDM membership must not create duplicate definitions.

## Recommended sequence

The ledger's primary buckets are planning assignments, not new runtime enums.
Their counts sum to 276: core 24, lands 21, costs 39, behold 6, harmonize 11,
Omen 13, counters 25, combat 17, selection 53, history 29, layers 16, copy 6,
permissions 11, replacements 3, advanced 2. Dependencies cross these buckets.

**1. Finish shared choice/result and event-history contracts.** The largest
batch is selection, and many other batches depend on it. Corroding Dragonstorm,
Fangkeeper's Familiar and Kishla Village need ordered multi-card surveil; the
current `Surveil` continuation only preserves top order. Armament Dragon,
Synchronized Charge and Twin Bolt need announced allocation across targets.
Death Begets Life needs actual destruction results, not selected-object count;
Trade Route Envoy needs actual draw success; Lie in Wait needs a moved card's
contextual power. Consolidate these around scoped results and resumable traversal,
retiring overlapping search/discard/result handlers as they migrate. Do not
expand the present non-suspending `ForEachTarget` loop.

Flurry can reuse the existing second-spell trigger. Its surrounding cards need
richer shared history: graveyard departures, first counter event per creature,
cast provenance, and permanent-lifetime damage history. Karakyk Guardian's
condition is not reset each turn. Index history by the actual event and exact
object; avoid rescanning a retained game log for every static query.

**2. Generalize costs and their executable witnesses.** Monocolored hybrid
`{2/W}` and its siblings currently hit the two-color-only parser's panic branch
in [mana.rs](../../../../managym/state/src/mana.rs). Introduce typed alternatives
at the common cost owner. Maelstrom of the Spirit Dragon requires Dragon-or-Omen
spending restrictions; Herd Heirloom requires a card-type restriction, neither
expressible by the existing three-subtype bit mask. Dragonbroods' Relic adds a
second creature tap to a mana ability's cost. Sidisi binds a sacrifice's mana
value to target eligibility. These require consistent offer, payment and
execution semantics, not broader affordability guesses.

The Devotees' once-per-turn, non-tapping filters are **not** automatically
excluded: the current registration guard accepts `tap_source || once_each_turn`.
Review their actual nested-payment behavior before asserting a defect.
Preserve source reservations, exact remainders and finite-resource reasoning.

Behold shares one typed choice between a controlled qualifying permanent and a
private hand reveal, but appears at different times. Osseous Exhale chooses it
while casting; Sarkhan chooses it during resolution. Molten Exhale's timing
permission depends on electing the additional cost. Store the actual paid
choice and public reveal, not a later “controls Dragon” approximation.

**3. Unify alternate characteristics and casting permissions.** Omen is an
alternative spell characteristic set, not a transforming back face or an
alternative cost. The Dragon remains the ordinary non-stack identity; the chosen
Omen uses its own name, cost, targets and mana value on the stack. Successful
resolution shuffles it, while countering does not. Generalize the common
characteristic/cast path so later Adventure and other modal casting families
can reuse it without conflating their different exit rules.

Harmonize composes graveyard permission, alternate payment, optional creature
tap, power-derived generic reduction and a stack-departure replacement. Reuse
flashback's departure authority, but retire duplicated implementation as both
migrate. Songcrafter Mage grants this to an exact graveyard object temporarily;
do not mutate its catalog definition. Teval's delve, free-cast effects and
permission windows belong to these shared owners. Taigam additionally requires
suspend/time-counter scheduling, and copying must handle both spells and cards
outside the stack.

**4. Build counter and combat compositions.** Renew is mostly a graveyard
activation with self-exile cost, sorcery timing and ordinary effects; it should
not get a bespoke runner. Endure combines a mandatory permitted alternative
with counters or a variable-size Spirit, including zero and absent-source cases.
Keyword counters, finality and stun require their respective rules behavior,
not merely new counter labels. Host of the Hereafter captures every counter kind
from a departing incarnation.

Mobilize composes attack occurrence, token creation with attack-destination
choices, and remembered delayed sacrifice. Entering attacking does not emit a
declared-attack event. Mardu Siegebreaker uses the controller's next end step;
ordinary mobilize uses the next end step. Zurgo prevents sacrifice during a
specified step, including costs, without deleting pending watchers. All-Out
Assault extends the existing phase scheduler and next-attack watcher rather
than introducing an independent extra-combat path.

**5. Complete replacement, layer and advanced object authorities.** Elspeth
needs token-number replacement; Neriv needs damage modification conditioned on
entry history; New Way Forward needs a chosen source, consumable prevention and
the actual prevented amount. They should extend a resumable replacement pipeline
with correct chooser and re-evaluation. Dragonstorm Globe and conditional lands
need prospective entry characteristics, including entering controller.

The Siege cycle retains an as-entry choice controlling shared statics/triggers.
Clarion Conqueror adds activation prohibitions. Dragonologist's untapped-Dragon
hexproof requires tap-sensitive cache invalidation. Naga Fleshcrafter needs
temporary layer-one copies rather than permanently rewriting copied values.
Planeswalker damage support alone does not supply loyalty activation rules.
Ugin and Elspeth therefore remain explicit completion work, alongside protection
and Call the Spirit Dragons' distinct-recipient win condition.

## Acceptance and performance

Use small independently specified interactions before general games:

- Harmonize with changed creature power, colored costs, countering and non-graveyard
  stack departure; show that permission and payment use the same witness.
- An Omen countered versus resolved; a copy resolving; free casting of its
  selected face; captured Command/replay identity for each form.
- Mobilize with split attack destinations, doubled attack triggers, source
  departure and Zurgo's sacrifice prohibition. Never count token entry as attack.
- Call the Spirit Dragons choosing one multicolor Dragon several times versus
  five distinct recipients; preserve all sequential choices through forks.
- Host of the Hereafter in simultaneous deaths; New Way Forward with competing
  replacements; temporary copy expiry after source departure.

Every new fact, choice and characteristic must reach native/Python policy input,
LLM text, learning tokens and Academy, retaining privacy and capacity failures.
Token closure includes Soldiers, prowess Monks, Birds, Warriors, Zombie Druids,
Goblins, Elephants, flying Dragons, variable-size Spirits, Treasure and Reliquary
Dragon, plus dynamically copied tokens. Reuse existing token definitions where
characteristics agree; alternate faces and token programs need identity and
observation coverage. These two planeswalkers create no emblems.

Retain the TLA budget and add bounded TDM workloads for graveyard payment,
mobilize/counter interactions and multi-choice casting. Measure ordinary steps,
search total/tail, and allocations with identical work. Cost combinations and
target allocations risk combinatorial action growth; evaluate shared structured
decisions without dropping legal strategies. Cache stable definitions and
candidate indexes, not state-dependent answers behind incomplete revision keys.
Keep clone/resume exact and debug recomputation checks. No performance gain is
claimed by this source review.

Validation: all 276 Oracle IDs occur exactly once, all input face names are
retained, and primary-batch counts total 276. No runtime tests, games, builds or
benchmarks were run. Card obligations are full-text planning judgments; remaining
cross-card rules questions require implementation-level directed tests.
