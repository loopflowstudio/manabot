# Spider-Man: casting provenance, permissions and contextual identity

Research: 2026-11-04, checkout `5dd11cec4`. All **193 identities** and full
rules text were inspected, including five modal double-faced pairs and five
Sagas. The [card ledger](spm.jsonl) assigns every input identity exactly once;
its ID/name/face coverage was checked against the downloaded input. This is
reviewed planning, not implemented behavior or a conformance certificate. No
games, tests or benchmarks were run.

## Inventory boundary

The Scryfall `e:spm`, `unique=cards` input is pinned in the parent
[source inventory](../source-inventory.json). SPM survives rotation. SPE and MAR
are distinct products and do not gain Standard legality merely by sharing
packaging. Existing exact-name source inspection finds only the five basics,
owned by `lea.rs`; 188 remaining identities need definitions, with varying
amounts of engine work.

The old digital-substitute story is now incomplete. Wizards' May 2026
[Arena announcement](https://magic.wizards.com/en/news/mtg-arena/marvel-super-heroes)
says SPM became available June 16, with Through the Omenpaths equivalents
converted to card styles and existing decks retained. An import policy may need
explicit historical-name mappings, but these are identity/presentation concerns,
not justification for duplicate card programs. Do not infer tabletop
interchangeability or copy-limit rules solely from Arena's collection migration.
The current tabletop inventory remains SPM; OMB/MAR legality is separate.

## Shared batches

| Primary batch | Cards | Representative work |
|---|---:|---|
| Existing composition | 36 | Ordinary damage, keywords, triggers, tokens and basics |
| Choice/result continuations | 39 | Connive, multizone selections, reflexive triggers, ordering and paired-player results |
| Permissions and history | 32 | Mayhem, cast origin, turn counts, persistent exile/top-library permission |
| Costs and mana | 30 | Web-slinging, return/tap/discard/counter costs, target-dependent pricing, restricted production |
| Characteristics and grants | 23 | Modified predicates, layers, Equipment, persistent ability grants and contextual CDAs |
| Replacements and copying | 10 | As-entry choices, copy exceptions, prevention/redirection and life exchange |
| Faces and Sagas | 10 | Five modal DFC pairs and five chapter programs |
| Linked object lifetimes | 6 | Delayed returns, linked exile, departed-source results and exact identities |
| Stack operations | 3 | Spell/ability copies, retargeting, copy exceptions and per-turn use |
| Designations | 2 | Harnessed and suspected, with different semantics |
| Combat constraints | 2 | Contributor selection and combined blocker restrictions |

These are primary planning owners, not isolated implementations. Start with a
small core slice alongside generalized costs, retained results and permissions.
Then exercise replacements/layers with real cards before completing face and
copy combinations. Saga, crew and ordinary keyword names do not prove their
new combinations already work.

## Highest-leverage shared work

**Casting is a composition of permission, face, timing and costs.** Mayhem
requires this-turn discard provenance, permits normal-timing graveyard casting,
and does not impose flashback's exile-on-exit replacement. Oscorp Industries
extends the same principle to land plays and their ordinary limit. Green Goblin
grants mayhem and separately reduces costs; Sandman's Quicksand checks how its
cost was paid. Keep these dimensions separate in one cast proposal and retained
provenance record. Replace overlapping flashback/impulse special cases as their
consumers migrate, without changing existing semantics.

Web-slinging combines an alternative mana cost with a selected tapped-creature
return. Scarlet Spider, Ben Reilly needs the returned creature's captured mana
value at entry. Urban Retreat uses that same return operation as a hand
activation cost, while Pumpkin Bombardment chooses discard or extra mana.
Supportive Parents taps two creatures for mana; Iron Spider removes counters
across artifacts. Generalize the current
[`AdditionalCost` and activation fields](../../../../managym/state/src/card.rs)
into shared typed obligations and result bindings rather than new per-card
booleans. Convoke shares contribution selection but keeps its own colored/generic
payment semantics. Interdimensional Web Watch requires source-zone spending
predicates beyond the current three-subtype `ManaUse` vocabulary.

**Permissions need precise lifetime and visibility.** Black Cat inspects an
opponent's library, retains private face-down exiles, and permits flexible mana
spending. Gwen Stacy's permission lasts while the exact source stays controlled;
Superior Foes' lasts until its next successful exile; Madame Web continually
exposes only the top card; Gwenom substitutes a life cost. Existing
[`PlayWindow`](../../../../managym/state/src/ability.rs) has only turn-boundary
durations. A shared permission record should own exact subjects, controller,
zone, expiry, timing and payment override. Readers in offers, execution, search,
observations and text must agree. Do not duplicate this in Academy.

**Make selection and results truly composable.** Connive must retain the actual
discard and recipient; a departed recipient can still cause draw/discard but
cannot receive a counter. Parker Luck binds both players' revealed cards before
cross-reading their values. Behold the Sinister Six selects distinct names;
Terrific Team-Up has multiple damage sources and a separate victim; Electro,
Shriek and Rhino's Rampage require separately targetable reflexive occurrences.
Extend the shared continuation/result authority, retiring narrow discard/search
handlers where applicable. The Clone Saga and Risky Research also require full
multi-card surveil ordering, not the current count-one shortcut.

## Face, layer and replacement traps

SPM's five pairs are **modal DFCs that can also transform**. Either face can be
cast; stack/battlefield mana value follows the relevant face. TLA's transforming
front-face mana-value convention is therefore insufficient. Add explicit face
kind and contextual face queries; reuse transform identity only where the rules
agree. Preserve TLA meanings and frozen input contracts. This distinction is
explicit in the [official release notes](https://magic.wizards.com/en/news/feature/spider-man-release-notes).

Superior Spider-Man combines as-entry copying with fixed name/base-stat/type
exceptions and a reflexive exile of the chosen graveyard card. Chameleon changes
only its name. Spider-Punk grants riot choices and prohibits prevention and
countering; Anti-Venom converts prevented damage into counters; With Great Power
redirects damage. These are strong cross-set tests for a replacement pipeline
that allows the correct chooser and re-evaluates applicability, rather than
adding preferred-order exceptions.

Modified requires all counter kinds, any attached Equipment, but only Auras
controlled by the creature's controller. Departure triggers need that predicate
in LKI. Hydro-Man replaces card types temporarily; Lizard replaces color,
subtypes, abilities and base stats indefinitely. Symbiote Spider-Man grants
abilities whose source becomes the recipient, excluding the granting activation.
One characteristics authority should own these questions.

Harnessed is a noncopiable permanent designation enabling an ability; copying
the Stone does not copy that designation. Suspected instead implies menace and
a blocking restriction. Share designation storage/observation patterns where
appropriate, not their rule meanings.

## Acceptance, closure and performance

Use bounded native Command scenarios, then fork/replay and policy/text/Academy
projection checks:

- Return a modified-cost creature for web-slinging, remove relevant sources,
  and verify locked cost plus captured entry value.
- Discard/recast a mayhem card, move it between zones, and test ordinary timing,
  land limits and absence of flashback exile semantics.
- Cast either MDFC face under a mana-value restriction, transform, then change
  zones; compare contextual mana value and legal naming of either face.
- Combine riot, prevention prohibition and damage conversion; clone while an
  affected player's replacement choice waits.
- Connive after recipient departure; order retained surveil cards; counter a
  reflexive trigger after its payment or fight has committed.
- Remove The Spot before its exile resolves, or prevent its later graveyard
  move; linked returns must follow actual movement and exact incarnations.

Created-object closure includes Food, Treasure, green-white Human Citizens,
green reach Spiders, blue Illusion Villains, flying artifact Robots, and arbitrary
spell/permanent copies with exceptions. Persistent grants and copied triggered
abilities must survive source departure correctly.

Profile shared payment search, named-card candidate expansion, top-library
permissions, layer invalidation and copied-effect allocations. Keep catalog
names shared and support the complete naming choice despite fixed policy
capacity. Compare both ordinary steps and fixed-budget search on one runner;
preserve exact digests when semantics are unchanged. This review establishes no
speedup and no universal capacity guarantee.
