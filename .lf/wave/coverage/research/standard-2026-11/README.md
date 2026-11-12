# Standard card milestone research — 2026-11-04

Status: all 20 set reviews completed; milestone sequence proposed. Jack Heart requested full support for any Standard
deck, prioritizing post-rotation cards; increasing engine composability and
modularity; and improving or preserving both search and ordinary runtime speed.
He requested independent per-set analyses, an implementation retrospective, an
audit of current shortcuts, and comparison with other Magic engines.

The four explicitly authorized foundation Tasks are listed below. The
[proposed milestone plan](milestone.md)
maps eight additional Tasks across two milestones, with training validation and
a rough later arc. A separate [MTG-245](loopflow://manabot/task/MTG-245) now covers training/evaluation methodology for uneven card-pool growth, with its research launch authorized. Tasks 04 and 08 remain actual training runs; the near-term plan therefore contains nine new Tasks.
This research changes no runtime behavior.

Jack Heart clarified the acceptance standard on 2026-11-04: perfect correctness
is not the target, and useful training should proceed on a generally working
engine with known bugs. The consequential risk is later invalidating conclusions
believed learned. Prioritize systematic strategic distortions and evidence
integrity rather than making every obscure interaction a prerequisite. Preserve
run identities so later defects can be scoped to affected claims.

## Inventory boundary

The [official Standard page](https://magic.wizards.com/en/formats/standard)
and [October 12 rotation announcement](https://www.magic.wizards.com/en/news/announcements/banned-and-restricted-october-12-2026)
were read on 2026-11-04. The working target is tabletop Standard after the first
2027 rotation. The inventory below covers every currently listed tabletop set,
including The Big Score. Through the Omenpaths is a distinct digital-product boundary; the SPM review
records its identity mapping considerations. Do not count adjacent digital
products as additional tabletop sets.
Upcoming releases require a separately dated inventory update; the presently
released survivor pool is not the complete future format.

| Priority | Sets |
|---|---|
| Survive the next rotation | FDN, DFT, TDM, FIN, EOE, SPM, TLA, ECL, TMT, SOS, MSH, HOB, FRA |
| Rotate next; audit for reuse and later coverage | WOE, LCI, MKM, OTJ, BIG, BLB, DSK |

Scryfall set queries (`e:<set>`, `unique=cards`, paginated) provide secondary
Oracle transcription and identity joins. The downloaded research inputs were held locally in
`.cache/standard-research/<set>.json`; that cache is not published.
[source-inventory.json](source-inventory.json) retains source URLs, retrieval
dates and page digests. The published ledgers retain identity/face joins and
card-specific planning obligations. These are research inputs, not engine card
definitions, and a later refetch may differ from this dated snapshot.
Wizards rules/release notes remain the independent rules authority.

The initial fetch contains 5,336 set/card memberships, 5,177 distinct Oracle
identities. The 13 surviving sets contain 3,624 memberships and 3,526 distinct
identities. These are inventory counts, **not legal-card or implementation
coverage counts**: bans, reprints, alternate identities, and product boundaries
still require reconciliation. A card's first-printing module does not establish
its Standard legality.

## Review contract

Each set gets an independent researcher and two artifacts:

- `sets/<code>.md`: source boundary, existing support, shared mechanisms,
  ordered implementation batches, cross-set reuse, performance risks and
  focused acceptance scenarios.
- `sets/<code>.jsonl`: every fetched card identity assigned to a primary batch,
  with other mechanism dependencies and a short card-specific obligation.

The ledger must account for every input identity exactly once, include all
faces, and distinguish an implementable definition from a required engine
extension. A keyword-only classification is insufficient: inspect full rules
text, including unusual restrictions, timing, choices, identity and permissions.
Evidence limits and uncertain cards stay explicit. Set batches are candidate
views over shared engine work, not 20 independent engine implementations.

The additional reviews are [implementation history](implementation-history.md),
[current engine audit](current-engine-audit.md), and
[external engines](external-engines.md). Their findings inform the shared implementation order in the milestone plan.

The resulting [engine principles](../../../../../managym/PRINCIPLES.md) preserve the
implementation lessons and architectural boundaries.

## Implementation Tasks launched

Jack Heart authorized three audit-derived Tasks and subsequently one separate
layers Task. All four were confirmed running the code Graph's implementation
step on 2026-11-04; their primary conversations were opened in the existing
`manabot` Rex session. They publish for interactive review, with no merge or
training experiment launch authorized by this request.

| Task | Outcome | Brief |
|---|---|---|
| [MTG-241](loopflow://manabot/task/MTG-241) | Composable execution, selection, replacements and contextual reads; deliver the engine principles doc | [Execution](tasks/composable-execution.md) |
| [MTG-242](loopflow://manabot/task/MTG-242) | General mana spending restrictions and resource transitions | [Mana](tasks/mana-payment.md) |
| [MTG-243](loopflow://manabot/task/MTG-243) | Complete, scalable naming decisions through policy and human consumers | [Naming](tasks/catalog-decisions.md) |
| [MTG-244](loopflow://manabot/task/MTG-244) | Layer evaluation with Beorn/Kellan and ten additional interaction witnesses | [Layers](tasks/layers.md) |

These are shared foundations, not all of the work required for the Standard
milestone. Casting profiles/payment receipts, scoped histories, persistent
permissions and new game procedures still need their own concrete card batches.
No additional implementation Task is selected merely by a set ledger.

## Architectural and performance criteria

Card definitions and their card-specific tests necessarily grow with cards.
The shared executor should grow with distinct rules concepts and combinations,
not one custom handler for each printed card. Prefer orthogonal selectors,
predicates, values, verbs, costs, durations, permissions and continuations.
Composition must preserve timing, simultaneous events, last-known information,
target clauses, replacement choices and suspended decisions.

For each proposed shared mechanism, identify existing implementations to reuse
or consolidate, a single rules authority, real cards demonstrating reuse, and
the tests needed to prevent approximation. Recommend substantial refactors when
the current ownership is wrong; do not preserve a shortcut behind another path.
The existing native `CardDefinition` remains the executable representation.

Performance evidence must cover search replies and ordinary `Env::step`.
Preserve the existing Engine Budget, exact-result digests, same-runner paired
measurements and debug correctness checks. Extend bounded representative
workloads as new mechanics arrive. Track cumulative milestone cost as well as
individual changes; resetting a reference does not erase a regression.
Changed gameplay must be separated from equivalent-workload speedup claims.
No artificial load, exhaustive interaction explosion or unbounded game cohorts.

## Product boundary to resolve in synthesis

The current importer enforces `development-v1`, not Standard legality.
Sideboards support outside-game access, but not between-game sideboarding;
game setup does not currently offer mulligans. These are real limits on a full
tournament-play claim. Card-file completeness, format validation, and match
procedure need explicit acceptance boundaries rather than silently equating
them or treating all three as already authorized implementation.

## Publication status and set index

Research was inspected at the revisions recorded in each report. Publication
was synced to main `8e8d702f8`; Secret of Bloodbending/player control has since
landed (`4463e57f5`). Findings are dated evidence, not a live defect inventory.
The four foundation Tasks and MTG-245 continue independently; publishing these
docs neither completes nor merges them. MTG-245's selected initial approach is
a reference league of authored functional decks, frozen sampling mixtures and
stable evaluation anchors. Tasks 04 and 08 still require actual training runs.

The engine principles guide is included in this documentation publication;
MTG-241 may consume it rather than republish the embedded initial draft.

| Surviving-set review | Rotating-set review |
|---|---|
| [FDN](sets/fdn.md) · [ledger](sets/fdn.jsonl) | [WOE](sets/woe.md) · [ledger](sets/woe.jsonl) |
| [DFT](sets/dft.md) · [ledger](sets/dft.jsonl) | [LCI](sets/lci.md) · [ledger](sets/lci.jsonl) |
| [TDM](sets/tdm.md) · [ledger](sets/tdm.jsonl) | [MKM](sets/mkm.md) · [ledger](sets/mkm.jsonl) |
| [FIN](sets/fin.md) · [ledger](sets/fin.jsonl) | [OTJ](sets/otj.md) · [ledger](sets/otj.jsonl) |
| [EOE](sets/eoe.md) · [ledger](sets/eoe.jsonl) | [BIG](sets/big.md) · [ledger](sets/big.jsonl) |
| [SPM](sets/spm.md) · [ledger](sets/spm.jsonl) | [BLB](sets/blb.md) · [ledger](sets/blb.jsonl) |
| [TLA](sets/tla.md) · [ledger](sets/tla.jsonl) | [DSK](sets/dsk.md) · [ledger](sets/dsk.jsonl) |
| [ECL](sets/ecl.md) · [ledger](sets/ecl.jsonl) |  |
| [TMT](sets/tmt.md) · [ledger](sets/tmt.jsonl) |  |
| [SOS](sets/sos.md) · [ledger](sets/sos.jsonl) |  |
| [MSH](sets/msh.md) · [ledger](sets/msh.jsonl) |  |
| [HOB](sets/hob.md) · [ledger](sets/hob.jsonl) |  |
| [FRA](sets/fra.md) · [ledger](sets/fra.jsonl) |  |

Publication checks: all 20 ledgers match their fetched Oracle IDs, names and
face lists exactly (5,336 memberships; 5,177 distinct identities), with no
duplicate or missing rows. Required planning fields are present. Markdown local
links resolve; the self-contained HTML passes desktop/mobile, navigation and
print checks. No game, benchmark or training result was produced by this research.
