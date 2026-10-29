# TLA delivery batches — reconciled 2026-10-28

Jack Heart accepted any valid user-supplied TLA deck through the shared human/bot
engine, including decklist-file input, and seven TLA PRs plus one importer PR on
2026-10-28. Implementation/publication is authorized; every merge requires his
review. At most two or three card groups run concurrently. T1 is Task 892c47;
D1 is parallel Task f871a8. Later groups remain planned, not launched here.

The final research from Task 0d9ba3 supersedes the early overlapping mechanic
sketch at `2a44b763c:scratch/tla-delivery-batches.md`. Its exact partition is
implementation planning within the accepted eight-PR boundary. Durable ownership
and dependencies live in `docs/rules/tla/allocation.jsonl` and `delivery-plan.json`.
Every one of 286 identities has one owner, including all 294 faces. Nine fixed
tokens belong to T1, the Copy marker to T6, and Treasure is explicitly excluded.

| Batch | Complete outcome | Cards | Dependencies |
| --- | --- | ---: | --- |
| T1 | One typed content source, lowering, presentation and world pool identity; straightforward cards and fixed tokens | 91 | Accepted MTG-177 resolution semantics where a witness requires them |
| T2 | Casting/payment, airbend, exact-zone grants and earthbend return, private search/foretell | 51 | T1 |
| T3 | Effective characteristics/LKI, attachments/crew, typed counters and replacement choices | 49 | T1/T2; build beside T2, integrate shared costs before landing |
| T4 | Trigger placement/history, resolution choices, stack copy and retargeting | 75 | T2/T3 |
| T5 | Copiable values and linked abilities; Joo Dee, Phoenix Fleet Airship, Ran and Shaw | 3 | T4 |
| T6 | Faces/Sagas, face-aware copies, extra turns/combat | 16 | T4/T5 |
| T7 | Secret of Bloodbending and full-file interaction/consumer/admission closure | 1 | T5/T6/D1 |
| D1 | Source-preserving file import and native construction/admission for Academy/manabot | — | Build beside T1, consume its pool interface |

Implementation corrected five research T1 assignments: Foggy Swamp Hunters to T3
(conditional keywords); Pirate Peddlers/Zhao, Vindictive Warden and Rocky Rebuke to
T4 (respectively another-source matching, untargeted damage and invalid-victim
resolution). Each reason is retained beside the allocation row. This changes the
research counts of T1 96/T3 48/T4 71 without splitting any card's clauses or faces.

T1 status is in [892c47-author-tla-cards.md](892c47-author-tla-cards.md). Jack
Heart's 2026-10-29 direction replaces qualification with support by definition:
a later batch's card is simply undefined until that batch authors it. Per-batch
card lists (`- name: obligations`, multi-face obligations joined by ` / `) are
exact projections of `allocation.jsonl` grouped by `planned_pr`; regenerate them
rather than keeping copies.

The staged integration remains T1+D1, then T2+T3, then T4→T5→T6→T7. Preparatory
work depends on settled contracts. T5 owns copy machinery; arbitrary face-aware
copy consumers finish in T6. Defects stay with their owning group or explicit
full-file integration work, within eight PRs. The named product format remains
unresolved; a supported development pool does not establish tournament legality.
