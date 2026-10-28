# TLA support map — 2026-10-28

Draft evidence at `c5d406e3d2a701525c15c8500a30f7da7896cc40`, runtime w14/hash9.
[Implementation plan](full-tla.md) · [exhaustive face/token ledger](tla-card-ledger.jsonl)
· [source pins](tla-evidence/sources.json) · [check receipt](tla-evidence/validation.json).

## Denominators and dispositions

| Inventory | Count | Meaning |
| --- | ---: | --- |
| Distinct TLA cards | 286 | 281 nonbasics plus five basics; every card included |
| Card faces | 294 | Eight extra reverse faces; all matched to official checklist |
| TLA printings / printing faces | 394 / 414 | Art variants collapse to gameplay identities |
| Native registered cards | 76 | 71 nonbasics plus basics; **partial**, no full-card certification |
| Any embedded compiled catalog | 74 | Explicit authored setups only; presence is not rules evidence |
| Current Elemental compiled pack | 69 | Its exact ten setups, not arbitrary full-set admission |
| Missing native cards / faces | 210 / 218 | No executable registration found; not inferred from audit unknowns |
| Required fixed token templates | 9 | Ally, Spirit, Clue, Food, Dragon, Monk, Soldier, Bear, Ballistic Boulder |
| Required dynamic marker | 1 | Copy: characteristics derive from the copied object and effect |
| Adjacent token retained/excluded | 1 | Treasure; TTLA sheet includes it but TLA producers do not require it |

No card is labelled fully verified. **Verified supported subsets** are the assertions
in 201 passing directed rules tests; some tests use non-TLA setup cards, so this is
not 201 TLA behavior obligations. The 100-definition parity test checks agreement,
not an independent oracle. Per-face correctness beyond those assertions remains
unreviewed even when a row's overall disposition is partial.

Native cards outside current Elemental: Avatar Enthusiasts, Glider Kids, Kyoshi
Warriors, First-Time Flyer, Tiger-Seal, Earth King's Lieutenant, Suki, Kyoshi Warrior.
Only the first two lack any compiled catalog definition. Existing source text says
“complete registered behavior”; that header is not accepted as evidence.

## Shared prerequisite map

Paths below are under `managym/src/`; detailed rule IDs, per-face obligations and
owner links are in `tla-evidence/mechanics.json` and the ledger. Row families overlap.

| Family | Current proven/narrow path | Full-file gap and concrete consumer |
| --- | --- | --- |
| Content authority | Typed IR lowers to native definitions; exact setup binding | Two live authoring paths and registry-only presentation lookup. Delete duplication before adding 210 missing cards. `semantic/content.rs`, `cardsets/` |
| Targets/resolution | Shared target predicate, target clauses, partial legality | MTG-177's retained eligibility remains separate; abilities/retargets need full stack identity. Redirect Lightning, Sokka's Haiku. `flow/resolution.rs`, `flow/action.rs` |
| Exile/object identity | ObjectRef, Jailer links, narrow flashback and earthbend | Airbend permission absent; earthbend delayed return retains wrong kind of identity. Airbender's Reversal, Avatar's Wrath. `flow/zones.rs`, `flow/play.rs` |
| Costs/mana | Shared PaymentModel, fixed waterbend, kicker, restrictions | X and cost scopes, broader restrictions, exhaust, crew, free casts. North Pole Patrol, Crashing Wave, White Lotus Hideout. `flow/mana.rs`, `state/mana.rs` |
| Trigger events/order | Controller-selected APNAP order, ordinary events | True modal/reflexive placement, trigger-on-trigger/multiplicity, bending history. Appa, Earth Rumble, Katara, Ascension. `flow/trigger.rs`, `flow/triggers.rs` |
| Effective characteristics | Additive P/T, counters, selected grants/CDA | Layer dependencies, control/name/type/ability overwrite, complete LKI. Honest Work, Day of Black Sun, Toph. `flow/statics.rs` |
| Replacements/counters | Four narrow replacement forms plus special paths | Affected-player iterative choice; stun/finality/lore/quest/keyword counters. Fated Firepower, Wolfbat, Rowdy Snowballers. `flow/proposed_event.rs`, `flow/sba.rs` |
| Attachments/face lifecycle | No general representation found | Aura legality, Equipment, Saga final chapter, transform versus new object. Honest Work, Twin Blades, all eight DFCs. `state/card.rs`, `state/permanent.rs`, `flow/turn.rs` |
| Copies/ability inheritance | Ran and Shaw nonlegendary self-copy | Arbitrary copies, exceptions, copied stack objects, inherited activated/triggered programs. Ember Island Production, Koh, Fire Lord Azula. `flow/resolution.rs`, `flow/zones.rs` |
| Turns/decision control | Two-player normal turn and decision flow | Extra phases/turns; controller versus decision actor; authorized private view. Bumi, Avatar Kuruk, Secret of Bloodbending. `flow/turn.rs`, `experience.rs` |
| Consumer capacity | Complete compound offers; legacy overflow is explicit | All new choice shapes and ragged object state must reach training/Academy without caps. Appa's arbitrary target set; repeated token production. `agent/structured_offer.rs`, `agent/observation_encoder.rs` |

## How to read or reproduce the ledger

One `card_face` row per official face, with all printing IDs/numbers; token rows
identify producer closure and explicitly retain Treasure outside it. `obligations`
is a manually written summary of each face, not generated rules behavior. The
source card text is retained separately. `prerequisite_families` is pattern-assisted
triage over inspected text/notes; it is not exhaustive adjudication of subrules.
`rule_refs` are verified family entry points in the pinned CR. No claim that every
listed subrule applies to every face is intended.

`native_registration` is a source location, `compiled_admission` names each authored
pack and direct main/sideboard bindings, and `behavioral_evidence` points to named
existing tests with actual execution flags. Token reachability is through producers,
not direct deck entries. Test fixture mentions can be setup-only; read assertions
before using them as a clause's certificate. A missing card needs all listed
obligations, not just its keyword. A registered card needs unresolved core and
cross-mechanic obligations checked before a full-file support upgrade.

Offline check: `uv run --no-project --python 3.12 python scratch/tla-evidence/verify_ledger.py`.
The generator is `build_ledger.py`; `card-obligations.tsv` and `mechanics.json` are
its explicit human-authored inputs. Rebuilding is deterministic at the pinned
checkout. Raw sources, extraction products and test logs are evidence, not future
implementation instructions. Do not equate inventory completeness, source parity,
consumer agreement or family membership with whole-rules conformance.
