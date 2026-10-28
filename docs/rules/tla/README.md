# TLA content and qualification

Current definitions are authored once in
[`content/semantic/current/definitions.json`](../../../content/semantic/current/definitions.json).
They contain typed characteristics, programs and presentation text. Setup manifests
only choose definition IDs and counts. Generate every native/learning projection
with `uv run python -m scripts.compile_current_content`; `--check` rejects stale
outputs. Frozen v1/w11 sources remain historical evidence and are not generator
inputs. Current execution uses world **w16** and learning schema **8**.

`SemanticPack::compile_content_pack` lowers these programs into the ordinary rules
runtime. The generic native pack also lowers its TLA definitions from that source;
there is no handwritten TLA registration or Oracle-text rules interpreter.
Fixed token templates are declared separately from deck-eligible cards. New token
references must resolve before compilation succeeds.

`managym.content_pool(pack_key)` describes definition IDs, semantic keys,
world/content/compiled identity and `eligible` or `unsupported` status with a
reason. `tla-pool-v1` is the content pool; existing authored pack keys expose their
own projections. Rust consumers use `semantic::content_pool` and `pool_content`.
Decklist Task **f871a8** owns construction policy, native `admit_deck`, input
parsing and consumer entry points. Merely appearing in an authored setup does
not qualify a card for arbitrary submitted combinations.

Qualifications bind independent witness references to a complete transitive definition-closure hash
in `qualifications.json`. Editing/removing any clause makes generation fail until
its qualification is explicitly reviewed. This is a guard against stale evidence,
not a proof that tests or generated definitions implement every rule. Tokens
cannot become deck entries. Unimplemented cards and preserved but unqualified
programs remain explicitly unsupported by the shared pool descriptor.

This cut qualifies **31 cards**: the five basic lands, migrated Avatar Enthusiasts
and Glider Kids, and these 24 additions:

- Earth Kingdom Protectors, Water Tribe Captain, Iguana Parrot, Boar-q-pine,
  Fire Sages, Treetop Freedom Fighters, Yuyan Archers, Walltop Sentries,
  Cat-Owl, Earth Kingdom Soldier, Wandering Musicians.
- Cunning Maneuver and Pillar Launch.
- Rumble Arena; Airship Engine Room, Boiling Rock Prison, Foggy Bottom Swamp,
  Kyoshi Village, Meditation Pools, Misty Palms Oasis, North Pole Gates,
  Omashu City, Serpent's Pass, Sun-Blessed Peak.

The [allocation](allocation.jsonl) retains all 286 card identities, all 294 faces,
and nine fixed tokens plus the Copy marker. Treasure is an explicit adjacent
product exclusion. All faces of a card share one completion owner. The
[delivery plan](delivery-plan.json) records the reconciled seven TLA groups plus
D1; no additional Task is launched here. Allocation is planning evidence and
never the runtime eligibility database.

Five final-research T1 candidates require primitives beyond this content cut:
Foggy Swamp Hunters moves to T3 for conditional keyword grants; Pirate Peddlers
and Zhao move to T4 for `another` sacrifice-source matching; Vindictive Warden
moves to T4 for untargeted opponent damage. The precise reasons are retained
beside those rows. Rocky Rebuke also moves to T4: its power-damage operation
ignores an invalid victim when the source remains legal, pending MTG-177
integration. Their other clauses do not justify partial admission.

Source inventory comes from exploration Task **0d9ba3**, finalized 2026-10-28.
[Source pins](source-pins.json) retain URLs, retrieval times and byte digests.
The ten copied official/secondary raw sources and 72 repository sources were
checked against those pins and the named base revision. Scryfall is the
secondary Oracle transcription; Wizards' checklist defines inventory and the
release notes/rules supply independent authority. Historical 201-test and
parity receipts establish bounded prior behavior, not new-card qualification.

MTG-177's pending w15 resolution repair is not included or presumed accepted.
Existing program availability is preserved without certifying full fidelity on
that unresolved dependency. Historical worlds, receipts and checkpoints remain
unchanged; current checkpoints must bind the new world and content identities.

The new `tla-baseline-v1` authored proof pairs white Captain/Glider and red
Treetop/Fire Sages decks. It exercises the same programs through native Commands,
exact forks, learning projection and Academy observation serialization. It adds
no product matchup UI and leaves all existing authored deck counts unchanged.
