# TLA content and qualification

Current definitions are authored once in
[`managym/src/cardsets/tla.rs`](../../../managym/src/cardsets/tla.rs).
The typed Rust definitions contain characteristics, programs and presentation text.
Set membership also owns native roots and fixed tokens. Setup manifests only choose
definition IDs and counts. The engine compiles every native/learning projection
from that source in process; there are no generated card files. Frozen v1/w11
sources remain historical evidence and are not inputs to current content. Current execution uses world **w16** and
learning schema **8**.

`SemanticPack::compile_content_pack` lowers these programs into the ordinary rules
runtime. The generic native pack also lowers its TLA definitions from that source;
there is no handwritten TLA registration or Oracle-text rules interpreter.
Fixed token templates are declared separately from deck-eligible cards. New token
references must resolve before compilation succeeds.

`managym.content_pool(pack_key)` describes definition IDs, semantic keys,
world/content/compiled identity and `eligible` or `unsupported` status with a
reason. `tla-pool-v1` is the content pool; existing authored pack keys expose their
own projections. Rust consumers use `semantic::content_pool` and `pool_content`.
Policy catalogs bind directly to these descriptors without creating a game.
Decklist Task **f871a8** owns construction policy, native `admit_deck`, input
parsing and consumer entry points. Merely appearing in an authored setup does
not qualify a card for arbitrary submitted combinations.

Qualifications bind independent witness references to a complete transitive definition-closure hash
in `qualifications.json`. Editing/removing any clause makes compilation fail until
its qualification is explicitly reviewed. This is a guard against stale evidence,
not a proof that tests or compiled definitions implement every rule. Tokens
cannot become deck entries. Unimplemented cards and preserved but unqualified
programs remain explicitly unsupported by the shared pool descriptor.
The runtime derives pack membership and definition identity from compiled IR and
eligibility from current qualifications; `inventory.json` names unimplemented cards.

The current pool qualifies **86 cards**: the five basic lands, Avatar Enthusiasts
and Glider Kids, the 55 cards documented in the [T1 clause review](t1-clause-review.json),
and these 24 cards:

- Earth Kingdom Protectors, Water Tribe Captain, Iguana Parrot, Boar-q-pine,
  Fire Sages, Treetop Freedom Fighters, Yuyan Archers, Walltop Sentries,
  Cat-Owl, Earth Kingdom Soldier, Wandering Musicians.
- Cunning Maneuver and Pillar Launch.
- Rumble Arena; Airship Engine Room, Boiling Rock Prison, Foggy Bottom Swamp,
  Kyoshi Village, Meditation Pools, Misty Palms Oasis, North Pole Gates,
  Omashu City, Serpent's Pass, Sun-Blessed Peak.

The [T1 clause review](t1-clause-review.json) covers all 60 remaining candidates
against the pinned rules and card text. Named independent witnesses cover printed clauses;
registration and compilation are not the evidence. Water Tribe Rallier now reveals
its chosen card, and temporary haste grants now permit attacks and tap-symbol
costs without erasing summoning sickness when the grant expires.

T1 still owns 91 cards; these five retain explicit prerequisites and stay unsupported:

| Cards | Required behavior | Prerequisite owner |
| --- | --- | --- |
| Accumulate Wisdom | Controller orders the library remainder, rather than randomizing it | T2 private choices |
| Suki, Kyoshi Warrior | CDA power applies outside the battlefield | T3 characteristics |
| Water Tribe Rallier | Library power filter evaluates CDAs (notably Suki); reveal is repaired | T3 authority / T2 selection |
| Cat-Gator; Lightning Strike | “Any target” includes planeswalkers and battles (CR 115.4) | T3 target characteristics |

The review retains T1 completion ownership and links exact implementation evidence
for each prerequisite. It does not launch work, add a cyclic landing dependency,
or claim the 91-card allocation is delivered. Existing exact authored setups can
still contain these cards; arbitrary pool admission remains closed.

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
beside those rows. Rocky Rebuke retains its T4 completion owner; the original
invalid-victim finding is historical following main's integrated resolution repair.
That integration alone does not qualify Rocky Rebuke's complete card behavior.

Source inventory comes from exploration Task **0d9ba3**, finalized 2026-10-28.
[Source pins](source-pins.json) retain URLs, retrieval times and byte digests.
The ten copied official/secondary raw sources and 72 repository sources were
checked against those pins and the named base revision. Scryfall is the
secondary Oracle transcription; Wizards' checklist defines inventory and the
release notes/rules supply independent authority. Historical 201-test and
parity receipts establish bounded prior behavior, not new-card qualification.

Main's MTG-177 resolution repair (`7d7736c6a`) is integrated. Allies at Last now
qualifies after four directed debug witnesses covering affinity, zero/one/two
sources, current power and lifelink, a departed source, and an illegal victim.
Historical worlds, receipts and checkpoints remain unchanged; current checkpoints
must bind the new world and content identities.

The new `tla-baseline-v1` authored proof pairs white Captain/Glider and red
Treetop/Fire Sages decks. It exercises the same programs through native Commands,
exact forks, learning projection and Academy observation serialization. It adds
no product matchup UI and leaves all existing authored deck counts unchanged.
