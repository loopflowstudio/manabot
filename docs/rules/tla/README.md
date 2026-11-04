# TLA content

TLA cards are engine `CardDefinition` functions in
[`managym/cards/src/tla.rs`](../../../managym/cards/src/tla.rs); the
fixed tokens TLA cards make are in
[`tokens.rs`](../../../managym/cards/src/tokens.rs) with every other token. Basic lands are Alpha cards (`lea.plains` …
`lea.forest` in `lea.rs`); TLA decks admit them as basics. TLA reprints live in
the set that first printed them (Corrupt Court Official in `ptk.rs`, Epic
Downfall in `eld.rs`, Lightning Strike in `ths.rs`). Setup manifests only choose
definition keys and counts. The engine runs those definitions directly and
derives learning tokens from them. Current execution uses world **w17** and
learning schema **10**.

`managym.content_pool("world")` is the world pool: every card any set file
defines, with definition IDs, keys, Scryfall ids, world/content identity and
`eligible` or `unsupported` status with a reason. `managym.set_cards(["tla",
"lea"])` narrows it to TLA cards and basics for deck generation or validation.
Rust consumers use `cards::content_pool`, `pool_content` and
`world::set_cards`. Decklist Task **f871a8** owns construction policy, native
`admit_deck`/`check_decks` (against the world, never a pack), input parsing and
consumer entry points.

Every defined non-token TLA card is supported. Cards not yet defined remain
unknown; tokens and rules fixtures are not deck entries. Jack Heart clarified
on 2026-11-03 that known rules bugs in previously playable cards belong in bug
Tasks without revoking playability. Playable does not mean bug-free.

MTG-239 implements owner-chosen library remainder order for Accumulate Wisdom,
contextual CDA power for Suki and Rallier's library filter, and shared any-target
legality/damage for Cat-Gator and Lightning Strike. Card tests in
`managym/cards/tests/cards/` and shared rules tests in `managym/tests/rules/`
witness behavior; they do not qualify cards for admission. The former exclusion
list is historical at `6d30c9013:docs/rules/tla/README.md`.

The [allocation](allocation.jsonl) retains all 286 card identities, all 294 faces,
and nine fixed tokens plus the Copy marker. Treasure is an explicit adjacent
product exclusion. All faces of a card share one completion owner. The
[delivery plan](delivery-plan.json) records the reconciled seven TLA groups plus
D1; no additional Task is launched here. Allocation is planning evidence and
never runtime data.

Five final-research T1 candidates require primitives beyond this content cut:
Foggy Swamp Hunters moves to T3 for conditional keyword grants; Pirate Peddlers
and Zhao move to T4 for `another` sacrifice-source matching; Vindictive Warden
moves to T4 for untargeted opponent damage. The precise reasons are retained
beside those rows. Rocky Rebuke retains its T4 completion owner; the original
invalid-victim finding is historical following main's integrated resolution repair.

Source inventory comes from exploration Task **0d9ba3**, finalized 2026-10-28.
[Source pins](source-pins.json) retain URLs, retrieval times and byte digests.
The ten copied official/secondary raw sources and 72 repository sources were
checked against those pins and the named base revision. Scryfall is the
secondary Oracle transcription; Wizards' checklist defines inventory and the
release notes/rules supply independent authority. Historical 201-test and
parity receipts establish bounded prior behavior.

Main's MTG-177 resolution repair (`7d7736c6a`) is integrated. Allies at Last is
witnessed by four directed debug tests covering affinity, zero/one/two
sources, current power and lifelink, a departed source, and an illegal victim.
Historical worlds, receipts and checkpoints remain unchanged; current checkpoints
must bind the new world and content identities.

The new `tla-baseline-v1` authored proof pairs white Captain/Glider and red
Treetop/Fire Sages decks. It exercises the same programs through native Commands,
exact forks, learning projection and Academy observation serialization. It adds
no product matchup UI and leaves all existing authored deck counts unchanged.

T7 defines Secret of Bloodbending through native scheduled player control. The
[full-file reconciliation](acceptance.md) links every allocated face to its
definition and behavior witness, preserves token/exclusion denominators, and
names limits of the combined evidence. Academy and training use the same
chooser-bound actions with ordinary native trivial-decision skipping.
