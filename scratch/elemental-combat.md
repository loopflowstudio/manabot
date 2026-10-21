# Play the Elemental combat and Learn card pool

Accepted implementation direction from Jack Heart, 2026-10-20, in MTG-157. The full deck selection is in elemental-decks.md and elemental-decks.json beside this file; the cube export is elemental-cube-snapshot.csv. These are actual transferred contents, not references to another checkout.

## Outcome and ownership

Humans and agents can faithfully play the selected combat, Ally, draw-two and Learn cards, using the existing engine and compiled semantic path. Own hybrid mana, shared basic-land library search needed by Field Trip and Shared Roots, prowess/noncreature-cast triggers, targeted modal spells, ordinary graveyard returns/mill, and conditional stats/keywords needed by these cards. Preserve full existing Oracle behavior on reused registrations. Introduce reusable, typed rules operations rather than card-name exceptions. Ship the owned real cards and directed play tests; do not stop at a substrate-only refactor.

Owned new real cards:

- Abandon Attachments
- Azula Always Lies
- Boomerang Basics
- Cat-Gator
- Corrupt Court Official
- Dai Li Agents
- Deserter's Disciple
- Enthusiastic Study
- Epic Downfall
- Eyetwitch
- Field Trip
- Fists of Flame
- Foggy Swamp Spirit Keeper
- Gather the White Lotus
- Giant Cindermaw
- Gnarled Professor
- Haru, Hidden Talent
- Iroh's Demonstration
- Jet, Freedom Fighter
- Knowledge Seeker
- Lightning Strike
- Master Piandao
- Messenger Hawk
- Momo, Friendly Flier
- Monastery Swiftspear
- Octopus Form
- Origin of Metalbending
- Ostrich-Horse
- Ozai's Cruelty
- Power Word Kill
- Pretending Poxbearers
- Professor of Symbology
- Savor
- Shared Roots
- Sokka's Haiku
- Spectral Sailor
- Tiger-Dillo
- Toph, the Blind Bandit
- True Ancestry
- Uncle Iroh
- Wartime Protestors
- Zuko's Exile

## Accepted constraints and delivery

- Implement the printed Oracle rules from verified source text. Simple rarity does not waive abilities. No silent proxies, rules reductions, stale compatible-looking identities, or another consumer-owned rules engine.
- Cube names only. Quantity caps apply independently to each deck INCLUDING its outside-game Lesson pool. Reuse between decks is allowed; basic lands are unlimited. Ten strategic identities remain distinct. UG is Temur with Iroh, Grand Lotus and Ran and Shaw; BG is landfall plus sacrifice.
- This explicitly authorizes the selected content expansion beyond the older Allies/Lessons chapter boundary. Preserve that existing world and frozen evidence. This does not authorize a general expansion or writes to Wave memory/chapter files.
- Jack Heart said: “make 3-4 autonomous tasks to build these cards ... i dont need to review PRs at all. launch them all, you can stage them however you like.” Implement, inspect/review, run required checks, publish and land autonomously; no demo, design, PR-review or merge-click approval is required. Ask in the existing MTG-157 conversation only if genuinely consequential new scope cannot be resolved; do not invent a new human-review gate.
- Work only in this Task's checkout. Read AGENTS.md and package guides. Python commands use uv. Debug cargo tests are mandatory for Rust changes; rebuild the extension from the root using the guide. Respect configured authored-world/release-browser and pinned Linux checks. Do not relax required checks or overwrite historical receipts to achieve green.
- Each card needs directed meaningful rule evidence, source/Oracle conformance and compiled/native coherence. Exercise privacy, stale rejection, replay and fork/rollback for new decisions and state. Broaden tests when integration evidence warrants it.
- Deliver after preserving a durable coverage/contract note outside scratch. Use Loopflow publication and waited landing, then verify the merge and complete the Task. No Co-Authored-by commit trailers. No paid training or cloud rentals.

## Execution staging

The launch is explicitly serialized to avoid concurrent edits to shared ability, payment, compiler, observation and fixture owners. The authored Flow waits on its predecessor, then an XOR routing step requires authoritative Task completion before selecting the build branch and syncing merged main. A blocked dependency stops this Flow; it never treats a finished/failed Flow as delivered. Do not bypass that check. Resolve normal sync conflicts and CI failures autonomously; preserve other contributions.

The local elemental-161 Flow composes dependency wait/verification → pursue (implement/compress/sync/realign/iterate/publish) → ship (gate and waited landing) → Task completion. Its definition is run context. Do not leave an implementation gap for another agent simply because one stage ended. The fourth Task owns final ten-deck admission and full-matrix integration, not the earlier card Tasks.

This Task is MTG-161. It is the first implementation stage and has no implementation predecessor.
