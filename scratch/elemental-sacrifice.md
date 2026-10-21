# Play Elemental sacrifice and landfall engines

Accepted implementation direction from Jack Heart, 2026-10-20, in MTG-157. The full deck selection is in elemental-decks.md and elemental-decks.json beside this file; the cube export is elemental-cube-snapshot.csv. These are actual transferred contents, not references to another checkout.

## Outcome and ownership

Humans and agents can play the selected WB go-wide sacrifice, RB sacrifice control and BG landfall/sacrifice engines faithfully. Own sacrifice-selected activation/additional costs, optional payment alternatives, sacrifice events and targeted triggers, per-turn limits, landfall, fetch lands including their complete Oracle text, and Food/Treasure tokens. Reuse the predecessor's search/hybrid/decision work rather than creating another authority. Prove payment atomicity, legal targets, triggered effects including opponents' sacrifices, hidden-library safety, nested rollback and fetch/counter/Clue interactions. Ship owned cards through the compiled and ordinary paths.

Owned new real cards:

- Bastion of Remembrance
- Callous Inspector
- Curious Farm Animals
- Cycle of Renewal
- Deadly Dispute
- Deadly Precision
- Fabled Passage
- June, Bounty Hunter
- Lotus Cobra
- Mayhem Devil
- Prismatic Vista
- Sazh's Chocobo
- Tireless Provisioner
- Tireless Tracker
- Tolls of War
- Unlucky Cabbage Merchant
- Weaponize the Monsters

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

The local elemental-162 Flow composes dependency wait/verification → pursue (implement/compress/sync/realign/iterate/publish) → ship (gate and waited landing) → Task completion. Its definition is run context. Do not leave an implementation gap for another agent simply because one stage ended. The fourth Task owns final ten-deck admission and full-matrix integration, not the earlier card Tasks.

This Task is MTG-162. It depends on verified completion of MTG-161.
