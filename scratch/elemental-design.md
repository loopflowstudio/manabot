# Play Temur Lessons with Iroh and Ran and Shaw

Accepted implementation direction from Jack Heart, 2026-10-20, in MTG-157. The full deck selection is in elemental-decks.md and elemental-decks.json beside this file; the cube export is elemental-cube-snapshot.csv. These are actual transferred contents, not references to another checkout.

## Outcome and ownership

Humans and agents can play Temur Lessons with the explicitly selected Iroh, Grand Lotus and Ran and Shaw, using the correct printed rules. Own restricted/granted mana and its observations, flashback and timing/cost permissions with exile on leaving the stack, and Ran and Shaw's cast-origin plus graveyard condition and nonlegendary token copy. Build on both predecessors' landed mechanics. Preserve trigger subjects, token characteristics and mana-spending legality through Commands, agent actions, replay and forks. Ship all owned cards and adversarial interactions, not only illustrative simplified versions.

Owned new real cards:

- Great Divide Guide
- Hermitic Herbalist
- Iroh, Grand Lotus
- Prophetic Prism
- Ran and Shaw
- Waterbending Lesson

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

The local elemental-163 Flow composes dependency wait/verification → sync → pursue → ship → Task completion. Its authored files define the execution boundaries; the implementation and evidence status is maintained in elemental-temur.md. The fourth Task owns final ten-deck admission and full-matrix integration, not the earlier card Tasks.

This Task is MTG-163. It depends on verified completion of MTG-162.

## Concept review — 2026-10-21

Outcome: no change to the accepted design and no unresolved product choice.
This review preserves Jack Heart's 2026-10-20 requirements; it adds no approval
or release certification. Remaining verification belongs to the existing gate.

The interaction is ordinary play with the selected cards. A player chooses a
legal action, supplies its offered choices, and sees the resulting game state.
Humans and agents use the same authoritative offers; neither needs to choose a
native versus compiled rules implementation or manage a separate flashback mode.

| Player action | Domain concept and state-changing API |
| --- | --- |
| Activate Herbalist and choose its two colors | `ActivateAbility` then `ChooseMode`; `Mana.lesson` retains the Lesson-casting restriction, including in observations and offer labels. |
| Cast a Lesson from the graveyard while Iroh permits it | Ordinary `CastSpell` offer and payment; `SpellOnStack` captures casting origin and flashback so later source loss cannot erase the exile obligation. |
| Resolve Waterbending Lesson | Draw first, then `WaterbendTap`/`PayCost` or `Decline` before waterbend payment starts; the suspended effect frame retains the continuation. A waterbend tap pays a cost without producing mana. |
| Resolve Ran and Shaw's qualified entry trigger | Recheck the graveyard condition, then create a nonlegendary token with detached copiable characteristics and no inherited cast status. |
| Retry after a rejected stale action | Obtain the current decision frame and submit a current `Command`; rejection must leave state unchanged. Exact search forks use rollback, rather than introducing a live-play undo promise. |

Normal path: Herbalist's restricted mana can pay Iroh's Lesson flashback casting
cost; resolving Waterbending Lesson then draws three before its separate payment
or discard choice. The restriction does not permit spending that mana on the
resolution-time waterbend cost. This distinction follows the printed rules and
must remain visible rather than being merged into an undifferentiated mana pool.

Failure and recovery: losing Iroh after casting, countering the spell, or moving
it off the stack another way must still exile it. A stale Command must not repeat
draws, taps or payment; replay and exact forks must retain the pending decision,
restricted pool, provenance and copied token characteristics. The shared zone
movement, Command validation and journal own these guarantees. No new recovery
service or consumer-owned rules state is implied.

Counterexamples to simplification remain constraints: treating waterbend as mana
production incorrectly triggers mana bonuses; deriving flashback replacement
from a currently present Iroh loses it after source removal; copying only a
registry identity loses the token's nonlegendary exception. The current model
keeps the necessary distinctions while deriving granted abilities from the live
battlefield. w9 and hash/schema changes identify the new meaning without
rewriting historical evidence. Academy consumes managym through manabot.

Next proof: the existing gate must validate current-world admission, full
learnability and affected debug/runtime suites, plus the configured release
browser and pinned Linux visual checks. Directed local tests and shared-primitive
parity do not establish independent rules agreement or full-deck strength.
The durable rules usage/ownership note already describes these contracts;
no usage or skill-instruction redesign is proposed.
