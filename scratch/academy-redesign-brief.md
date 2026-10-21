Task 295863: Redesign Academy around the agent’s observations and actions

Jack Heart requested this on 2026-10-21 after jointly reviewing Elemental combat/Learn PR #328 and sacrifice/landfall PR #331. His accepted direction: “make the interface manifest the way the agents experience the data and action space.” He then clarified: “just top down evaluate and redesign academy from that perspective.” He requested a new Task worker opened in cmux.

Outcome: evaluate Academy from the top down and produce a coherent redesign in which its interface makes the acting agent's viewer-safe observations, decision structure, legal actions and resulting consequences understandable to a human. This is a whole-product concept/design task, not a narrow payment-panel fix or cosmetic refactor. Own the experience across entering a match, understanding the board and available information, identifying whose decision is pending, exploring and choosing actions, following consequences, reconnect/recovery, and replay/Study where those surfaces share the decision contract. Follow the experience through domain types, observation projections, offers, Commands, persistence and infrastructure.

Accepted principles:
- Humans and agents act through the same authoritative rules and complete legal action space. Show the same available information with clear language and visual structure. This does not mean exposing padded tensors, numeric feature vectors or implementation jargon as the product UI.
- Labels, layout, search and grouping can help people understand the shared data/actions, but do not silently select actions, hide strategic decisions, introduce human-only game semantics, or reveal information unavailable to the viewer.
- If an engine decision is awkward or redundant, evaluate improving the shared observation/action contract rather than concealing it with a human-only shortcut. Shared contract changes must preserve legal strategic choices, privacy, timing, deterministic replay and compatibility.
- The earlier suggestion of a separate compact human-only revisable casting/payment panel with suggested payment was superseded by Jack Heart's direction. Do not inherit it as approved design. Cancellation or undo is not approved merely by this Task; any proposal must define a shared explicit action and cannot erase committed costs or revealed information.
- Preserve canonical Command authority, stale rejection, reconnect restoring the actual pending decision, exact replay/forks, accessibility and reduced-motion behavior. Existing worlds, checkpoints and frozen evidence retain their identities. No paid compute, training or hosted deployment is included.

Method and observable acceptance:
1. Begin with concept-review of the real Academy experience and current agent observation/offer/action pipeline, inspecting existing documentation and real interactions. Implementation is evidence, not a commitment to existing layout.
2. Draft proposed usage documentation and an overall information/interaction model before designing code. Produce concrete screen/interaction proposals sufficient to assess the redesign, with current behavior, proposed behavior and unresolved product decisions distinguished.
3. Map each principal displayed concept and action to its authoritative observation, decision frame/offer and Command. Explain ownership and necessary changes across frontend, Academy, manabot and managym without creating another rules engine.
4. Challenge the design using ordinary play, a large choice, failure/stale submission and recovery. Include Learn's retrieve/discard/decline, modal/targeted casting, combat, sacrifice-as-cost versus resolution, Treasure mana/payment, and private fetch/search as representative cases, not the entire scope.
5. Deadly Precision provides one exact walkthrough: target, sacrifice versus four additional mana, Treasure activation/color, finishing payment, consequences and legal priority. Compare human display and agent choices at each step and reconnect before/after a committed mana ability.
6. Keep a durable concise design and update relevant worker guidance with the accepted shared-interface principle so later Academy/card workers receive it. Keep proposed redesign details separate from Jack Heart's accepted direction. State an implementable sequence and proof plan; do not declare runtime behavior delivered solely from a design or mockup.

Context: #328 https://github.com/loopflowstudio/manabot/pull/328 and #331 https://github.com/loopflowstudio/manabot/pull/331 have merged. Durable mechanics contracts are docs/rules/elemental-combat.md and docs/rules/elemental-sacrifice.md in this repository. They use typed definitions, announced action/payment state, private decisions and validated Commands. The two validation packs are not final ten-deck admission; MTG-164 owns that separately. The selected decks include Temur Lessons with Iroh, Grand Lotus and Ran and Shaw; BG is landfall plus sacrifice. Reuse shipped large-choice navigation and consequence/history work (MTG-77 and MTG-76) as evidence rather than reopening their old tasks. Keep this cohesive redesign in Game, collaborating with Rules/Intelligence where the shared contract requires it.

Repository: repo_95e2c543e080396cb3a71c89adda4a86
Task Work: task_2958630cfcbc4b2ea1ad51ca525643d6
Task directive snapshot synced at: not observed
Worktree: /Users/jack/src/manabot.295863-redesign-academy-around
Branch: jack-heart/295863-redesign-academy-around
Base commit: a04dcdc27d6d705cb2daade8b6071b7e29c35537

A Task has zero or one pull request. Merge leaves the Task open until completion. Optional later work uses ordinary Task creation; no follow-up disposition is required. Dependent pull requests belong to separate stacked Tasks.

<lf:task-workspace>
Checkout base: a04dcdc27d6d705cb2daade8b6071b7e29c35537
Current HEAD: a04dcdc27d6d705cb2daade8b6071b7e29c35537
Changes across the checkout base, index, worktree, and untracked files:


Patch (included=true, binary=false, truncated=false):

</lf:task-workspace>