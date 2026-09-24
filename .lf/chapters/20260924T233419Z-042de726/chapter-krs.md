# Proposed next chapter — play full games against trained challengers

Status: direction proposal for human review, not applied to Linear or Wave charters. Source: the User's 2026-09-24 direction conversation.

Outcome: the User wants to play complete Allies versus Lessons games in Etude, and can repeatedly train and challenge bots progressing toward beating them. Keep exactly one Project and three KRs per continuing Wave. Proposed roster: Game, Intelligence, Rules; Study is part of Game.

This is a new chapter proposal, superseding simple consolidation of old priorities. one-project-per-wave.md preserves the 56 historical KR claims and their consolidation mapping as review input. They do not automatically become acceptance gates for this chapter. The formal baseline review and explicit carry/rewrite/complete/retire dispositions remain required before final application.

## Game

1. **Make the game's progress legible.** Address the User-reported lack of transitions so the User can follow what happened after an action, what changed on the table, and whose decision comes next throughout an Allies versus Lessons game. Use authoritative semantic events and validate through real play; exact animation design and timing follow observed needs.
2. **Make every intended play straightforward to execute.** The User can find and complete the legal play they intend in every encountered decision, including high-branching and rare cases, without fighting the controls. Treat any reported awkward action as a blocker to the full-game outcome, reproduce it as an exact position, fix it, and validate the interaction with the User. Build coverage across the selected matchup's reachable interaction families; do not infer universal usability from a few successful games or average ratings. Preserve the full legal choice space and server Command authority. Grouping and staged selection are candidate designs to test, not accepted implementations.
3. **Complete games against trained challengers.** Make identified trained checkpoints easy to challenge in the same demo, demonstrate the User completing games and choosing to play again, and retain reproducible match records tied to the bot version. Continue discovering other friction through play; automated terminal tests cannot substitute for the human outcome.

## Intelligence

1. **Make training repeatable.** Run a documented training-to-play loop for the existing matchup that emits usable, versioned checkpoints with reproducible inputs and measured compute cost.
2. **Demonstrate improvement.** Compare candidates against the current bot and appropriate frozen controls at declared compute, across multiple seeds and both deck assignments, reporting legality, playing strength, competencies, uncertainty, and cost. Retain honest negative results and the next training decision.
3. **Produce a human challenger.** Evaluate a trained candidate against the User in recorded Allies versus Lessons games and demonstrate repeatable wins under an agreed match protocol. Define the cohort, deck assignments, stopping rule, and success criterion before claiming the bot beats the User; one isolated victory is insufficient.

## Rules

1. **Make the selected matchup play by its actual rules.** Complete representative Allies versus Lessons games with faithful card behavior, complete legal choices, deterministic replay, and no hidden-information leakage. Start with the User-reported Learn/Lesson discrepancy: the current engine deliberately supports only discard-then-draw, omitting the outside-game Lesson choice. Establish the intended Lesson pool and format semantics and support the real interaction; do not silently simplify a mechanic to fit missing infrastructure. Turn each discovered discrepancy into a regression exercised through the real demo and agent path.
2. **Support the real training workload.** Measure and remove demonstrated engine or possible-world bottlenecks, meeting declared throughput, latency, and memory budgets on actual Intelligence consumers while preserving semantics.
3. **Keep play and training on the same authority.** Exercise trained agents, human Commands, replay, and any required search forks against consistent revision-bound semantics, with exact isolation and reproducible failures and no consumer-specific rules reconstruction.

## Bounds and open planning work

- Keep the existing Allies versus Lessons matchup. Content or Lesson-pool setup needed for its expected mechanics is in scope; unrelated deck/card expansion is not. Advice, richer Study, learned beliefs, portability work, and new search methods must justify their contribution to this chapter outcome.
- Discovery is part of Game implementation; the User need not supply the UI diagnosis before work starts.
- The User's stopping rule is explicit: one awkward action can end a session. Known interaction friction is a completion blocker even when uncommon; frequency-weighted polish scores cannot waive it. Report exercised coverage and untested cases honestly.
- Review dates, training budget, baselines, human evaluation protocol, scoped Wave challenges, and every prior Project/open Task disposition remain to be reconciled.
- Keep the User's existing decisions: ETU-21 abandoned; ETU-31 deferred pending prerequisite work; ETU-55 provider-only ship distinct from scientific calibration closure; do not disrupt ETU-34 work.
- This terminal Run is not Task-bound. Formal baseline review, child planning, and tracked chapter archive continue in a Task worktree after direction acceptance, following start-chapter.
