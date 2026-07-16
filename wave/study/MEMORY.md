# Study memory

## Product identity

> "I love the experience of playing chess and go with engines showing me
> visually what else is worth thinking about and allowing me to see superior
> lines. Would love to make the same thing for MTG."

Study is the engine-analysis and human-learning surface of Etude Fantasia's superhuman
AI research program. It is not merely a replay viewer and not a generic
analytics dashboard. The player should experience a better idea on the same
Magic table, then be able to challenge, branch, compare, annotate, and share it.

The product synthesis from the July 2026 research is:

> Triage like Chess.com; explore like OGS; retain 17Lands' replay and sharing
> ergonomics; handle hidden information in a way only an AI-native Magic engine
> can.

## Durable decisions

- Guided Review answers "where should I look?" before free Study answers "what
  if?"
- Retry precedes reveal. Do not remove the pleasure and evidence of finding a
  stronger line oneself.
- The table is the primary analysis canvas. Highlight and ghost cards, mana,
  attackers, blockers, targets, stack objects, and resulting semantic beats.
- Magic alternatives are plans and contingent response trees, not one
  deterministic principal variation.
- Default to the historical player's knowledge. Hindsight is an explicit
  alternate lens.
- Never collapse policy probability, search value, visits, world robustness,
  and uncertainty into one confidence number.
- Branches are ephemeral by default and always retain a one-click return to the
  recorded match.
- Explanations must be grounded in typed semantics, resources, and searched
  contingencies. An LLM may phrase evidence but may not invent it.
- Study behavior can be valuable training/evaluation data, but collection and
  use must be transparent and mode-labelled.
- Mobile is table-first with a bottom sheet and swipeable landmarks, not a
  compressed desktop inspector.

## Ownership boundaries

- **Game owns:** live play, canonical replay, semantic presentation,
  accessibility, visual polish, recovery, packaging, and client adapters.
- **Rules owns:** semantic programs, structured offers and commands, viewer-safe
  projections, search evidence, exact forks, rollback, and deterministic state.
- **Study owns:** landmark triage, analysis evidence presentation, retry,
  alternate-line experience, branch lifecycle, annotations, sharing, and the
  human study/research loop.

Study must consume canonical frames, offers, commands, and presentation events.
It must not infer legality or semantic events from snapshot differences.

## Existing seams

- The replay page already loads traces, navigates frames, and plays semantic
  presentation events.
- `presentationInspectorRows` is explicitly intended to pair canonical semantic
  rows with future policy/search metadata.
- The engine has viewer projections, deterministic match hashes, policy/value
  models, determinized search, and increasingly explicit safe-fork contracts.
- Experience protocol v1 carries revisioned frames, offers, presentation, and
  recovery identity across Rust, Python, and TypeScript.

## Research evidence

- Chess.com Game Review: summary, selected key moments, progressive reveal,
  Retry, and nearby unrestricted Self Analysis.
- OGS AI Review: board-native candidates, win/score graph, worst-move triage,
  selectable analysis runs, and interactive analysis of variations.
- 17Lands: faithful Arena reconstruction, action/turn navigation, deep-linked
  positions, honest missing-data warnings, and an external social feedback loop.

## Open tensions

- Landmark selection must be useful before value estimates are fully calibrated.
- The default value target may be game win, match win, or a richer strategic
  vector; labels must not outrun evidence.
- Sampled-world summaries must be understandable without hiding their budget or
  uncertainty.
- Policy/search/model comparison should teach rather than turn the table into a
  benchmark console.
- Saved branches and shared decisions need durable identity without freezing
  every internal search artifact forever.
- The first slice should prove one complete study moment before building a
  general annotation library or collaborative review system.
