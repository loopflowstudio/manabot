# MTG-195 accepted design and execution boundary — 2026-10-26

Design: [trigger-order.md](trigger-order.md). Jack Heart accepted the design with
“seems reasonable” in the MTG-195 primary conversation on 2026-10-26, responding
to the three review points below.

- **Accepted representation:** sequential next-to-stack choice and shared
  observation/action contract.
- **Execution boundary:** Jack Heart authorized implementation with “implement for
  sure” in this conversation. Implementation and local validation proceed here. Jack Heart then selected
  concept-review option 1a and authorized landing with “after that you can land”
  on 2026-10-27. Implement the shared exact-ability descriptions, validate, publish,
  merge and verify completion. Training experiments and paid compute remain excluded.
- **Accepted learnability interpretation:** MTG-195 asks for “full learnability evidence,” but
  `b3681297f16e27a9e42b5e49f4fe928292fe0bb2` (PR #372) deliberately removed the
  full verifier and PPO positive control at Jack Heart's request. Extend and run
  the maintained model suite, both-preference
  choice-separability fits, and ordinary Elemental PPO/checkpoint reload checks.
  This is not the old three-seed/65,536-transition diagnostic or strength evidence.
  This satisfies the reviewed interpretation; do not resurrect the removed verifier.
- **Accepted scope limit:** implement the two-pass scheduler and exercise
  both trigger classes with typed native fixtures. Current registered conditions
  cannot trigger on another ability triggering; adding that condition, its causal
  event machinery, or new cards is not assumed authorized. The design labels this
  proof limit; it does not claim general CR 603 implementation.
- **Reversible demo assumption:** use the existing scenario setup surface before
  ordinary Commands for a deterministic existing-card human/agent witness. Record
  and replay its explicit root recipe; do not call it a naturally reached authored
  prefix. No new production state-injection endpoint is needed.

Inherited MTG-184 questions were committed, unrelated parent context, not MTG-195
direction. Their complete retained source is
`ce43a5cf76d52b99c4c13c6debb21aab79ac9f82:scratch/questions.md`; that Task's model
choices, performance receipts and shipping authorization do not transfer.
