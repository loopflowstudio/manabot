# Assumptions (reversible)

- Merged PR #489's branch early to reuse its harness and docs section; #489 has since landed.
- A/B report name: `experiments/precision_ladder.md` / `.py`.
- Ladder keys on `ModelConfig.hidden_dim` only (the Task says "model width"); depth ignored.
- Width-changing builders re-resolve `auto`; explicit precisions are kept. Taken as the
  simpler reading of "explicit in code"; otherwise `with_capacity(batched_baseline(), width=384)`
  would silently keep float32 from the narrow base.
- Saved stage dicts with no `precision` key (written before PR #489) read as `float32`,
  not `auto`: they trained in float32 and resuming them must not switch to bf16.
- Did not start the L4 measurements: MTG-318/319 still held the L4 queue on 2026-11-12.
- A/B length: 64 updates, not "a few hundred". At an assumed ~150 s/float32 update at
  w384l8 on an L4, 2 arms × 2 seeds × 300 updates is ~USD 30, over the USD 25 no-approval
  line and the Task's USD 8. 64 updates fits USD 8 reserved (7.80 training + 1.50 eval,
  cap 10). Longer needs a person to approve the amount.
- "Frozen anchor" = the fixed heuristic over ELEMENTAL_PAIRS, plus bf16-vs-float32
  head-to-heads; no frozen checkpoint anchor exists for the w384 Elemental recipe.
