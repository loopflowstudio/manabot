# Assumptions (reversible)

- Merged PR #489's branch into this one to reuse its harness and docs section, rather than
  waiting for it to land.
- A/B report name: `experiments/precision_ladder.md` / `.py`.
- Ladder keys on `ModelConfig.hidden_dim` only (the Task says "model width"); depth ignored.
- Width-changing builders re-resolve `auto`; explicit precisions are kept. Taken as the
  simpler reading of "explicit in code"; otherwise `with_capacity(batched_baseline(), width=384)`
  would silently keep float32 from the narrow base.
