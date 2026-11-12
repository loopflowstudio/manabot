# Assumptions (reversible)

- Merged PR #489's branch early to reuse its harness and docs section; #489 has since landed.
- A/B report name: `experiments/precision_ladder.md` / `.py`.
- Ladder keys on `ModelConfig.hidden_dim` only (the Task says "model width"); depth ignored.
- Width-changing builders re-resolve `auto`; explicit precisions are kept. Taken as the
  simpler reading of "explicit in code"; otherwise `with_capacity(batched_baseline(), width=384)`
  would silently keep float32 from the narrow base.
- Saved stage dicts with no `precision` key (written before PR #489) read as `float32`,
  not `auto`: they trained in float32 and resuming them must not switch to bf16.
- A/B length: 300 updates per arm after the approved USD 30 budget (Task cap USD 35);
  the earlier 64-update plan is superseded.
- "Frozen anchor" = the fixed heuristic over ELEMENTAL_PAIRS, plus bf16-vs-float32
  head-to-heads; no frozen checkpoint anchor exists for the w384 Elemental recipe.
