# Belief evidence

When simplifying belief plumbing, preserve exact probability and receipt bytes.
The reference model's consumed history range counts `ViewerHistory.events`;
the learned model's range counts `semantic_events`. These are distinct evidence
coordinates, even though both updates use the same receipt constructor.
PPO and BC checkpoint writers share `belief_checkpoint_fields`; keep schema
admission and rejection rules there.

Qualify belief-learning claims by the sampled population and split unit. The
160-deal scripted population demo holds out episodes, not opponent policies or
event order. Semantic commitment pooling is order-invariant; receipt hashes are
provenance, never history features. A nonzero intervention policy delta proves
the input boundary, not a trained held-out strategic action change. Preserve
these distinctions when updating Wave memory or Linear KRs.

For a belief gate, run `uv run manabot belief-demo`,
`uv run manabot belief-learn-demo`, and `uv run pytest tests/belief/ -q`.
Changes to shared Agent/checkpoint plumbing also require the affected
`tests/model/`, `tests/agent/`, and `tests/sim/` consumers. CI's existing
integration selection does not include the belief suite. Preserve exact
marginal bytes and encoding receipts when optimizing projection; compare
prior, conditioned, and nonuniform sparse distributions at a native root.
Checkpoint or source-binding changes also require
`uv run pytest tests/etude/test_belief_advisor.py tests/etude/test_live_advice.py -q`:
these consumers bind frozen source/checkpoint artifacts. Preserve the evidence
and report typed unavailability when those artifacts are no longer admissible.
