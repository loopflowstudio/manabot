# Shared-effect v7 demonstration

Task 5da5f4 · demo decision pass 1 · 2026-11-04 05:33 UTC

**Decision: advance the bounded repair to delivery.** The preceding pass intended
to remove per-object/per-keyword expansion from capacity admission while retaining
complete effects and routing them into affected objects before attention. The
current implementation demonstrates that behavior through native replay, the
ordinary policy, learner and checkpoint loader. No concrete implementation gap
was found that warrants returning to implement.

This was an autonomous, headless demonstration. No person observed this run or
gave new feedback. Jack Heart's recorded 2026-11-04 03:01 UTC direction was
“this sounds good. ship it,” explicitly accepting the bounded shared-effect repair
and separating engineering correctness from learning-quality claims. My decision
applies that instruction; it does not infer acceptance from absence or claim that
Jack selected a learning architecture winner.

Current design: [effective state and shared effects](5da5f4-design-effective-state.md).
Usage: [manabot README](../manabot/README.md).
Architecture: [observation walkthrough](../docs/observation-architecture.html).
Published change: [PR 419](https://github.com/loopflowstudio/manabot/pull/419),
head `91635f583bc50d8bda2bbd0abe4a24031ddaf3ce`.

## Experience and observations

The retained native action prefixes were replayed with their supplied seeds,
players and trivial-decision settings. Requesting old and prospective inputs at
the resulting decision produced:

| Position | Existing frozen input | Explicit v7 result |
| --- | --- | --- |
| Seven Drill activations, ten affected objects | v5 rejects `facts 140 > 128` | 7 complete payloads, 14 detail tokens, 70 membership edges |
| Nine activations, fifteen affected objects | v6/256 rejects `facts 270 > 256` | 9 complete payloads, 18 detail tokens, 135 membership edges |

Each payload retains both lost keywords. The native rules records remain intact;
this is lossless factoring, not semantic deletion of repeated activations. The
first header in the 270 case owns detail offset/count `(0, 2)` and membership
offset/count `(0, 15)`. Its first three membership rows are `(163, 1)`, `(164, 2)`,
`(165, 3)`: one-based public object joins and original record ordinals. IDs route
the payload to objects; the model does not learn membership from numeric IDs.

The fresh default remains v6. Shared buffers allocate 31,744 padded bytes in both
cases; active shared data occupies 1,232 and 1,944 bytes respectively. The smaller
active representation is not a padded-memory or throughput improvement claim.

The following actions completed successfully:

1. Replay both retained prefixes for both viewers with open and closed decks.
   Native scalar, in-place, vector and Python reference buffers agree exactly.
   Closed opponent deck rows remain zero. Hidden-hand permutations leave the
   viewer's text capture and tensors unchanged.
2. Pass v7 inputs through all configured reducer selections, compute backward
   gradients, permute object rows and rebind joins, save and reload checkpoints.
   Outputs remain finite, shared gradients flow, object-addressable policy
   outputs survive permutation, and checkpoint outputs agree. Duplicate links
   contribute twice. `rows` and `typed_pool` use the same v7 reducer and are not
   represented as distinct scientific candidates.
3. Execute `local_smoke` with explicit v7 through `execute_regime`, then load the
   exported raw checkpoint. Run `300661dc3dd24ac8ae2e2aee199e8900` completed one
   learner update, 290 learner transitions and two games on CPU with one thread.
   Raw checkpoint SHA256:
   `b380d54c9ce9dff507c5d380ef5d2d15384a9d19f6fa6e8a566ac5ef76e41fa7`.
   This is a bounded execution proof, not replicated learning evidence.
4. Run four debug native counterexamples: preserved rules records; duplicate
   additive memberships and excluded later entrants; intervening grants with
   different expiries; and exact source A/B bindings that diverge after A leaves.
   All passed. Equal v6 source-dependent inputs become distinguishable in v7.
5. Check rejection paths: absent historical provenance and overflow of each
   shared block fail explicitly, without truncation. Frozen v1–v6 encodings ignore
   the new shared projection.

## Reproducible proof and retained failures

From the repository root:

```sh
uv run --extra dev --extra artifacts pytest tests/verify/test_shared_effect_transport.py tests/model/test_shared_effects.py -v
CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path managym/Cargo.toml --test rules_tests shared_effects -- --nocapture
```

Observed: **23 Python checks passed in 5.63 seconds; four debug native checks
passed.** The initial Cargo invocation used the nonexistent target `rules` and
failed before running tests. Correcting it to `rules_tests` succeeded; both logs
are retained rather than treating the initial attempt as a pass.

Evidence under [`.runs/representation/demo-v7/`](../.runs/representation/demo-v7/):

- [Observed counts and source/native identity](../.runs/representation/demo-v7/observed.json).
- [Python demonstration log](../.runs/representation/demo-v7/pytest.log) and
  [JUnit results](../.runs/representation/demo-v7/results.xml).
- [Initial Cargo target error](../.runs/representation/demo-v7/native.log) and
  [corrected native proof](../.runs/representation/demo-v7/native-corrected.log).
- [Ordinary training receipt](../.runs/representation/demo-v7/pytest/test_shared_effect_policy_uses0/training/run.json),
  with exported checkpoints retained beside it.
- [Hosted check snapshot](../.runs/representation/demo-v7/ci.json).

The earlier full local gate is recorded in
[publication verification](publication-verification.md). Its statement that
hosted checks are pending is superseded by this snapshot: Engine Budget, Rust,
Python unit tests, long integration, protocol/semantic conformance, Academy and
package checks pass. Main integration is still in progress; merge readiness is
not yet established. No hosted check was observed failing.

## Acceptance boundary and next action

No design changes are needed for this accepted repair. Advance the existing Graph
to its delivery step, wait for remaining required CI, and ship PR 419 when those
gates hold. Do not restart the Graph, mark the Task complete, or merge from this
demo step. MTG-213 still needs an explicitly prospective input/protocol selection;
its frozen v3 run is unchanged.

Replicated learning, v7 CUDA execution, relation-table redesign and persistent
memory remain unproved and outside this bounded delivery. Sum scaling and learned
embedding collisions remain research questions. The pre-existing keyword
grant/removal ordering defect in [questions.md](questions.md) remains unresolved;
it forbids assuming arbitrary timestamp coalescing is sound. No such coalescing
ships here. Capacity is bounded, not guaranteed for all legal games.

There was no paid allocation, no external notification and no new human approval
request. All processes started for this demonstration exited. No rendering
environment was available; this is a training/native demonstration, so there is
no claim of browser interaction or screenshot review. The current record is this
Markdown note; the existing architecture walkthrough remains the presentation.
