# Implementation notes

- The current visible-card encoder has no `CardDefId`/name identity to share
  with a belief row. Reusing its dense projection would erase the distinction
  between cards with the same public numeric features. The keystone therefore
  shares the current-state attention/decision core while using an explicit
  schema-bound card-definition embedding for belief rows. A later content
  vocabulary migration can make the card embedding itself shared without
  weakening the intervention boundary.
- The checkpoint-loader change correctly trips the retained INT-8
  `loader_source_identity` and arena source-drift guards. Those artifacts bind
  the old loader bytes. They are not rewritten or ported; future experiment
  models and frozen evidence are regenerated under the exact schema-bound
  checkpoint ABI.
- The supervised belief slice uses deterministic hashed features of canonical
  `ViewerHistory.events`. The current viewer-history contract exposes event
  digests rather than decoded event semantics, so the bounded overfit proves
  exact-world NLL, authority isolation, and runtime wiring—not generalization
  from card/action meaning. A held-out learned-belief continuation should first
  add a schema-bound semantic history projection and then evaluate by whole
  match and opponent version.
