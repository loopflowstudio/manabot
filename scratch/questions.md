# Implementation notes

- `lf rebase --plan` failed because this Run's control lease is malformed,
  stale, stopped, or unknown. The checked-out design branch predates the exact
  belief package and possible-world Python bindings now on `origin/main`, so
  this implementation stays additive and mirrors their public adapter shape.
  `PossibleWorldSpace.from_engine` is the production boundary; the retained
  demo uses `from_fixture` until Loopflow can integrate the branch.
- The current visible-card encoder has no `CardDefId`/name identity to share
  with a belief row. Reusing its dense projection would erase the distinction
  between cards with the same public numeric features. The keystone therefore
  shares the current-state attention/decision core while using an explicit
  schema-bound card-definition embedding for belief rows. A later content
  vocabulary migration can make the card embedding itself shared without
  weakening the intervention boundary.
