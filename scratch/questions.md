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
- managym has no mulligan or opening-hand `Keep(k)` event. `ScryKeep` is a
  different in-game action and cannot stand in for that missing rule. The
  smallest current typed policy-dependent seam is managym's
  `PublicCommitment`: pass, cast, play-land, discard, or decline-discard with
  canonical public card names where applicable. Learned belief history now
  consumes those viewer-relative typed commitments and binds their closed kind
  set and card vocabulary in its schema; opaque event and command identities
  remain receipt evidence only.
- A standalone mid-game semantic `Observation` still carries only event
  identities, not the decoded earlier `PublicCommitment` sequence. The learned
  updater therefore generalizes over histories captured from the initial root
  and advanced through canonical transition receipts. Supporting arbitrary
  mid-game attachment requires a managym-owned semantic-history replay
  projection; this slice does not reconstruct meaning from hashes.
