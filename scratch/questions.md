# MTG-337 assumptions

Active plan: `scratch/payment-search.md`. Reversible assumptions taken headless:

- The harness is not a CI job and asserts nothing ("no work contract, no
  suite-wide assertions, no ceilings machinery"); the earlier replay test and
  its call limit are removed rather than kept beside it.
- State names, layers and default sizes are a first cut; the doc table follows
  the harness's `--list`.
- Python-side layers are a separate time-only script over the Rust export, so
  no Python board builder exists. Academy frame construction needs a live
  session (match, revision, viewer, stop state) and is omitted; the steer asked
  for it only "if they are cheap to reach".
- The audit table goes in the PR body, not a committed report.
- `restricted-filter` repeats three card restrictions; `distinct-restrictions`
  adds Powerstone for the catalog's maximum of four. Neither claims arbitrary
  pairwise-different restrictions.
- Only the local scan reduction is taken; nested-filter pruning and cross-step
  payment caching stay outside this Task (reasons in the plan and `--list`).
- Compress pass 2: with `recorded` gone, no constructed checkpoint reaches the
  text renderer's unsupported `announcement` stage or a closed decklist, so
  the script and harness fail on those errors instead of printing
  `unavailable` rows. A future state that reaches one should restore a row or
  fix the renderer.
- The recorded-state choice is settled as (b) by comment `9f42783b` (quoted in
  the plan), which also approved shipping.
- Sync onto main (2026-11-13): retain this Task's assumptions despite main's
  deletion of the shared scratch path; these notes belong to the active branch.
