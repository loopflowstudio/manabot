# CI under 8 minutes (Task 38ef25, also delivers MTG-263)

## Docs written before implementation

- `docs/development.md`, CI section: the Integration row of the job table and
  the new **Job time** paragraph (8-minute limit per PR job, how to read job
  times with `gh run view`, order of remedies, Engine Budget exception).

The example output in that paragraph is illustrative; replace its numbers and
group names with the PR's own CI run once the groups are final. If a group is
renamed or Perf proof/training smoke move out of `evaluation`, update the
Integration table row. The comment above the matrix in
`.github/workflows/manabot-integration.yml` still quotes the old group times.

## MTG-263 brief (deliver here; say so in the PR so it can be closed)

"Measure per-file durations, then either balance file shards across matrix
groups or adopt pytest-xdist where tests are process-safe (check tmp dirs,
ports, DuckDB files, subprocess fixtures). Target: no group above ~8 min wall
time; report before/after from CI runs. Do not drop or weaken tests."

## Assumptions

- "A file listed in no group does not run in CI" is stated as current
  behavior (groups name exact paths); no completeness check is promised.
- The macOS nightly is not held to the 8-minute limit (the Task says "on the
  ordinary ubuntu runner").
