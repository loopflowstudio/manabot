# Compact clone plus undo preregistration v1

Task: W2-198

Contract: `manabot.search-branching.v1`

Registered: 2026-07-16, before implementing or timing
`compact_clone_undo/current_game_v1`.

## Correctness gate

No timing artifact is valid unless full clone and compact clone plus undo have:

- zero witness, outcome, cap, event-boundary, action-order, or RNG mismatches;
- identical simulation and transition counts plus ordered logical checksums for
  every matched cell, seed, and worker; and
- passed the unchanged branch contract on both fixtures, including nested
  rollback, failed-command no-ops, reverse isolation, hidden projections,
  allocation rollback, zone order, and stale object references.

## Performance expectations

These expectations interpret W2-198 evidence; they do not select a branching
representation or complete Search KR2.

- A material sequential win is at least 20% higher simulations/s in both
  `flat-single-64-v1` and `flat-saturated-64-v1`, with no more than 10% worse
  p99 root latency or peak RSS in either cell.
- Retained cells are expected to remain within 10% of full-clone throughput
  and peak RSS because every live slot still requires an exact clone. A larger
  regression is evidence of journal overhead. A memory-win claim is invalid
  unless `max_live_states` and every workload dimension remain identical.
- Report simulations/s, transitions/s, root p50/p95/p99, absolute peak RSS,
  peak RSS delta, max live states, eager forks, checkpoint copies, journal
  marks/entries/rollbacks, peak journal bytes, rollback time, cap rate, and
  deterministic checksum status for every primary cell.
- Unsupported allocation or COW counters remain `null` with a reason; they are
  never reported as zero.

Whether the expectations hold or fail, retain the raw matched results. Dense
page-COW plus undo remains W2-199, and no representation decision is permitted
before that evidence exists.
