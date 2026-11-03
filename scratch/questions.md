# Questions and assumptions

## MTG-237 — headless, 2026-11-02

Jack Heart's steer permits publication and demo, without shipping approval.
Headless decisions below are delegated judgment, not human statements.

- Interpret the committed baseline as a reference commit measured A/B on one
  runner: identical code varied 6.8k–10.3k steps/s across CPU models. Retain
  10%; two EPYC 7763 runs agreed within 2.5%. Jack Heart's PR review decides
  this interpretation and the native-throughput interpretation below.
- Use native `Env::step` random-legal play, 512 seeds, skip_trivial on. The
  retired rul-9 Python semantic-Command runner was deleted in #343; its 588.5
  steps/s is not comparable. Every ordered search score is part of correctness,
  including losing scores. Rules/deck/benchmark changes reaching this workload
  require a deliberate reference move when digests change.
- A CI job owns its VM; pin `ubuntu-24.04`, report the CPU, and compare only
  within it. Candidate-first order preserves recovery when the reference no
  longer builds; retain it absent evidence of bias exceeding tolerance.
- `measurement.json` identifies the candidate; `record` ignores the nested
  reference run and preserves tolerance. Reference-build recovery is tested
  with injected binaries/Cargo failure/metadata and a copy of the upload path;
  only comparison-failure transport has hosted proof.
- Keep merged main `51fbacbbf`'s optimization. Run 41555845634 demonstrated
  30.26% less worst-reply time, 28.36% less total search time and 36.65% less
  time per step, with matching digests/counts. Production adoption is complete;
  publication and demo remain. The [design](mtg-237-engine-performance-budget.md)
  and [demo evidence](engine-budget-demo.md) own details.

Inherited MTG-236 decisions (including deferred interning and quiet-machine
absolute targets) are at `51fbacbbf:scratch/questions.md`. T5/decklist assumptions,
the unresolved printed-P/T decision and sync evidence are at
`24cfc0a8b:scratch/questions.md`; older Wave/evaluation context is at
`460732f81:scratch/questions.md`. They select no MTG-237 work. Full preceding
MTG-237 assumptions, including the temporary T6 measurement ref (now deleted),
are at `4b5e9c0e4:scratch/questions.md`. Pre-curation uncommitted notes are
preserved under `.cache/engine-budget-demo/pre-compress-4b5e9c0e4/`.
