# Questions and assumptions

Inherited T5 and decklist assumptions, their unresolved printed-P/T decision,
and sync evidence are retained at `24cfc0a8b:scratch/questions.md` (T5's
source is main `a8abf01b1`). Older Wave rename and evaluation context is at
`460732f81:scratch/questions.md`. These histories select no MTG-237 work.
The notes-only sync onto main `a8abf01b1` preserved both histories and all
MTG-237 implementation. It supplied no shipping approval.

## MTG-237 (engine budget), 2026-11-02 — headless choices

- The headless steps/s workload is native (`Env::step`, both seats uniformly
  random over legal actions, 512 seeded games, skip_trivial on), not the
  rul-9 Python semantic-Command path (its runner was deleted in #343). It
  measures the engine the training environment steps, builds in the same
  cargo invocation as `bench_search` and needs no extension build in the job.
  Its steps/s is therefore not comparable with rul-9's 588.5 (Python,
  semantic Commands); the baseline is its own.
- Digests are asserted identical for both workloads, so any rules change
  reaching the curated decks (or the deal) moves the baseline deliberately
  with a reason, as the review asked for search results. The random-play
  digest also sees every legal-action count; search now covers all ordered
  scores, not only winning actions. Expected cost: one `record` per rules PR touching these cards.
- The budget runs as a `ci.yml` job via `workflow_call` so the Tests Result
  aggregate owns it and docs-only PRs skip it; every job is its own VM, which
  is the quiet runner. `ubuntu-24.04` is pinned rather than `ubuntu-latest` so
  the baseline's hardware profile does not move with the label.
- Counterexample to the design's committed absolute numbers (2026-11-02): the
  four dispatches landed on three CPU models (EPYC 7763, Xeon 6973P-C, EPYC
  9V74) and the identical T4 engine spread 6.8k-10.3k steps/s between them,
  far outside 10%. The comparison is therefore A/B on one machine: the job
  rebuilds the reference commit with the change's benchmarks and compares
  the candidate to that fresh run; the file holds the reference commit,
  digests and its recorded figures. Two EPYC 7763 runs of one engine agreed
  within 2.5%, so 10% stands for now. Jack Heart's review decides whether to
  keep this reading of "committed baseline".
- T6's head (`2c1ba169f`) is unpushed, so measuring it on CI needs its commit
  reachable from origin; pushed as a plain ref under `refs/engine-budget/`
  for the dispatch and deleted afterwards (not a branch, no PR).
- Headless readings (2026-11-02; decisions, review findings and the
  proposed Wave memory are in the plan): the two formerly draft decisions
  (A/B against a committed reference commit; native `Env::step` random play)
  are read as implied by the accepted outcome plus the runner evidence, so no
  review returned Blocked; Jack Heart's PR review confirms or reverses them.
  The artifact is named `engine-budget` in both modes (the run id identifies
  a measurement); `measurement.json` carries the full `lscpu` model name,
  orientation only; compare's FAIL hint leaves `<run-id>` as a placeholder so
  the line reads the same in CI and in tests; the artifact also contains the
  `reference/` run, which `record` ignores; `record` takes the tolerance from
  the committed file rather than defaulting it. Running the candidate before
  the reference on the one VM is assumed not to bias the comparison beyond
  the 10% tolerance (two runs of one engine on separate EPYC 7763 machines
  agreed within 2.5%); the order exists so a run whose reference cannot build
  with the change's benchmarks still records its candidate. Reverse it only
  with evidence that order matters.
- Fourth headless concept review at 078d37ef4: interpret the Task's "score
  digests" literally as preserving the ordered action scores, not just the
  winning action. This follows the accepted correctness requirement; it is
  not new approval from Jack Heart. The counterexample and next implement
  direction are in the working design. Keep all completed workflow work and
  historical measurements; a new digest needs a new recording, not rewritten
  historical evidence. No open product choice; no shipping approval.

## MTG-237 demo repair — 2026-11-02

The bounded reference-build recovery proof uses injected benchmark binaries,
Cargo failure and machine metadata, while executing the checked-in workflow's
recording shell and the real adoption CLI. Copying the declared upload path is
the simpler local substitute for GitHub artifact transport. This proves local
failure-path recovery only; publication/demo retains the hosted-job evidence
boundary. No production baseline move or load generation is needed.
