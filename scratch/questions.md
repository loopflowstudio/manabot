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

## Sync onto main 51fbacbbf — 2026-11-02

Preserved MTG-237 demo assumptions and the new MTG-236 notes below.
Main's complete inherited history remains at
`51fbacbbf:scratch/questions.md`; the older T5/decklist history is already
referenced above. These inherited notes select no new MTG-237 work and
supply no shipping approval. Only this notes file required resolution.

## MTG-236 (derived state revisions) — 2026-11-02, headless

- `Revisions` has `characteristics`, `mana_pools` and `triggers`; the brief's
  `offers` counter is omitted because no derivation caches offers today
  (`current_action_space` is recomputed per decision), and a counter nothing
  reads would be a setup nothing uses. Add it with its first consumer.
- Mana pools get their own counter rather than joining `characteristics`:
  one static condition reads them (`UnspentManaAtLeast`, The Lion-Turtle),
  and pool writes happen on every mana tap and step boundary. `StaticIndex`
  records whether a source reads the pools, and the views follow
  `mana_pools` only then.
- Only the write sites the probe showed hot in playouts moved to typed
  accessors (events, taps, damage, life, pools, step flags, stack, trigger
  queues, incarnations, zone moves). Every other write still advances all
  counters through `DerefMut`; converting more is safe whenever a profile
  asks for it, because debug builds recompute every cached value on read.
- `TrackedState::move_card` advances `characteristics` for moves touching
  the battlefield or a graveyard and `triggers` for the battlefield only;
  hand, library, stack and exile moves advance nothing. Graveyard triggers
  (`InGraveyard`) are matched by scanning graveyards, outside the index.
- The bench machine carried load averages of 25–40 from other sessions
  during every run, so absolute times here are not comparable to the
  brief's (4.97 s / 151.8 s); deltas against the same-session baseline
  and identical score digests are the evidence.

## MTG-236 decision pass 1 — 2026-11-02, headless

- Item 5 (interned subtypes) is deferred, not built: after items 1-4 the
  release profile puts all of `Game::characteristics` at ~2% inclusive and
  its string building under 1%, so interning cannot move the acceptance
  numbers. Reversible: it stays listed in the PR body as deferred evidence
  for Jack Heart, and a separate Task covers it or mana payment search if
  the quiet-machine bench misses acceptance.
- The quiet-machine `bench_search` thresholds are owned by the gate/CI
  owner; this machine carried load 25-70 throughout (35 at this pass). The
  headless equivalent is the 31% CPU-second drop with identical score
  digests, projecting ~3.4 s / ~105 s against the 4.34 s / 128.7 s bound.
