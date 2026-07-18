# Measure and Stress Production Study Fork/Apply/Return

## Problem

Current main has the production Study authority seam: `GameSession.fork_study`
resolves a canonical historical address, `StudyForkProvider` clones the retained
managym root, `StudyBranch.submit` applies a bound native structured offer, and
`return_to_recorded` consumes the branch while returning the canonical replay
decision plus its source digest. The focused proof covers one pass-priority
command and sibling isolation, but it does not yet provide production-shaped
latency/RSS evidence, sustained stress, explicit object-incarnation binding, or
an auditable zero-fallback execution receipt.

RUL-6 closes that evidence gap without changing the selected full-clone
representation or widening the Rules frontier.

## Demo

Run:

```bash
uv run scripts/bench_study_branch.py verify
uv run --extra dev pytest tests/etude/test_study_branch.py tests/bench/test_study_branch_benchmark.py -q
```

The checked artifact reports fork, offer-publication, structured-apply, return,
and end-to-end latency for 2,000 production Study cycles; retains 512 sibling
branches while measuring RSS; and admits only evidence with zero exactness,
privacy, incarnation, failure, or fallback mismatches.

## Approach

### Production boundary

Keep `root.clone_env()` and `env.step_structured(...)` as the only fork and
apply paths. Normalize native offer/submission rejection to
`StudyBranchUnavailableError`, invalidate each published native offer set after
one submission attempt, and verify that a rejected command did not mutate its
branch. A rejected command that did mutate is consuming and fails closed.

Extend the authority-private return value with a frozen execution receipt:

- driver: `full_clone/current_game_v1`;
- command path: `structured_offers/step_structured_v1`;
- published offer sets, accepted commands, rejected commands, and committed
  engine actions;
- fallback commands: exactly zero.

This is not a public protocol or replay-schema change. It makes the existing
absence of a Study fallback path executable evidence instead of a source-code
inference.

### Fixed production workload

Build one deterministic completed Etude match through `GameSession` with seed
7, UR Lessons versus GW Allies, hero first-offer policy, villain last-offer
policy, and no live authority fallback. Select the earliest retained hero
priority decision whose native offer surface includes a targeted cast with an
`ObjectRef`. The workload always targets the first authority-published
candidate.

One measured cycle is exactly:

1. `GameSession.fork_study(address)`;
2. `StudyBranch.structured_offers()`;
3. `StudyBranch.submit(bound_submission)`;
4. `StudyBranch.return_to_recorded()`.

Each cycle checks the source digest, canonical returned decision, source offer
projection, object entity/incarnation, post-apply zone change, viewer-relative
observation, replay bytes, presentation events, consuming close, and execution
receipt.

### Stress and failure cells

- 2,000 sequential cast/apply/return cycles after warmup, retaining raw
  nanosecond samples for fork, publish, apply, return, and end-to-end latency.
- 512 simultaneously retained siblings with published offer sets; apply on
  alternating branches, re-read the untouched siblings, return all branches,
  and record process RSS at each phase.
- Typed failures for submit-before-publish, unknown offer, unsupported native
  decision surface, invalid address, missing address, other viewer, retained
  root drift at return, and retained root drift at later fork.
- A caller-mutated projected incarnation is ignored because execution remains
  bound to the authority-held native offer set; the intended object moves and
  the retained source does not.

### Checked evidence

`scripts/bench_study_branch.py` owns measurement, report rendering, artifact
hashing, exact source/binary identity, and fail-closed verification.

- Contract: `docs/benchmarks/study-branch-contract-v1.md`
- Raw artifact: `experiments/data/rul-6-study-branch-v1.json`
- Human report: `experiments/rul-6-study-branch-v1.md`

The verifier recomputes summaries from raw samples and rejects stale source or
contract identity, artifact tampering, a failed performance gate, any mismatch
counter, any privacy exposure, missing incarnation evidence, untyped failure,
or nonzero fallback count.

## Performance gates

The first interactive budgets are deliberately wider than the measured local
baseline while still product-relevant:

- end-to-end p95 at most 3 ms;
- fork p95 at most 1 ms;
- structured apply p95 at most 1.5 ms;
- return p95 at most 1 ms;
- at least 500 sequential cycles/s;
- at most 128 MiB RSS delta while retaining 512 siblings.

These are gates for this fixed workload and source/binary identity, not claims
about arbitrary hardware or future content packs.

## Scope

In scope: the production Etude Study adapter, typed failures, authority-private
execution receipts, fixed-match performance and RSS measurement, retained-root
and sibling stress, exact return, viewer privacy, object-incarnation binding,
and zero fallback.

Out of scope: representation changes, page COW or undo, public Study API/UI,
saved branches, replay reconstruction, alternate viewers, arbitrary historical
pack coverage, search/evidence generation, and broader Rules work.

## Done when

- Study native errors are typed at the Etude boundary and rejected submissions
  cannot reuse a published native offer set.
- Every consuming return carries a validated structured-only, zero-fallback
  execution receipt.
- The canonical 2,000-cycle/512-sibling artifact passes all performance and
  zero-mismatch gates.
- Focused production, benchmark-verifier, and debug full-clone contract tests
  pass.
