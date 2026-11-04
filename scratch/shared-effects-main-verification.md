# Shared effects verified on main

Task 5da5f4, verify decision pass 1, 2026-11-04 UTC.

**Outcome: Follow-up; advance the verified bounded delivery.** PR
[419](https://github.com/loopflowstudio/manabot/pull/419) merged at
2026-11-04 05:55:57 UTC as `fb5ad2c1372f9c967b550f0b8ae0f84883f75563`.
The pre-existing keyword ordering defect is captured separately as **MTG-240**.
There is no concrete repairable gap in the accepted shared-effect transport.

## Authority and progress

Read the Task outcome, all delivered steers, the saved design, demo and gate
notes from `411d02503`, and the merged PR. Jack Heart's 03:01 UTC ship direction
accepts the bounded versioned shared-effect/pre-attention-routing repair. The
design and demo explicitly defer persistent memory and a newly frozen v7
replicated-learning study; neither is represented as delivered or proven here.
The earlier full research brief does not establish a scientific result.

The preceding pass intended to publish the corrected expiry test, pass final CI
and merge. New evidence establishes all three: the PR is merged, final-head
Tests Result and Engine Budget passed, and Linux/macOS package build and install
checks passed. Release publication was skipped, as expected for a PR. No PyPI
release is claimed. CI evidence is retained in
`.runs/representation/verify-main/pr.json`.

## Main identity and checks

Fetched `origin/main`. Its tracked tree and this supplied checkout's HEAD tree
both equal `e217b2e8981769774438b0ce99eeb64e6cf9905a`; their diff is empty.
Thus the checks exercise main's exact sources without moving the managed
checkout or creating another worktree.

- Built the release wheel from the root with `uv run --extra play maturin build
  --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`.
  Atomically installed its cp312 extension, SHA256
  `e0ac3baebbabdbcde1a0f6f0dc02a317a29af2ca20f89d4c1675de5a04ac34e6`.
- Ran `uv run --extra dev --extra artifacts pytest
  tests/verify/test_shared_effect_transport.py tests/model/test_shared_effects.py
  -q`: **23 passed in 5.90 seconds**. Logs, JUnit and ordinary learner/export
  artifacts are retained under `.runs/representation/verify-main/`.
- Ran `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path
  managym/Cargo.toml --test rules_tests shared_effects -- --nocapture`:
  **four passed with debug assertions**.

These checks replay the retained 140/270-fragment native action prefixes and
admit them as 7/9 complete payloads with 70/135 memberships. Scalar, in-place,
vector and Python tensors agree for both viewers; closed decklists and hidden
hand permutations preserve viewer safety. Provenance absence and each bounded
buffer overflow fail explicitly. V1–v6 ignore the new projection.

The model checks exercise pre-attention routing, duplicate contributions,
shared backward gradients, object permutations, legal action focus, checkpoint
reload, and a completed ordinary CPU learner update/export. Native checks retain
rules records, additive multiplicity, fixed affected sets versus later entrants,
intervening grants/expiry, and exact source bindings across a zone change.
The full earlier debug/Python gates and final hosted matrix supplement this
focused post-merge verification; they were not unnecessarily rerun locally.

README and architecture HTML describe explicit v7, unchanged fresh v6 defaults,
bounded capacities and the larger padded allocation. Neither smaller active
payloads nor these engineering checks prove learning or throughput superiority.
No browser behavior changed and no rendering environment is available; there is
no screenshot or visual-review claim.

## Separate finding

The retained counterexample in `scratch/questions.md` remains supported by main:
`keywords_with` unions grants then subtracts all lost keywords, while
`recorded_keywords_with` collects grants without ordering against individual
keyword removals. This predates PR419. Created **MTG-240, Apply keyword grants
and removals in layer timestamp order**, linked to this Task and PR419, with
the expiry sequence, code locations, compatibility constraints and acceptance
checks. Creation receipt: `.runs/representation/verify-main/follow-up.json`.
It is unstarted; this verification launches no additional work.

No paid compute, notifications or long-lived processes were started. The earlier
provider estimate remains an estimate pending final billing reconciliation.
MTG-213 adoption still requires a prospective protocol; frozen runs are unchanged.
Headless judgment applies the recorded bounded acceptance and advances this
Task without claiming human review during this run.

`lf task complete 5da5f4` succeeded. Loopflow retained the checkout because this
verification process and earlier execution records remain live or unresolved;
completion did not delete the evidence or move the running Graph.
