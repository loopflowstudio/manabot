# INT-4 Visit Teacher Iteration — Engineering Smoke

## Question

Can one frozen command now execute the real visit-based Teacher-1 pipeline
through replay-aligned labels, a matched 2×2 student ablation, a four-agent
arena with neural PUCT, and a viewer-safe Study artifact?

This run is `engineering_smoke_non_admission`. Its one training seed and
two-game arena cells cannot support strength or method claims. The
pre-registered `iteration` profile remains 256 games, four arms × three seeds,
48-game arena cells, and 8/32/128-traversal teachers.

## Command

```bash
uv run experiments/runners/run_visit_teacher_iteration.py \
  --contract experiments/contracts/int-4-visit-teacher-iteration-v1.json \
  --profile smoke \
  --out-dir .runs/int-4-visit-teacher-smoke-post-pr132-v1

uv run experiments/runners/run_visit_teacher_iteration.py \
  --contract experiments/contracts/int-4-visit-teacher-iteration-v1.json \
  --profile smoke \
  --out-dir .runs/int-4-visit-teacher-smoke-post-pr132-v1 \
  --verify
```

Contract SHA-256:
`9c3f0f600b70ca4fef7131086f6d9b350f9612e690cbb9d621e961a5de27d03c`.
Runtime source SHA-256:
`14a3c5ff6594ad3c354abd06a9e888a3ad2c2d6e2d741db8954efa4f68c89dea`.
The complete checked-in receipt is
[`data/int-4-visit-teacher-smoke-v1.json`](data/int-4-visit-teacher-smoke-v1.json).

## Result

The end-to-end substrate passed. Four self-play games produced 507 legal
visit/value labels. A separate authority audit replayed all 175 decisions in
its source game and exactly reran the declared search root, including aggregate
and per-world visits, Q values, root values, action, nodes, depth, and cap hits.
There were zero learner/audit, frame, command, outcome, privacy, or search
mismatches.

All four seed-197 arms trained from initialization
`b3566276af0b2c14934c358bc4dd41d55e6f93bc1ba9d8ecf84e72c791b7ea6f`.
After the single smoke epoch, visit/value validation policy KL was 0.1432 and
root-value Brier was 0.0990. The complete teacher, policy-only student,
visit/value student, and student-guided PUCT round robin ran without illegal
actions or caps. The two-game results are deliberately not interpreted.

The selected historical decision produced Study artifact
`35e0949d2e1c325ca52768e2649fd4ca987990213259fcea4cbf36d3e6365e3a`.
Both `etude.StudyArtifact` and the Rust-owned
`managym::study::StudyArtifact::validate` accepted it. No deal seed, search
seed, opponent hand, or sampled hidden state entered the artifact.

## Teacher economics

| Traversals | Decisions | p50 | p95 | Labels/s | Traversals/s | Max depth | Cap hits |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 105 | 1.59 ms | 3.22 ms | 593.7 | 2,375.0 | 2 | 0 |
| 8 | 100 | 3.35 ms | 5.74 ms | 294.3 | 2,354.2 | 3 | 0 |

These same-host numbers are fresh cost evidence, but the host was not quiesced
and the cells were not isolated child-process RSS measurements. The production
profile must rerun them under its declared isolation and controls.

## Integrity finding fixed before the run

The viewer-equivalence conformance probe initially failed: determinization
shuffled a hidden pool in its current authority-private order. Two roots with
the same viewer information but different hidden allocations therefore
produced different labels under the same search seed. Determinization now
sorts hidden card IDs before seeded shuffling. Rust and Python conformance tests
prove that viewer-equivalent authorities sample the same worlds and produce
identical aggregate and per-world PUCT evidence.

## Verdict and next build

`revise` means “engineering smoke only; no admission decision,” not that the
student failed. Before the `iteration` profile can make its declared
continue/revise/kill decision, the runner still needs:

1. runnable paths for the frozen Teacher-0 policy-only and policy/value bytes;
2. same-host flat-MC and student-guided-search latency calibration;
3. the five competency cells and isolated RSS/core-hour receipts;
4. the full three-seed, 256-game production execution.

The strongest confound is scale: one initialization and two games per arena
cell make every win rate compatible with noise and seat/deal effects. Only the
integrity, determinism, artifact, and execution claims are admitted here.
