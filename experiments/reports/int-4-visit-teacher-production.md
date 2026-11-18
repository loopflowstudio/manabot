# INT-4 Production Visit-Teacher Iteration

> **Status on 2026-07-17: fail-closed harness ready; production not run.**

The production-only runner and additive contract now bind the landed INT-4
engineering substrate to the registered 256-game dataset, four student arms at
seeds 197/419/887, 48-game arena cells, exact frozen Teacher-0 controls,
same-host flat-MC calibration, S1-S5 competencies, resource receipts, replay,
Study validation, and an explicit admission decision.

Every completed or failed attempt now appends a hash-chained resource receipt.
The runner reconstructs cumulative wall, core, and artifact totals across
failures and resumes and checks remaining capacity before every stage and
spawned child. Job and stage results live at attempt-specific immutable paths;
their exact SHA-256 values bind the stage and manifest copies used by the
teacher and admission gates.

This is durable implementation progress, not production evidence. Neither
registered Teacher-0 checkpoint is present in the repository `.runs` cache:

- `policy_only` — `3bfedccf5aa6ed7621d99284ea8cea3975d8b195cecf6426d37dd7abc812c978`
- `policy_value` — `92ced7abb31bc68298b48cc08ed7eb57f3dde50295a22d50ea2fe32f7e359176`

The command rejects a missing path, wrong hash, wrong arm, incompatible model,
or runtime/source identity before it creates a production run directory. The
controls must not be retrained, reconstructed, ported, or substituted.

## Run and verify

```bash
uv run experiments/runners/run_visit_teacher_production.py \
  --contract experiments/contracts/int-4-visit-teacher-iteration-v1.json \
  --production-contract experiments/contracts/int-4-visit-teacher-production-v1.json \
  --policy-only-control /absolute/path/to/frozen-policy-only.pt \
  --policy-value-control /absolute/path/to/frozen-policy-value.pt \
  --out-dir .runs/int-4-visit-teacher-production-v1

uv run experiments/runners/run_visit_teacher_production.py \
  --contract experiments/contracts/int-4-visit-teacher-iteration-v1.json \
  --production-contract experiments/contracts/int-4-visit-teacher-production-v1.json \
  --policy-only-control /absolute/path/to/frozen-policy-only.pt \
  --policy-value-control /absolute/path/to/frozen-policy-value.pt \
  --out-dir .runs/int-4-visit-teacher-production-v1 \
  --verify
```

The independent verifier performs no training or evaluation. It rehashes all
contracts, controls, source/runtime identities, resource-ledger receipts,
exact job and stage result bytes, shards, checkpoints, and Study evidence;
proves the manifest copies match those bytes; exactly replays the sampled
trajectories and searches; and recomputes the teacher and admission gates from
the bound stage results.

## Frozen identities and verification receipts

- Base iteration contract: `9c3f0f600b70ca4fef7131086f6d9b350f9612e690cbb9d621e961a5de27d03c`
- Production source: `30f0db1800cef320c1ef02265f18ee93e8450e0eb66647b4e6374f2371d74dc1`
- Unchanged PR #133 smoke source: `14a3c5ff6594ad3c354abd06a9e888a3ad2c2d6e2d741db8954efa4f68c89dea`
- Unchanged PR #133 Study artifact: `35e0949d2e1c325ca52768e2649fd4ca987990213259fcea4cbf36d3e6365e3a`

Fresh verification on 2026-07-17:

- 60 focused Python tests passed through `uv`, including cumulative-cap,
  failed-resume, exact-result tamper, manifest-copy tamper, ledger-prefix, and
  verify-no-generation regressions;
- the complete debug `managym` Cargo suite passed, including 12/12 search
  state contract tests and 11/11 search tests;
- PR #133's smoke independently replayed all 175 decisions and its sampled
  search root with zero mismatches or private-card exposures, then revalidated
  the same Study artifact in Python and Rust;
- a missing-control preflight left no production run directory.

INT-4 remains open. The next admissible evidence begins only when both exact
control files are available and the registered production command completes
within its 16 wall-hour, 64 core-hour, four-worker, and 4 GiB caps.
