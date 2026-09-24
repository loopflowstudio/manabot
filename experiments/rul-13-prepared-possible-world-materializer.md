# RUL-13 prepared possible-world materializer

RUL-13 ships the Rules provider repair and its bounded likelihood consumer.
INT-17 calibration remains open. No calibration result or KR closure is claimed
by this provider slice.

## Verify

From the repository root:

```sh
./scripts/verify-prepared-possible-world-materializer
```

This rebuilds the release extension, runs the debug prepared-provider and focused
Python regressions, and verifies the checked receipt against the current source
and extension identities. The retained binary fingerprint belongs to the host
and toolchain recorded in the receipt; another build profile must produce its
own measurement before claiming byte identity. Portable tests validate source,
content, ABI, and evidence without equating binaries from different platforms.

To measure the provider through the release extension already built by that
script, write a separate receipt:

```sh
uv run --extra dev python experiments/runners/run_rul13_prepared_materializer.py \
  --out /tmp/rul-13-prepared-provider.json
uv run --extra dev python experiments/runners/run_rul13_prepared_materializer.py \
  --verify --out /tmp/rul-13-prepared-provider.json
```

The provider probe replays the frozen 132-Command live/headless/persisted trace
and materializes all 121,485 rows at ordinal 3. It does not run calibration
inference. The [checked receipt](data/rul-13-prepared-possible-world-materializer-v1.json)
records the source witness, canonical space identity, scalar parity witnesses,
source and extension fingerprints, host/toolchain, resource samples, and exact
construction/branch counters. Timing is evidence rather than a flaky test
threshold. The source-bound maximum batch is 256; the peak RSS limit is 2 GiB.

## Provider and consumer

`Env.prepare_possible_world_materializer(viewer, expected_space_identity,
max_batch_size)` enumerates and hashes one canonical Rules space. Its handle
keeps the same live root alive, checks source revision/viewer state and the
hidden pool once per request, and materializes retained indexes in caller order.
A changed source fails closed. Each returned Env is an isolated full clone.
The scalar provider remains the parity reference and shares the same row
primitive. Neither the row primitive nor the batch path enumerates support.

`FrozenPolicyLikelihood` prepares directly against the belief's existing space
identity, then encodes bounded native batches. It releases branches before
inference and releases inference buffers before calling its optional synchronous
progress observer. An observer exception prevents the next provider call.
The prepared Python handle rejects another batch while any preceding wrapper
or its `clone_env()` descendant remains live. Explicit caller clones can exceed
the returned batch count; diagnostics count those clones, and the consumer does
not create them. The bound applies per handle, not globally across independent
handles.

## Deferred calibration and preserved evidence

The existing INT-17 v2 working contract and six output files were generated
before the required preregistration commit. They are inadmissible historical
working evidence, not a scientific result. Their bytes and provenance were
preserved outside the publishable tree, together with the original runner/test
changes and a full dirty-tree patch, at:

`~/.local/share/etude/working-evidence/ETU-55/20260924T203758Z/`

The original-tree manifest SHA-256 is
`2f5f9b1753bdc70a5a7b2b419dbe3f6e5582707b8705f8c93ddf6d539cf5b61b`.
The excluded output retains artifact identity
`c6a6c9697fc5429128759cac352339f908e359614928be1583af632c79b3fbf5`.
The original files were checked against that manifest after separation. They
were neither regenerated nor published as calibration evidence. The v1 contract
and failure directory remain unchanged.

The explicit follow-up is to lock the shipped provider/runtime, use a distinct
execution identity with all ten frozen scientific sections unchanged, require
an actual preregistration commit before generation, restore batch-boundary
wall/CPU/RSS caps and ledger through the observer, then execute and verify the
unchanged 903,063-row one-worker experiment. Preserve the old v1/v2 records and
report the frozen prediction honestly. Hash-locking source is not a substitute
for preregistration.

Creating that scoped follow-up through `lf pm task create` was blocked by
Loopflow's repository PM migration requirement. The human-approved disposition
excludes that migration and abandoned PR #183, so no repository-wide PM changes
were made. This punch list remains actionable without those changes; ETU-55's
full calibration objective is not marked complete by the provider merge.
