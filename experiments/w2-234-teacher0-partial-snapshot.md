# W2-234 interim Teacher-0 prefix snapshot

> **Result on 2026-07-16:** both matched arms completed from the immutable
> prefix under the cap. This is a bounded recovery experiment, not the
> 3,000-game preregistration and not Teacher-1 evidence.

## Question and claim boundary

Can the existing matched policy-only and joint policy/value students train
cleanly from an immutable prefix of the live Teacher-0 dataset while the
canonical 3,000-game generator continues unchanged?

Every result from this slice is **partial interim flat-determinized-Monte-Carlo
Teacher-0 evidence**. Its policy target is the existing score softmax, not MCTS
root visits. It cannot complete the full preregistration, satisfy Search Project
KRs 2-5, authorize Teacher-1, or authorize later distillation.

## Frozen configuration

- Source: the first 64 complete durable shards (indexes 0-63) from
  `search-supervised-overnight-recovery-20260716`, exactly 512 of the declared
  3,000 games. The source remains read-only and continues generating.
- Snapshot: copied atomically into
  `.runs/w2-234-teacher0-partial-v1/snapshot`; every NPZ and JSON sidecar is
  bound by SHA-256 along with the parent run fingerprint, explicit cutoff,
  dataset source, and trainer source. Snapshot identity is
  `a8e6c20317a0b5a2908449def9d4630a6b2d2b4d1663e4b87ee186b19767f2a6`;
  parent run fingerprint is
  `f3122a6e5024124da195788bed0b0e4375a2479e6c6dde81d3c32c131b7531fb`.
- Student arms: `policy_only` (`value_weight=0`) and `policy_value`
  (`value_weight=1`). Both use score-softmax policy targets, terminal outcomes,
  the same snapshot, initialization seed 197, split, capacity, optimizer,
  learning rate 0.001, batch size 1024, 25 epochs, and MPS device.
- Evaluation: the fixed held-out game split produced by seed 197, 64
  seat-balanced games versus random per arm, and identical single/batched MPS
  inference measurements.
- Cap: 45 wall minutes total, checked before every epoch and arm. Exceeding it
  records `stopped_wall_cap`; it does not extend the run.

Predictions: both arms reduce held-out policy loss from their identical initial
state; the joint arm reaches held-out value Brier below 0.25 without losing
more than 10 points versus random relative to policy-only. Failure is diagnostic
only: policy failure points to optimization/data volume, value-only failure to
the outcome target or loss balance, and a cap failure to experiment economics.

## Commands

Freeze the declared prefix without writing to the source run:

```bash
uv run experiments/runners/run_teacher0_partial_snapshot.py \
  --stage freeze \
  --source-run /Users/jack/src/manabot.intelligence-overnight/.runs/search-supervised-overnight-recovery-20260716 \
  --snapshot-dir .runs/w2-234-teacher0-partial-v1/snapshot \
  --shard-count 64
```

Verify the copied snapshot:

```bash
uv run experiments/runners/run_teacher0_partial_snapshot.py \
  --stage verify \
  --snapshot-dir .runs/w2-234-teacher0-partial-v1/snapshot \
  --snapshot-identity a8e6c20317a0b5a2908449def9d4630a6b2d2b4d1663e4b87ee186b19767f2a6
```

Run the matched bounded arms:

```bash
uv run experiments/runners/run_teacher0_partial_snapshot.py \
  --stage train \
  --snapshot-dir .runs/w2-234-teacher0-partial-v1/snapshot \
  --snapshot-identity a8e6c20317a0b5a2908449def9d4630a6b2d2b4d1663e4b87ee186b19767f2a6 \
  --out-dir .runs/w2-234-teacher0-partial-v20/training \
  --epochs 25 \
  --batch-size 1024 \
  --lr 0.001 \
  --val-fraction 0.1 \
  --seed 197 \
  --policy-temperature 0.05 \
  --device mps \
  --evaluation-games 64 \
  --wall-cap-minutes 45
```

The earlier `.runs/w2-234-teacher0-partial-v1/training` attempt was interrupted
with `BrokenPipeError` before either arm produced a receipt. It is preserved as
failed, excluded from results, and was not overwritten.

## Result

The immutable prefix contains 64 contiguous complete shards, 512 games, and
61,897 decisions. All 64 NPZ/JSON pairs verified against the snapshot before
training; there were zero invalid teacher actions and zero winnerless rows.
The snapshot manifest binds every copied and source path, byte count, shard
identity, and SHA-256 digest. The complete manifest, training history, runtime
identity, and receipts are checked in at
`experiments/data/w2-234-teacher0-partial-v20.json`.

Both arms began from identical held-out metrics: policy loss 1.00430, policy
KL 0.12282, action agreement 0.4090, and value Brier 0.25516. Only
`value_weight` differed.

| metric | policy only | policy + value |
| --- | ---: | ---: |
| final policy loss | 0.97248 | 0.97627 |
| final policy KL | 0.09101 | 0.09480 |
| action agreement | 0.5440 | 0.5229 |
| final value Brier | 0.25262 | **0.16684** |
| value accuracy | 0.4561 | **0.7529** |
| win rate vs random, 64 seat-balanced games | 0.8594 [0.7538, 0.9242] | 0.8906 [0.7910, 0.9460] |
| MPS p50 / p95 single-observation latency | 1.464 / 2.305 ms | 2.307 / 3.938 ms |
| MPS batch-256 throughput | 19,454 obs/s | 19,587 obs/s |
| arm wall time | 257.6 s | 259.8 s |

Total wall time was 517.6 seconds (8.63 minutes), below the 45-minute cap.
Both arms reduced held-out policy loss. The joint arm cleared the registered
value-Brier threshold and did not lose strength to the policy-only arm in this
small random-control evaluation. These are one-seed interim diagnostics, not a
method-level strength claim.

Content-addressed checkpoint receipts:

- policy only:
  `eef14a4b4d4c275d2989854a6696d128abd5ea3919001b3796d6b8cf9318949c`
  (429,351 bytes)
- policy + value:
  `d95d5af6ba9fb529f0ebe134f2ba6eac880479108f5ddbefe873b59fbf4d3bbd`
  (429,401 bytes)

The canonical 3,000-game generator remained live and unchanged throughout.
This result does not satisfy the full Teacher-0 preregistration, does not use
MCTS visit counts, does not satisfy Search Project KRs 2-5, and does not unlock
Teacher-1 admission or later distillation.
