# W2-234 Teacher-1 admission pilot

> **Status on 2026-07-16: pre-registered; not run.** The live Teacher-0
> recovery is not restarted, mutated, or counted as Teacher-1 evidence. This
> experiment cannot start until its separately checked-in control lock binds
> the terminal recovery manifest, frozen checkpoint, and same-host latency
> calibration.

## Question

Does the current deterministic multi-world PUCT Teacher-1 produce strong,
stable, legal, viewer-safe, replayable, and affordable root visit/value targets
at budgets 8, 32, and 128, such that a three-seed distillation comparison is
warranted?

This is the smallest experiment not already owned by the resumable Teacher-0
recovery. It evaluates the Teacher-1 substrate landed in PR #106 and uses the
crash-safe Teacher-0 result from PR #108 only as a frozen control after that run
reaches a terminal state. It generates no student shards or checkpoints.

## Frozen contract

The machine-readable source of truth is
`experiments/contracts/w2-234-teacher1-pilot-v1.json`. It pins:

- world `w2`, complete observation shapes, action/observation/protocol hashes,
  engine source and binary hashes, content digest, and the symmetric
  `INTERACTIVE_DECK` matchup;
- Teacher-1 budgets 8/32/128 total traversals, four worlds, `c_puct=1.5`,
  uniform priors, random terminal leaves, and a 2,000-step cap;
- 48 seat-balanced games per matchup cell, 100 runs per competency cell, 192
  stability roots searched three times, and four replay-audit games;
- separate `random`, `scripted`, `search`, and `checkpoint` opponent classes;
- the Apple M4 Max host, four-worker limit, 8 wall-hour / 32 core-hour gate,
  2 GiB artifact cap, seeds, predictions, hard gates, and next branches;
- future paired training seeds 197, 419, and 887. No training is authorized by
  this slice.

The Teacher-0 control is matched by measured p50 decision latency, not by the
ambiguous `sims` spelling: Teacher-0 counts playouts per legal action while
Teacher-1 counts total tree traversals across worlds. Each mapping must be
within 10% on the same host and is immutable once its control lock is checked
in. The checkpoint control is the recovery's `policy_value` arm, selected
before its result is read and played by deterministic argmax; the lock rejects
any substituted arm.

## Pre-registered predictions and gates

The high-budget teacher must pass every integrity and quality condition:

1. zero illegal commands, legal-mask/offer mismatches, non-finite or
   out-of-range targets, authoritative-root mutations, private-hand leaks, or
   replay frame/command/outcome mismatches;
2. at least 55% against `t1-8-w4`, at least 55% against the frozen policy-only
   checkpoint, and at least 80% against random;
3. at least 70% repeated-search top-action agreement and at most 0.10 median
   Jensen-Shannon divergence at 128 traversals;
4. strictly increasing mean node count and mean maximum depth from 8 to 128;
5. at least 20% correct on one of S1, S2, or S5 and no S1-S5 result more than
   10 points below random;
6. at most 500 ms p95, at least two labels/second/worker, below 0.1%
   playout-cap hits, and a realized p50 gap within 10% for every matched-wall
   Teacher-0 cell.

Root-value Brier score, ten-bin ECE, and reliability bins against the played
terminal outcome are reported as diagnostics, not hard gates. The root value
estimates random-leaf continuation; terminal play follows the teacher, so
calling the former a calibrated outcome target would overstate the evidence.

If every gate passes, the next serial experiment freezes `t1-128-w4` and runs
the matched chosen-action versus visit-distribution comparison at all three
training seeds. Any failure takes the named branch in the contract; no result
authorizes post-hoc target temperature, pruning, capacity, or optimizer tuning.

## Commands

Inspect the frozen runtime before the control lock exists:

```bash
uv run experiments/runners/run_teacher1_pilot.py --print-runtime
```

After the terminal recovery and calibration artifacts have been hashed into
`experiments/contracts/w2-234-teacher1-control-lock-v1.json`, run or resume the
admission gate:

```bash
uv run experiments/runners/run_teacher1_pilot.py \
  --contract experiments/contracts/w2-234-teacher1-pilot-v1.json \
  --control-lock experiments/contracts/w2-234-teacher1-control-lock-v1.json \
  --stage teacher-gate \
  --out-dir .runs/w2-234-teacher1-pilot-v1
```

Independently replay the authoritative audit from a completed run:

```bash
uv run experiments/runners/run_teacher1_pilot.py \
  --contract experiments/contracts/w2-234-teacher1-pilot-v1.json \
  --verify .runs/w2-234-teacher1-pilot-v1
```

Until the control lock exists and validates, `--stage teacher-gate` fails
closed before creating a run directory or doing search work. This is deliberate:
the live recovery result may supply a control, but it cannot influence these
predictions or gates.
