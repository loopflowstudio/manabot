# MTG-249: GPU representation experiment — interim report

Snapshot: 2026-11-05 19:56 UTC. This is a descriptive progress report, not the final architecture comparison.

Seven of eight learners have completed the planned 256 updates / 65,536 games each. Three learners have complete, hash-verified evaluation sets. The last learner (GRU seed 74214, Run `ce70ec9470e1`) reports running in a fresh Etude status read. Four completed recovery learners await full evaluation; the last learner will also need evaluation.

The task's existing implementation Graph is alive. Its evaluation dispatcher is retrying capacity acquisition: the most recent log at 19:54 UTC says no matching Vast machine is available. Evaluation uses CPU on a separate rental whose GPU is unused. This is a capacity delay, not a completed study.

## What we are comparing

Both arms use the same v7 observations and object-attention model. Pooling versus GRU changes how effect-payload tokens and continuation-frame tokens are reduced into vectors **before object attention**. Neither arm has memory across decisions. This is distinct from MTG-252's post-attention value-token versus masked-mean critic readout.

Four paired training seeds (74211–74214), two arms, 65,536 self-play games per learner. The planned complete cohort has 524,288 training games. Seven finished learners account for 458,752 completed games, excluding the final learner and preserved failed attempts.

Training and evaluation use the fixed w17 Allies/Lessons matchup. Curve evaluation uses deals 85001–85032, both deck assignments and both seats: 128 games/checkpoint. Final held-out evaluation uses separate deals 86001–86064: 256 games/learner. The opponent is the frozen Rule-of-thumb heuristic. These are held-out deals, not held-out decks.

## Completed held-out evaluations

Scores are wins plus half draws. Intervals are the reports' within-seed 95% paired-deal uncertainty; they do not measure uncertainty across independent training seeds.

| Learner | Final held-out score | 95% interval | Games |
|---|---:|---:|---:|
| pool / 74211 | 47.66% | 43.75–51.56% | 256 |
| gru / 74211 | 42.58% | 37.50–47.66% | 256 |
| gru / 74212 | 48.44% | 42.97–53.52% | 256 |

For the only fully evaluated pair (74211), GRU minus pool is **−5.08 percentage points**. This is one seed pair; it cannot establish an architecture winner or equivalence. The unpaired GRU seed is shown transparently, not averaged into a treatment comparison.

## Learning curves

Each cell is score percent on the same predeclared curve deal set. The held-out table above uses a separate deal set.

| Update | Pool 74211 | GRU 74211 | GRU 74212 |
|---:|---:|---:|---:|
| 0 | 7.81 | 7.81 | 7.81 |
| 32 | 35.16 | 28.91 | 30.47 |
| 64 | 41.41 | 37.50 | 37.50 |
| 96 | 43.75 | 39.84 | 41.41 |
| 128 | 43.75 | 40.62 | 38.28 |
| 160 | 41.41 | 40.62 | 39.06 |
| 192 | 45.31 | 46.88 | 43.75 |
| 224 | 45.31 | 42.19 | 45.31 |
| 256 | 51.56 | 40.62 | 51.56 |

All three began at 7.81% and improved substantially. Their final curve scores are 40.63–51.56%; held-out scores are 42.58–48.44%. They have learned useful behavior, but have not demonstrated reliably stronger play than the heuristic.

**Plateau is not established.** Pool 74211 and GRU 74212 still rise at the final curve point. More fundamentally, the predeclared criterion requires all four paired seeds: each arm's final three interval gains must have a cross-seed 95% interval entirely within ±5 points. The full curves and cohort are needed to apply it. No interim score changes the horizon, selects a checkpoint, or decides a retry.

Across these three complete evaluation sets, 4,224/4,224 games replayed successfully with zero recorded game failures in the completed reports. Earlier failed/timed-out attempts remain in the attempt history; this does not claim all attempts succeeded.

## Recovery and remaining work

The original rental was deleted by our launcher's response to one unhealthy status read. Three completed learners survived in verified backups. Recovery retains those learners, restarts pool/74212 from initialization, and trains the remaining four. It uses a prospectively frozen amendment and repaired launcher control client; the worker source remains pinned.

Pool/74212's first evaluation attempt was cancelled while its initialization mismatch was checked. The mismatch was explained as small host QR rounding differences (at most 1.04e-6 in 18 matrices, with identical RNG streams and other tensors), and scoring was subsequently admitted. A delivery size mismatch remains recorded for that cancelled attempt; its partial evaluation is omitted from the completed-result tables here. Its remaining endpoints are queued.

Remaining: deliver the eighth learner; finish all five outstanding evaluation sets; verify receipts and replay integrity; compute four-pair held-out and plateau statistics; present the final report. Completion of the planned horizon alone does not guarantee the plateau objective.

## Budget snapshot

Original cumulative ceiling: **$25**. Prior accounting/reserves before recovery: **$12.1395**, including $6 ancillary reserve. Recovery ended-rental accounting in the dispatcher: **$4.3572**. The remaining live learner's fresh Etude estimate is **$1.5572**, already above its older $1.4844 dispatch cap; that cap is not treated as a billing guarantee. These figures total about **$18.05 accounted/reserved**, leaving about **$6.95 before further live cost and pending evaluations**. These are operational estimates/reserves, not settled invoices. The active operator needs to reconcile the live estimate and deadline before further commitments.

## Evidence

- [Frozen GPU protocol and recovery amendments](../experiments/protocols/state-representation-5da5f4-gpu.md)
- [Execution plan and recovery history](representation-learning-plan.md)
- [Earlier CPU-only report](../experiments/state-representation-v7.md): a separate shorter study; do not substitute its conclusion for this GPU run.
- Local evaluation evidence: `.runs/representation/recovery/ledger/{learner}/{endpoint}/attempt-*/monitor.json`.
- Hash-verified delivery receipts: `.runs/representation/recovery/deliveries/{run}/etude-receipt.json`.
- Operations: `.runs/representation/recovery/drive-state.json` and `drive-5.log`.

This report reads retained results and status only. It launches no work and changes no scientific selection or stopping rule.
