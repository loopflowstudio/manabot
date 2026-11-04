# Pooled versus GRU state reduction on v7

2026-11-04 · Task 5da5f4 · completed prospective comparison

**GRU did not demonstrate an advantage.** Across four paired training seeds, its
score against the heuristic was **0.98 percentage points lower** than pooling;
the 95% cross-seed interval is **−10.13 to +8.18 points**. Training took **1.06×**
as long. The predeclared criterion failed. No model or default was promoted.

All eight replicates completed 128 updates and 4,096 games each: **32,768 training
games and 4,056,651 learner decisions**. Every one of the **1,024 scheduled
evaluation games** completed and replayed successfully. There were no retries,
missing seeds, post-hoc checkpoint selections, illegal actions, private exposures or other
recorded integrity violations. All 1,024 training updates recorded zero nonfinite
values, legal-probability underflows and rejected optimizer steps.

## Scores and uncertainty

Each score uses 128 held-out games against the frozen heuristic, exchanging both
authored deck assignments and both seats. Bracketed intervals describe paired-deal
evaluation noise within one seed; they are not independent training replications.

| Training seed | Pool score, % [95% interval] | GRU score, % [95% interval] | GRU − pool, points |
|---|---:|---:|---:|
| 73201 | 20.31 [14.06, 26.58] | 23.44 [16.41, 30.47] | +3.12 |
| 73202 | 20.31 [14.06, 26.56] | 13.28 [8.59, 18.75] | -7.03 |
| 73203 | 24.22 [18.75, 30.47] | 19.53 [14.06, 25.78] | -4.69 |
| 73204 | 21.88 [15.62, 28.91] | 26.56 [20.31, 32.03] | +4.69 |

Mean scores were 21.68% for pooling and 20.70% for GRU. Both remain weak
against this heuristic at the measured horizon. There is no initialization or
no-update control, so this study does not establish improvement over initialization.

The primary interval uses the four paired seed differences, Student-t with three
degrees of freedom. It is conditional on this fixed deal schedule and assumes
approximately normal differences; four seeds provide weak power. The result
does not establish equivalence or superiority of pooling. The criterion required
a positive lower bound, at least +3 points, and a training-time ratio ≤1.25.

## What changed and what was held fixed

Both arms received the same complete v7 observation, including shared effects,
explicit memberships, local facts and legal actions. The comparison changes both
the effect-payload reducer and the typed continuation-frame reducer: sum pooling
versus a GRU over tokens within each record. Local fact encoding, object attention,
action focus, learner, curriculum, horizon and model width/depth are shared.
Neither arm has memory across decisions. This is not an isolated effect-reducer
ablation, a comparison with the production `rows` continuation, or evidence for
a history/memory architecture.

The strongest confounds are the joint treatment and extra GRU parameters, only
four seeds, a fixed short horizon and one authored Allies/Lessons matchup. Held-out
deals do not establish held-out-deck generalization. Architecture selection beyond
this experiment needs another accepted comparison; these results do not justify
promoting GRU.

## Execution and cost

The source was committed before scientific execution as
`92becff6091912a3ec83ae8fdd4e1c2347ff1d45`. Native world/content/input identities
are retained in every TrainingRun and checked again by evaluation. The frozen
[protocol](protocols/state-representation-5da5f4-v7.md) records the CPU fallback:
L40/L40S were unavailable and RunPod refused the quoted 4090 allocation before
acquiring a machine. The old v5 canaries and the new one-update CPU admission
are excluded from the comparison.

Training used two fixed four-process CPU batches, one thread and a 16 GiB
watchdog per replicate. Evaluation followed training in two groups of four
endpoints, each retaining its declared two-active-game/batch-16 envelope.
Mean replicate training times were 26.50 minutes (pool) and
28.09 minutes (GRU). Concurrent real training and unequal trajectory
lengths limit interpretation of this elapsed-time ratio as isolated throughput.

The earlier deleted L4 canary has a provider bill of **$0.07258259388618171**,
superseding its $0.07083924 estimate. This continuation acquired no rental and
used no MTG-213 allocation. Artifact/storage/transfer charges are not yet invoiced;
the **$6 ancillary reserve** remains within the **$25 total ceiling**. Local CPU
use has no incremental provider charge. Exact-name provider checks confirm the
old pod and refused allocation are absent. Etude retains a refused-session
metadata row without a provider receipt; it is not a live machine.

## Receipts and reproduction

The retained data directory is
`experiments/data/state-representation-v7-5da5f4/`. Its `cohort/comparison.json`
contains the recomputed contrast, per-seed intervals, timing, decisions, recipe
digests and checkpoint hashes. TrainingRun records, raw/EMA/optimizer exports,
sealed Etude metrics, every evaluation row and replay trace, admission failures,
billing/cleanup receipts and the exact frozen source archive are included.
Bytes are stored by digest through `experiments/data.manifest.json`.

The maintained runner is now
[state_representation_v7.py](state_representation_v7.py), matching the unchanged
v7 tracking namespace. Its previous filename at the scientific source commit
remains in the frozen source archive. This post-run move and report declaration
do not alter the experiment inputs or retained receipts.

The scientific recipes pin their executed 64-action capacity so retained results
remain admissible after fresh defaults change. Re-execution requires the archived
source and native runtime. The unscored engineering smoke uses the current native
action capacity (512); it checks learner integration and is not a replication.

```sh
uv run --extra artifacts python -m manabot.data fetch experiments/data/state-representation-v7-5da5f4
uv run --extra reports etude report experiments/state_representation_v7.py --source experiments/data/state-representation-v7-5da5f4/cohort --out .runs/representation-report
uv run python - experiments/data/state-representation-v7-5da5f4/cohort < experiments/data/state-representation-v7-5da5f4/analyze.py
```

The Etude page presents ordinary training diagnostics; the seed-level comparison
above is the scientific analysis. No visual rendering or browser review is claimed.
