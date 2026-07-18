# INT-7: Value Targets Inside the Visit-Trained Player

## Result

**Decision: `continue_visit_policy_only`.** On this one retained 507-row
teacher/data seed, the visit-policy-only control was the strongest complete
32-traversal PUCT player. Terminal-outcome, Teacher-1-root-value, and fixed
50/50-blend supervision all improved held-out value calibration or target
agreement, but each produced a weaker complete player in the registered smoke
matrix.

This is `engineering_smoke_only_no_admission_claim`. It is not a method,
rating, strength, promotion, or admission claim. The three training seeds
measure initialization sensitivity over one corpus; they are not independent
teacher/data seeds.

## Registered execution

The frozen contract was committed as `076e18c` before any training or
experiment generation. Its SHA-256 is
`01479b88619c39e6141445aa37561bdf718dfc1267bd860c2bac43fb398cb4ef`.
It binds the exact INT-8 retained input, the unchanged INT-6 contract, all four
training arms, seeds, compute, schedule, caps, and decision rule.

Run:

```bash
uv run python experiments/runners/run_int7_value_target_comparison.py \
  --input-manifest \
    experiments/data/int-8-retained-int-4-smoke-v1/sha256/13868767846b7004f140cfade3652909347bdbb6708b69cb8c10b36ec2756eb0/input-manifest.json \
  --out-dir .runs/int-7-value-target-comparison-v1/result
```

Verify without generating evidence:

```bash
uv run python experiments/runners/run_int7_value_target_comparison.py \
  --out-dir .runs/int-7-value-target-comparison-v1/result \
  --verify-only
```

The run trained twelve matched checkpoints, completed all 136 cells and 544
seat-paired games in the 17-player cohort, and profiled 128 matched roots per
player. Verification replayed all 81,449 decisions and reported
`no_generation=true`.

## Complete-player comparison

The primary comparison is the mean diagnostic rating across the three
seed-specific players. Anchor paired score uses the common five frozen INT-6
code-only anchors. Ratings are local diagnostics on this connected matrix and
do not extend the frozen INT-6 scale.

| Value target | Mean diagnostic rating | Anchor paired score | Within-seed head-to-head | S1-S5 correct / 30 | Mean isolated p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `visit_policy_only` | **1361.172** | **0.533** | **0.806** | 0 | 62.62 ms |
| `visit_teacher_root` | 1039.921 | 0.233 | 0.528 | 0 | **58.37 ms** |
| `visit_terminal` | 1032.013 | 0.250 | 0.361 | 1 | 58.45 ms |
| `visit_blend_50_50` | 930.997 | 0.117 | 0.306 | 1 | 60.59 ms |

Policy-only's minimum paired-score separation from another method was 0.283,
above the preregistered 0.05 continuation threshold. All methods were
competency-noninferior to the control under the registered aggregate rule,
although the absolute competency result was poor: no method exceeded 1/30.
Together with perfect integrity, this yields `continue_visit_policy_only` for
the next corpus. It does not make this checkpoint promotion-eligible.

Every joint arm had negative same-seed rating and paired-score uplift per node
and per CPU second relative to its policy-only control. The unnormalized
ratings, payoff cells, and measured cost remain primary; the normalized ratios
are matrix-specific diagnostics.

## Calibration and mechanism

Held-out terminal-outcome Brier was 0.251-0.253 for the policy-only seeds. The
joint arms improved it to 0.225-0.235 for terminal supervision, 0.231-0.239
for Teacher-1-root supervision, and 0.227-0.238 for the blend. Teacher-root
agreement similarly improved from 0.100-0.103 to 0.093-0.099. These
improvements did not translate into complete-player strength, which is why
calibration did not select the winner.

The fixed held-out game contained 111 rows: 6 beginning, 59 precombat main,
38 combat, 8 postcombat main, and no ending rows. Sparse phase cells are
retained as `insufficient_n`; Teacher-1-root and blend reliability are labeled
target-source agreement rather than calibration to ground truth.

The matched-root ablation shows two separate effects:

- Shared-encoder policy drift was small: control-to-joint prior KL was
  0.00025-0.00072, and neutral-value action agreement with the same-seed
  policy-only checkpoint was 0.906-0.984.
- Using learned rather than neutral value changed actions materially: action
  agreement was 0.578-0.656 and mean absolute root-value change was
  0.012-0.053.

Both modes perform one checkpoint forward for each expanded nonterminal node,
with the same 32 traversals and four worlds. Their total forward counts can
differ because learned values change the traversed tree and which leaves are
terminal; equal traversal/evaluator semantics, not equal realized tree paths,
are the matched-compute contract.

## Integrity, cost, and identities

All 544 game rows had zero illegal action, private exposure, offer-binding,
Command-fabrication, action/card/permanent truncation, or replay failure
counters. Exact Command-trace replay, row replay, input verification before
and after the run, and all 128-root preservation checks passed.

The cumulative ledger finished at 0.672 wall hours, 2.387 core hours, four
workers, and 26,273,550 artifact bytes, within the frozen caps of six wall
hours, 24 core hours, four workers, and 2 GiB. Marginal training used 47,520
examples in 63.665 seconds. Evaluation used 2,417.404 wall seconds, including
2,059.364 seconds for the arena and 245.782 seconds for isolated profiles.
The result also carries forward all 507 inherited label rows and the retained
INT-4 teacher/search cost instead of treating reused labels as free.

- Retained INT-8 payload:
  `13868767846b7004f140cfade3652909347bdbb6708b69cb8c10b36ec2756eb0`
- Retained trajectory/search audit:
  `ae03c3bda06bdd65b090fefcaf1e23bb717c6f6566cc08731092c7911770f14f`
- Frozen INT-6 contract:
  `fc9cb76c0d80ad64951455ac6fede94b1355f383dde2f2964d0e578f62671a71`
- INT-7 result manifest:
  `3f7a00e179fe49fe111ff8a361501ca8080e501fa7189abeaacb66194a5de5bf`

The immutable result is retained under
[`data/int-7-value-target-comparison-v1/sha256/3f7a00e179fe49fe111ff8a361501ca8080e501fa7189abeaacb66194a5de5bf/`](data/int-7-value-target-comparison-v1/sha256/3f7a00e179fe49fe111ff8a361501ca8080e501fa7189abeaacb66194a5de5bf/).
Its principal evidence includes the
[`decision`](data/int-7-value-target-comparison-v1/sha256/3f7a00e179fe49fe111ff8a361501ca8080e501fa7189abeaacb66194a5de5bf/result/decision.json),
[`full payoff matrix`](data/int-7-value-target-comparison-v1/sha256/3f7a00e179fe49fe111ff8a361501ca8080e501fa7189abeaacb66194a5de5bf/result/payoff-matrix.json),
[`calibration`](data/int-7-value-target-comparison-v1/sha256/3f7a00e179fe49fe111ff8a361501ca8080e501fa7189abeaacb66194a5de5bf/result/calibration.json),
[`matched-root profiles`](data/int-7-value-target-comparison-v1/sha256/3f7a00e179fe49fe111ff8a361501ca8080e501fa7189abeaacb66194a5de5bf/result/profile.json),
[`mechanism ablations`](data/int-7-value-target-comparison-v1/sha256/3f7a00e179fe49fe111ff8a361501ca8080e501fa7189abeaacb66194a5de5bf/result/mechanism.json),
and [`resource ledger`](data/int-7-value-target-comparison-v1/sha256/3f7a00e179fe49fe111ff8a361501ca8080e501fa7189abeaacb66194a5de5bf/result/resource-ledger.jsonl).

## Conclusion

The value heads learned visibly better scalar targets, and those learned
values changed PUCT choices, but none improved the complete player on this
retained corpus. Continue the visit-policy-only recipe when a larger,
independently seeded visit corpus becomes available. Do not extend this smoke
result into a general rejection of value learning: the data contains one
teacher seed, the teacher root target is a quantized eight-traversal estimate,
and the competency floor is nearly zero for every arm.
