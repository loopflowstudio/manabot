# INT-8: Which retained student signal should guide bounded PUCT?

**Result (2026-07-18, w2): `kill_retained_smoke_policy_guidance`.** Neither
the chosen-action nor visit-distribution policy-only checkpoint earned a new
corpus as a prior for 32-traversal PUCT. Both lost paired score against the
uniform-prior control, neither improved agreement with uniform-128 by the
registered 0.05, both added inference cost, and neither changed the S1-S5
correct count. This is
`engineering_smoke_only_no_admission_claim`: one 507-label training seed and
two smoke deals cannot support admission, promotion, gameplay-strength, or a
general claim about chosen versus visit supervision.

## Question and preregistration

Does either retained INT-4 policy-only student improve bounded PUCT when leaf
evaluation, root noise, world sampling, seeds, matchup, and traversal budgets
are held fixed? The committed prediction was that visit supervision would
better reproduce the source teacher distribution, while neither smoke student
would clear the joint arena-and-mechanism gates. The second clause held; the
first did not.

The experiment used the frozen INT-6 contract byte-for-byte at SHA-256
`fc9cb76c0d80ad64951455ac6fede94b1355f383dde2f2964d0e578f62671a71`.
The additive INT-8 contract bound the current extension/runtime and arena
source, the new retained-input and prior-guidance modules, the three candidate
identities, and the unchanged INT-6 key, cohort, rating prior, gates, smoke
seed plan, and scale. Checkpoint logits supplied legal-action priors only;
checkpoint values were discarded and every leaf used the same random terminal
evaluator.

## Arena result

The verified cohort contains all 28 eight-player cells: 10 anchor-anchor, 15
candidate-anchor, and 3 candidate-candidate cells, with two deals and both
seat legs per cell (112 games). The complete matrix, per-seat rows, residuals,
and diagnostic ratings are retained in
[`payoff-matrix.json`](data/int-8-student-signal-guidance-v1/sha256/64779c48ab1f3c9d54fbd697a373666ada7333840058cab88494e9ccf129b38e/diagnostic/payoff-matrix.json)
and
[`rating.json`](data/int-8-student-signal-guidance-v1/sha256/64779c48ab1f3c9d54fbd697a373666ada7333840058cab88494e9ccf129b38e/diagnostic/rating.json).

| 32-traversal prior | paired score delta vs uniform | diagnostic rating (global deal-block bootstrap range) | S1-S5 correct |
| --- | ---: | ---: | ---: |
| uniform | control | 1515 (1454–1648) | 2/10 |
| chosen policy-only | -0.15 | 1340 (1293–1429) | 2/10 |
| visit policy-only | -0.10 | 1370 (1355–1401) | 2/10 |

Candidate-only cells were uniform 3–1 over chosen, uniform 3–1 over visit,
and chosen 2–2 with visit. Each arm was correct once on S1 and once on S4,
and zero times on S2, S3, and S5. These smoke ratings describe this connected
diagnostic matrix only and are not promotion ratings.

## Mechanism and cost

The matched-root corpus contains 128 identical roots for all nine
prior/budget variants. Entropy and shift are computed per root and retained by
action-space kind and legal-action count in
[`mechanism.json`](data/int-8-student-signal-guidance-v1/sha256/64779c48ab1f3c9d54fbd697a373666ada7333840058cab88494e9ccf129b38e/diagnostic/mechanism.json).

| prior | traversals | p50 / p95 ms | nodes/s | decisions/s | CPU s/label | nodes/label | normalized visit entropy | prior→visit L1 / JSD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform | 8 | 9.1 / 16.4 | 1376 | 114.7 | 0.0087 | 12.0 | 0.648 | 0.583 / 0.143 |
| chosen | 8 | 17.5 / 24.2 | 705 | 58.8 | 0.0170 | 12.0 | 0.648 | 0.586 / 0.143 |
| visit | 8 | 16.8 / 27.1 | 714 | 59.5 | 0.0168 | 12.0 | 0.646 | 0.589 / 0.144 |
| uniform | 32 | 38.4 / 73.5 | 916 | 25.8 | 0.0388 | 35.5 | 0.901 | 0.302 / 0.045 |
| chosen | 32 | 61.9 / 89.3 | 590 | 16.6 | 0.0603 | 35.5 | 0.904 | 0.292 / 0.045 |
| visit | 32 | 60.5 / 491.5 | 337 | 9.5 | 0.1054 | 35.5 | 0.902 | 0.296 / 0.045 |
| uniform | 128 | 132.0 / 264.3 | 930 | 7.3 | 0.1371 | 127.5 | 0.983 | 0.141 / 0.005 |
| chosen | 128 | 239.1 / 360.1 | 536 | 4.2 | 0.2380 | 127.5 | 0.982 | 0.142 / 0.005 |
| visit | 128 | 262.0 / 376.2 | 500 | 3.9 | 0.2550 | 127.5 | 0.981 | 0.146 / 0.006 |

Uniform-32 agreed with uniform-128 on 86/128 roots (0.672). Chosen-32 reached
89/128 (0.695, +0.023) and visit-32 reached 87/128 (0.680, +0.008), both below
the registered +0.05 gate. Chosen and visit selected the same Command on
122/128 roots; each agreed with uniform-32 on 110/128. The learned priors
therefore changed some root choices but supplied little high-budget signal at
this scale.

The visit-32 p95 includes a local long-tail event: its p50 was comparable to
chosen-32, while p95 rose to 491.5 ms. This makes the precise visit throughput
ratio vulnerable to systems noise. It does not change the decision: both
learned arms independently failed the arena and label-agreement gates, and
chosen also failed the cost gates without that anomaly.

## Integrity, cap, and conclusion

All 18 new Command traces replayed exactly: 8,211 decisions across 72 games,
with zero actor, frame, offer, Command, state, outcome, trace, or private-
exposure mismatches. Match rows, roots, legal actions, playout caps, and
resource gates passed. The combined retained artifact used 3.9 MB at the cap
decision, 0.120 wall hours, and a conservative 0.481 core hours, below the
registered 1 GiB, 2 wall-hour, and 8 core-hour caps.

The mechanism prediction favoring visit targets was refuted. The useful
engineering decision is narrower: do not scale either recovered one-seed
smoke checkpoint as PUCT guidance. A later chosen-versus-visit comparison
requires a new multi-seed corpus and checkpoints; it must not treat this kill
as evidence against PUCT or either supervision method in general.

## Reproduce and verify

```bash
uv run python experiments/runners/run_int8_student_signal_guidance.py \
  --input-manifest \
    experiments/data/int-8-retained-int-4-smoke-v1/sha256/13868767846b7004f140cfade3652909347bdbb6708b69cb8c10b36ec2756eb0/input-manifest.json \
  --out-dir .runs/int-8-student-signal-guidance-v1/result
```

The exact verified output is retained at manifest identity
`64779c48ab1f3c9d54fbd697a373666ada7333840058cab88494e9ccf129b38e`;
the additive anchor manifest is
`cd14e23b7251fa0f5f3798f58e47a42aa467c133c5e4960657688b9f43d18f98`.
The retention boundary and raw manifest identities are in
[`retention.json`](data/int-8-student-signal-guidance-v1/sha256/64779c48ab1f3c9d54fbd697a373666ada7333840058cab88494e9ccf129b38e/retention.json).
