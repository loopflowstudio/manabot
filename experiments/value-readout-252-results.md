# Depth helps at 300 updates; a learned critic readout has no clear advantage

MTG-252 · Final scored report · 9 November 2026

**Keep masked mean as the simple reference readout. Depth 4 is promising at
equal updates, but this study does not establish better strength per training
hour.** The two depth-4 readouts are statistically unresolved. This is evidence
for the next training design, not a production-default promotion.

All five arms completed 300 updates with three seeds each. Across seven
milestones and head-to-head comparisons, the retained evaluation contains
**28,200 valid games in 7,050 complete four-game pairings**, with no missing,
duplicate or failed result slots. Each plotted arm/milestone uses 600 games.

## Learning through 300 updates

![Learning curves for all five arms](data/value-readout-252/learning-curves.png)

The depth-4 arms finish at 61.0% and 61.7% against the heuristic, compared with
54.3–58.0% for depth 2. The curves are not monotonic, and most of the separation
appears late. Lines show means over seeds, not confidence bands.

| Readout | Depth | Heuristic score at 300 | 95% interval | Individual seeds |
| --- | ---: | ---: | ---: | --- |
| Masked mean | 2 | 58.0% | 52.2–63.5% | 55.0, 58.5, 60.5% |
| Value token | 2 | 54.3% | 49.2–59.5% | 52.5, 54.0, 56.5% |
| Attention readout | 2 | 57.8% | 52.5–63.2% | 59.0, 57.0, 57.5% |
| Masked mean | 4 | 61.0% | 55.2–66.8% | 63.5, 57.5, 62.0% |
| Value token | 4 | 61.7% | 56.0–67.3% | 63.0, 63.0, 59.0% |

Score awards one for a win and one-half for a draw. All arms are width 64,
four heads, with the same Elemental recipe; depth and readout are the studied
architecture choices. No depth-4 attention-readout arm was run.

## What changed the outcome?

![Endpoint scores and paired differences](data/value-readout-252/endpoint-comparisons.png)

Paired differences compare the same seed and evaluation deal. Depth 4 improves
the token's heuristic score by **7.3 percentage points [1.0, 13.2]** and masked
mean's by **3.0 points [−3.5, 9.0]**. The token depth contrast is the clearest
positive signal, but remains exploratory with three training seeds.

At depth 4, token minus masked mean is just **+0.7 points [−5.5, 6.5]**.
At depth 2, attention readout minus masked mean is −0.2 points [−5.7, 5.8],
and token minus masked mean is −3.7 points [−8.5, 1.2]. None establishes a
readout winner.

Direct play at update 300 reinforces that uncertainty:

| First policy vs second policy | First policy's score | 95% interval |
| --- | ---: | ---: |
| Depth 4 masked mean vs depth 2 masked mean | 53.5% | 49.3–57.5% |
| Depth 4 value token vs depth 2 value token | 53.3% | 48.0–58.3% |
| Depth 4 masked mean vs depth 4 value token | 52.5% | 48.3–56.3% |
| Depth 2 masked mean vs depth 2 value token | 50.0% | 44.5–55.3% |
| Depth 2 masked mean vs depth 2 attention readout | 50.0% | 43.3–57.0% |

The small heuristic lead for depth-4 token reverses in direct play. Both
comparisons are compatible with a tie; neither supports selecting the token.

## Equal updates are not equal training budgets

![Learning curves at matched sample, exposure and work budgets](data/value-readout-252/matched-budgets.png)

These curves interpolate each seed between its evaluated checkpoints, then
average on support shared by all fifteen runs. They are descriptive estimates,
not newly evaluated policies or significance tests.

At **13.46 million collected transitions**, masked mean depth 4 scores an
interpolated 59.3%, token depth 4 58.4%, attention readout depth 2 57.6%,
masked mean depth 2 57.1%, and token depth 2 52.9%. The ordering of the two
depth-4 readouts already changes from the equal-update comparison.

At **4.30 recorded work hours**, the same arms score 56.8%, 57.6%, 57.5%,
57.0%, and 53.2%, respectively. Depth 4's apparent advantage largely disappears.
Work sums collection and learning durations; it excludes setup, exports and
idle rental time. These historical runs are not a controlled throughput test.

| Arm | Mean transitions | Mean optimizer exposures | Mean work hours | Mean last-75 target explained variance |
| --- | ---: | ---: | ---: | ---: |
| Depth 2 masked mean | 14.44M | 3.61M | 4.68 | .942 |
| Depth 2 value token | 15.50M | 3.88M | 4.78 | .944 |
| Depth 2 attention readout | 14.78M | 3.69M | 4.71 | .942 |
| Depth 4 masked mean | 14.00M | 3.50M | 4.92 | .938 |
| Depth 4 value token | 14.66M | 3.66M | 4.83 | .944 |

No retained update reports floor binding or an empty actor batch. The critic
fits bootstrapped lambda targets well in every arm; that does not measure
prediction of eventual wins. The older floor-starvation explanation is not
supported here. Final-outcome value calibration, inference latency and GPU
peak memory were not measured. Sampled host RSS ranges from 11.56 to 14.84 GiB
across runs and must not be presented as GPU memory.

## Implications for MTG-267

Keep the **same masked-mean readout across the two model-size arms**, unless
another explicit hypothesis requires changing it. MTG-252 supplies no evidence
that a learned readout deserves that additional axis. Treat extra depth as a
capacity candidate, not a proven cost-efficient improvement or evidence that
eight layers will beat four.

Preserve MTG-267's equal-training-hours comparisons, regular heuristic and
head-to-head evaluations, and calibration-based damping horizons. Also retain
transitions and optimizer exposures: an update count alone obscures meaningful
differences. MTG-267's newer inputs, catalog vocabulary and TLA population need
their own results; these Elemental scores cannot select its winning architecture.
Its single seed per size supports a case study, not a replicated superiority
claim. No additional run or default change follows from this report.

## Method, limitations and evidence

Seeds are 25201–25203. Every evaluation uses the same fifty non-mirror Elemental
deck/deal blocks, with four crossed deck/seat legs per block. Intervals use
4,000 deterministic bootstrap draws, resampling training seeds and **shared
deal blocks** independently, preserving the same deal draw across seeds and
the same indices across paired arms. This final analysis replaces the earlier
within-seed deal resampling; point estimates are unchanged, intervals account
for the shared schedule. There is no multiplicity correction or predeclared
promotion threshold. Repeated use of this schedule and selection of the depth
extension after early results make this an exploratory study, not a fresh
confirmatory holdout.

Continuation boundaries restart game streams and RNGs. Depth-4 masked mean
restarted at 150; fresh depth-4 token did not. Depth-2 continuations and capacity
recovery also differed historically. The token changes actor context and shifts
some initialization draws, so it is not a critic-only causal intervention.
Successful evaluation slots do not certify general engine correctness.

Retained whole-rental bills plus prior estimates total **USD47.5021 before the
last evaluation**. The last evaluation had a USD3 cap, keeping the conservative
total at **USD50.5021**, within the authorized USD55. Its worker completed all
2,250 new matches and exited zero; launcher state remained stale. Cancellation
was requested after verifying the results, then its idle pod was deleted through
the provider on 9 November. Final invoice reconciliation remains separate from
the completed scientific result; the cap is not a bill.

The [study and execution record](value-readout-252.md) preserves architecture,
source and continuation details. [Results JSON](data/value-readout-252/results.json)
holds every score and interval; [training summary](data/value-readout-252/training-summary.json)
holds seed-level budgets and diagnostics. The artifact manifest retains raw
evaluation blocks and selected per-update metrics. Four evaluation IDs are
listed in the results JSON. Reproduce score tables and graphs with
[value_readout_report.py](value_readout_report.py), passing the retained block
directory to `--source`, original training metrics to `--metrics`, and an output
directory to `--out`. All published evidence remains bound to its executed
source and world.
