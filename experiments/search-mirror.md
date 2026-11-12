# Mirror search on MTG-183: results

Final report, 2026-11-12. Exploratory: the settings were chosen after looking
at earlier results on the same deals, and no interval is corrected for multiple
comparisons.

Magnetic-mirror search plays each move by rolling the frozen MTG-183 policy
forward from every candidate move in several reshuffled copies of the game,
then nudging the policy toward the moves that did best. The study asked three
questions:

1. Does mirror search add strength, and how does it compare with PUCT at
   similar cost?
2. How does strength scale with the number of shuffles?
3. How much of the gain survives when the search must guess the opponent's
   hand?

The full report, with charts and every comparison, is published at
`experiments/data/search-mirror-v1/report.html`
(`uv run --extra artifacts python -m manabot.data fetch experiments/data/search-mirror-v1`).
Protocols: [v1](protocols/search-mirror-v1.md), [v2](protocols/search-mirror-v2.md).

## Answers

**1. Yes, clearly, and it beats PUCT at lower cost.** With playable uniform
hand sampling, mirror search beats the policy's own sampled play by
+12.5 points [+8.1, +17.0] at 4 shuffles, +15.7 [+11.4, +20.0] at 8 and
+17.4 [+13.2, +21.5] at 16. With the true opponent hand (oracle) it reaches
+22.7 [+18.4, +27.1]. PUCT with 1,024 simulations costs more (4.5 s per
decision against 3.5 s) but gains only +4.5 [−0.1, +9.1]. On the 478 deals both
played, 16-shuffle mirror finishes +13.2 points [+9.1, +17.3] ahead of it, and
4-shuffle mirror at 0.9 s finishes +8.4 [+4.2, +12.7] ahead. The two never
played each other; both were measured against the same opponent.

**2. More shuffles help, with diminishing returns.** With oracle hands, 4 → 8
shuffles adds +6.6 points [+3.0, +10.2] and 8 → 16 adds +2.8 [−0.4, +6.0].
With uniform hands the steps are +3.2 [−0.8, +7.2] and +1.8 [−2.0, +5.4]. Only
the oracle 4 → 8 step clearly exceeds zero. Over the whole range 16 shuffles
beat 4 by +9.4 points with oracle hands and +4.9 with uniform hands. Each
doubling doubles the cost.

**3. Most of it survives: about 94% at 4 shuffles, 76% at 16.** Knowing the
opponent's hand is worth +0.8 points [−3.2, +4.8] at 4 shuffles, +4.2
[+0.2, +8.2] at 8 and +5.2 [+2.0, +8.4] at 16. The gap grows with shuffles in
these games, but the test of that growth (the difference between the samplers'
shuffle gains) is −3.4 [−9.0, +2.2] for 4 → 8 and −1.0 [−5.8, +3.8] for
8 → 16, so the games do not establish that the samplers scale differently.

## Cost

| Setting | Gain over sampled policy | Seconds per decision | Games per hour, one worker |
|---|---|---:|---:|
| Uniform hands, 4 shuffles | +12.5 [+8.1, +17.0] | 0.92 | 37.8 |
| Uniform hands, 8 shuffles | +15.7 [+11.4, +20.0] | 1.85 | 19.0 |
| Uniform hands, 16 shuffles | +17.4 [+13.2, +21.5] | 3.53 | 10.2 |
| Oracle hands, 4 shuffles | +13.3 [+8.9, +17.7] | 2.85 | 13.4 |
| Oracle hands, 8 shuffles | +19.9 [+15.7, +24.3] | 5.51 | 6.8 |
| Oracle hands, 16 shuffles | +22.7 [+18.4, +27.1] | 11.13 | 3.4 |

The raw policy takes about 38 ms per decision. A worker is one process with one
inference thread on a Mac mini running seven at once. Uniform rows used the
faster native observation encoder with compact rollout evidence; oracle rows
used the original Python encoder with full audit evidence. At fixed positions
the two samplers cost about the same, so most of the cost difference comes from
the implementation, not the sampler. On fixed positions the compact/native path
made identical moves and was about 5× faster at one thread.

## Method

Depth 20, step size β 0.01, 4/8/16 shuffles, oracle and uniform hand sampling:
six settings, each playing the same 500 deals (3,000 games) against the raw
sampled MTG-183 policy. Gains are whole-deal paired differences against the
policy's own sampled play on those deals, with cell-stratified 95% bootstrap
intervals. An earlier 8,000-game sweep (v1) chose depth and step size against
both the raw policy and the heuristic; its best setting (depth 20, β 0.03,
4 shuffles, oracle) gained +14.7 points [+9.8, +19.6] against the policy and
+9.4 [+5.8, +13.0] against the heuristic.

## Caveats

- Oracle hands are a benchmark; only uniform sampling is playable.
- One frozen policy (MTG-183, one training seed, world w13), its ten decks and
  one opponent. Results may not carry over to other policies, decks or opponents.
- Exploratory selection, no multiple-comparison correction.
- One uniform 16-shuffle game hit the per-decision time limit and is retained
  as a failure; scoring it as a win or a loss changes no comparison's sign.

## Runs

| Run | Commit | Etude attempt | Cost |
|---|---|---|---|
| v1 sweep (8,000 games) | in `mirror-w13-study-frozen` history | mini, sequential workers | $0 |
| v2 oracle stage (`mtg269-mirror-v2`) | `mirror-w13-study-frozen` | `3a2b62144b9a`, cancelled at the oracle barrier as planned | $0 |
| v2 uniform stage (`mtg269-mirror-v2-uniform-native`) | `mirror-w13-study-frozen` | `b9b50aa247df`, succeeded | $0 |

The study ran on a frozen w13 checkout that cannot merge into main. The tag
`mirror-w13-study-frozen` keeps its runners, report builder, compact rollout
evidence and native batch encoder reproducible. Porting compact rollouts and
native batch encoding to main's search code is separate work.

- 2026-11-12: final report over all 3,000 v2 games.
