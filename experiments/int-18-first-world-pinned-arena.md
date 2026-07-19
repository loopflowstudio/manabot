# INT-18: The first world-pinned arena rating run

**Result (2026-07-18, w2): `rated_not_promotion_eligible`.** The frozen INT-6
arena completed its first retained production rating over the five code-only
anchors and the dPUCT-32 challenger. The result contains 720 games, all 15
payoff cells, 24 paired deal blocks and both seat legs per cell, one connected
six-player component, and 2,000 successful global-deal-block bootstrap fits.
dPUCT-32 rated 1333 on this scale: clearly above random and scripted-greedy,
clearly below flat-MC-64, and unresolved against flat-MC-4 and flat-MC-16.
It cannot be promoted because the frozen cohort contains no same-compute
incumbent. This is a production rating for these exact players and world, not
a general method or strength claim.

## Frozen authority and cohort

Generation used the authenticated release at commit
`76d0834797316c3b6e153ed10e5fadd146a8980a`. The frozen contract file SHA-256
is `fc9cb76c...`, its arena source closure is `b722c811...`, and its native
extension is `18d04fe6...`. The dPUCT registration identity is `6b1eb785...`
and its source closure is `7236414e...`. Current source did not impersonate
these identities: the historical bytes generated and replayed games, while
the current INT-18 runner authenticated, derived, and retained the result. Its
separate non-generating verifier receipt binds
`experiments/runners/run_int18_arena_rating.py` at exact SHA-256
`47976d1c9d393767f895b7704019c72267a0d683eed64fa13d9ba0ab710b91c6`
with role `current-int18-envelope-derivation-and-verification` and boundary
`post-generation-envelope-only`. Both the result and envelope manifest retain
and authenticate that receipt.

The complete result is retained at manifest identity
`af0c3f56745ba4f60e5e3f612787b11c65d6b125917ec1e59b1835e113765b2b`.
The anchor manifest is `cc83abb7...`; the challenge manifest is
`998ad75d...`. The exact identities and source-file receipts are in
[`int18-result.json`](data/int-18-first-world-pinned-arena-v1/sha256/af0c3f56745ba4f60e5e3f612787b11c65d6b125917ec1e59b1835e113765b2b/result/int18-result.json),
and the 42-file, 21,507,970-byte retention boundary is in
[`retention.json`](data/int-18-first-world-pinned-arena-v1/sha256/af0c3f56745ba4f60e5e3f612787b11c65d6b125917ec1e59b1835e113765b2b/retention.json).

## Ratings and paired uncertainty

The frozen estimator is seat-aware Gaussian-MAP Bradley-Terry, anchored at
random = 1000 Elo. It converged in six iterations with log loss 0.515 and a
fitted seat-0 effect of -96.3 Elo. Intervals below are the 2.5th and 97.5th
percentiles from 2,000 bootstraps that resample whole global deal blocks.

| Player | Point rating | Paired-bootstrap interval |
| --- | ---: | ---: |
| flat-MC-64 | 1513 | 1431–1627 |
| flat-MC-16 | 1369 | 1287–1467 |
| dPUCT-32, four worlds | 1333 | 1260–1423 |
| flat-MC-4 | 1321 | 1281–1380 |
| scripted-greedy | 1203 | 1144–1277 |
| random | 1000 | 1000–1000 |

The dPUCT-minus-opponent paired intervals are -265 to -107 Elo against
flat-MC-64, -109 to +26 against flat-MC-16, -58 to +82 against flat-MC-4,
+85 to +185 against scripted-greedy, and +260 to +423 against random. The
honest result is therefore a clear ordering only at the top and bottom of the
cohort; the middle three players are not separated by this run.

The complete rating fit, every pairwise difference interval, and all
per-deal paired sweep/split counts are retained in
[`rating.json`](data/int-18-first-world-pinned-arena-v1/sha256/af0c3f56745ba4f60e5e3f612787b11c65d6b125917ec1e59b1835e113765b2b/result/challenge/rating.json)
and
[`paired-deal-uncertainty.json`](data/int-18-first-world-pinned-arena-v1/sha256/af0c3f56745ba4f60e5e3f612787b11c65d6b125917ec1e59b1835e113765b2b/result/paired-deal-uncertainty.json).

## Complete payoff matrix

Every cell contains 48 games: 24 exact deals with both seat assignments. There
were no draws.

| Player A | Player B | A–B wins | A score | Paired blocks: A sweeps / splits / B sweeps |
| --- | --- | ---: | ---: | ---: |
| dPUCT-32 | flat-MC-16 | 19–29 | 0.396 | 2 / 15 / 7 |
| dPUCT-32 | flat-MC-4 | 28–20 | 0.583 | 5 / 18 / 1 |
| dPUCT-32 | flat-MC-64 | 14–34 | 0.292 | 1 / 12 / 11 |
| dPUCT-32 | random | 38–10 | 0.792 | 15 / 8 / 1 |
| dPUCT-32 | scripted-greedy | 34–14 | 0.708 | 10 / 14 / 0 |
| flat-MC-16 | flat-MC-4 | 23–25 | 0.479 | 3 / 17 / 4 |
| flat-MC-16 | flat-MC-64 | 18–30 | 0.375 | 2 / 14 / 8 |
| flat-MC-16 | random | 45–3 | 0.938 | 21 / 3 / 0 |
| flat-MC-16 | scripted-greedy | 30–18 | 0.625 | 8 / 14 / 2 |
| flat-MC-4 | flat-MC-64 | 11–37 | 0.229 | 0 / 11 / 13 |
| flat-MC-4 | random | 45–3 | 0.938 | 21 / 3 / 0 |
| flat-MC-4 | scripted-greedy | 28–20 | 0.583 | 5 / 18 / 1 |
| flat-MC-64 | random | 47–1 | 0.979 | 23 / 1 / 0 |
| flat-MC-64 | scripted-greedy | 41–7 | 0.854 | 17 / 7 / 0 |
| random | scripted-greedy | 17–31 | 0.354 | 1 / 15 / 8 |

The machine-readable full matrix is
[`payoff-matrix.json`](data/int-18-first-world-pinned-arena-v1/sha256/af0c3f56745ba4f60e5e3f612787b11c65d6b125917ec1e59b1835e113765b2b/result/challenge/payoff-matrix.json).
[`connectivity.json`](data/int-18-first-world-pinned-arena-v1/sha256/af0c3f56745ba4f60e5e3f612787b11c65d6b125917ec1e59b1835e113765b2b/result/connectivity.json)
authenticates six nodes, all 15 expected edges, one component, random-anchor
reachability, and no missing or unexpected cell.

## Competency, systems cost, and integrity

dPUCT-32 solved S1 counter-the-bomb on 40/100 runs and S4 race-vs-block on
3/100; it remained at 0/100 on S2 hold-the-wipe, S3 bolt-the-threat, and S5
hold-up-quench. Its rating therefore does not erase the standing delayed-control
failure. On the shared 128-root matched profile, dPUCT-32 measured 163.4 ms
p50 and 392.0 ms p95, 5.71 decisions/s, 182.6 simulations/s, a 6.7 MB peak
RSS delta, zero illegal actions, zero root mutations, and zero playout-cap
hits.

The combined run consumed 0.442 wall hours and a conservative 1.767 core
hours at four workers. The frozen cap was 16 wall hours, 64 core hours, 4 GiB,
and four workers; every clause passed. All 15 Command shards replayed under the
frozen extension. Across 720 games and 76,991 decisions there were zero actor,
frame, offer, Command, state, outcome, trace, missing-decision, private-
exposure, legality, root-mutation, truncation, or offer-binding failures.

## Exact-range evidence wait

The requested exact-range versus uniform-determinization comparison did not
run. The frozen INT-9 contract still marks its required w2 policy/value
likelihood checkpoint `unresolved_required`, with no path or SHA-256. A real
retained checkpoint at SHA-256 `06794769...` loads, but the registered contract
does not select it; choosing it after seeing availability would create a new
player identity. The frozen INT-6 registration and match lifecycle also do not
admit the stateful exact-range player. No neutral likelihood, authored belief,
or other substitute was used, and no exact-range game was started.

The typed failure is retained in
[`exact-range-evidence-wait.json`](data/int-18-first-world-pinned-arena-v1/sha256/af0c3f56745ba4f60e5e3f612787b11c65d6b125917ec1e59b1835e113765b2b/result/exact-range-evidence-wait.json).
It does not invalidate the independent frozen-anchor/dPUCT result, but the R4
belief comparison remains open.

## Decision and verification

Keep flat-MC-64 as the strongest player on the frozen arena-v1 scale. Retain
dPUCT-32 as a rated production challenger, but do not promote it: no
same-compute incumbent exists, its interval overlaps the lower search anchors,
and its delayed-control competencies remain poor. Development pairwise matches
remain explicitly non-admission evidence.

Verify the retained result without generating games:

```bash
uv run python experiments/runners/run_int18_arena_rating.py \
  --stage production \
  --out-dir \
  experiments/data/int-18-first-world-pinned-arena-v1/sha256/af0c3f56745ba4f60e5e3f612787b11c65d6b125917ec1e59b1835e113765b2b/result \
  --verify-only
```

The command reports `no_generation=true`, `no_replay=true`, current verifier
SHA-256 `47976d1c...`, and the same
`af0c3f56745ba4f60e5e3f612787b11c65d6b125917ec1e59b1835e113765b2b`
manifest identity. It authenticates the frozen exact-range evidence-wait file
through the retained result and manifest receipts rather than re-deriving that
result-time observation from a later INT-9 contract.
