# MTG-249: GPU representation experiment — final cohort result

Completed 2026-11-05 22:29 UTC under the frozen recovery amendment (protocol
section **Recovery amendment**, `RECOVERY` SHA256 `a0383ac7…f86f7be`; `SCORING`
unchanged, `f09750f1…67402`). This supersedes the 19:56 interim snapshot
(`c4a239dfc:scratch/mtg-249-gpu-interim-report.md`).

## Answer

**No qualified plateau.** The predeclared criterion requires every one of the
final three 32-update gains, for both arms, to have a four-seed 95% t interval
entirely within ±5 points. Only one of six does. Learning is slowing, but the
four-seed intervals are too wide to call it flat.

**No architecture difference established.** Held-out GRU minus pool at update
256 is **−1.95 points, 95% [−5.88, +1.98]** over four paired seeds. Neither
advantage nor equivalence is shown; the interval excludes a GRU gain above
about 2 points on this matchup and horizon.

## What ran

Both arms use v7 observations and the same object-attention model; pool versus
GRU changes only how effect-payload and continuation tokens are reduced before
attention. No memory across decisions. Four paired seeds (74211–74214), 256
updates × 256 complete self-play games = 65,536 games per learner, 524,288 in
total, on RTX 3090s. Evaluation: 72 curve endpoints (deals 85001–85032, 128
games each) and 8 held-out endpoints (86001–86064, 256 games), heuristic
opponent, both decks and seats. **11,264/11,264 scored games replayed; zero
game failures.**

## Results

Arm means across four seeds (score = wins + half draws, %):

| Update | 0 | 32 | 64 | 96 | 128 | 160 | 192 | 224 | 256 | Held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pool | 7.8 | 30.7 | 38.9 | 42.2 | 42.6 | 43.0 | 46.9 | 46.7 | 49.4 | 49.0 |
| GRU | 7.8 | 29.7 | 38.3 | 40.6 | 42.0 | 42.0 | 46.7 | 47.1 | 46.9 | 47.1 |

Plateau window, mean gain [95% interval], points:

| Arm | 160→192 | 192→224 | 224→256 |
|---|---|---|---|
| Pool | +3.9 [−1.7, +9.6] | −0.2 [−2.5, +2.2] ✓ | +2.7 [−1.4, +6.9] |
| GRU | +4.7 [−1.7, +11.0] | +0.4 [−5.0, +5.8] | −0.2 [−7.3, +6.9] |

Per seed, pool / GRU (update-256 curve; held-out): 74211 51.6/40.6, 47.7/42.6;
74212 50.0/51.6, 51.2/48.4; 74213 48.4/46.9, 50.4/50.0; 74214 47.7/48.4,
46.9/47.3. Only seed 74211 shows a large gap.

Interpretation limits: one matchup, against the heuristic only; about 49% means
roughly even with it, not strong play. Pool versus GRU changes two reductions
together. Paired learners of a seed ran on different hosts. Pool/74212 was
restarted from initialization; its weights differ from the lost run's by host
QR rounding only (≤1.04e-6).

## Attempts and costs

Every attempt is in the ledger. Unscored endpoint attempts were repeated only
for missing evidence: one host stall (two zero-decision timeouts), one
cancellation pending the initialization check, and three Vast creates refused
with HTTP 400. Two learner hosts never booted and were restarted from
initialization. Recovery spent about **$6.75** in Etude estimates. With the
$12.14 accounted before recovery, that is about **$18.89 of the original $25**,
an operational estimate, not a settled invoice. No live rentals remain.

## Evidence

Manifest-published under
`experiments/data/state-representation-gpu-5da5f4/recovery-20261105/`:
`comparison.json` (frozen analysis, attempt counts, scores), `ledger/` (every
attempt with receipts), learner deliveries with checkpoints, unit allocations,
driver state and logs, `init-comparison-typed_pool-74212.json`, and
`retained-admission.json`.
