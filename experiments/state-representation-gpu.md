# Pooled versus GRU state reduction on v7, GPU horizon

2026-11-05 · Task 5da5f4 · completed prospective comparison under the
[GPU horizon protocol](protocols/state-representation-5da5f4-gpu.md) and its
recovery amendment (`RECOVERY` SHA256 `a0383ac7…f86f7be`; `SCORING` unchanged,
`f09750f1…67402`)

## Answer

**No qualified plateau.** The predeclared criterion requires every one of the
final three 32-update gains, for both arms, to have a four-seed 95% t interval
entirely within ±5 points. Only one of six does. Learning is slowing, but the
four-seed intervals are too wide to call it flat.

**No architecture difference established.** Held-out GRU minus pool at update
256 is **−1.95 points, 95% [−5.88, +1.98]** over four paired seeds. Neither
advantage nor equivalence is shown; the interval excludes a GRU gain above
about 2 points on this matchup and horizon. No model or default was promoted;
fresh observation input remains v6.

This study supersedes the [CPU comparison](state-representation-v7.md) as the
evidence for this question. That 128-update study remains preliminary and is
not enough to choose an architecture.

## What ran

Both arms use v7 observations and the same object-attention model. "Pool" and
"GRU" name two ways to reduce effect-payload and continuation tokens before
attention, within a single decision; the comparison changes only that reducer.
Neither arm carries memory across decisions, so this is not an agent-memory
result.

Four paired seeds (74211–74214), 256 updates × 256 complete self-play games =
65,536 games per learner, 524,288 in total, on RTX 3090s. Evaluation: 72 curve
endpoints (deals 85001–85032, 128 games each) and 8 held-out endpoints
(86001–86064, 256 games), heuristic opponent, both decks and seats.
**11,264/11,264 scored games replayed; zero game failures.**

## Results

Arm means across four seeds (score = wins + half draws, %):

| Update | 0 | 32 | 64 | 96 | 128 | 160 | 192 | 224 | 256 | Held-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pool | 7.8 | 30.7 | 38.9 | 42.2 | 42.6 | 43.0 | 46.9 | 46.7 | 49.4 | 49.0 |
| GRU | 7.8 | 29.7 | 38.3 | 40.6 | 42.0 | 42.0 | 46.7 | 47.1 | 46.9 | 47.1 |

Plateau window, mean gain [95% interval], points (✓ = within ±5):

| Arm | 160→192 | 192→224 | 224→256 |
|---|---|---|---|
| Pool | +3.9 [−1.7, +9.6] | −0.2 [−2.5, +2.2] ✓ | +2.7 [−1.4, +6.9] |
| GRU | +4.7 [−1.7, +11.0] | +0.4 [−5.0, +5.8] | −0.2 [−7.3, +6.9] |

Per seed, pool / GRU (update-256 curve; held-out): 74211 51.6/40.6, 47.7/42.6;
74212 50.0/51.6, 51.2/48.4; 74213 48.4/46.9, 50.4/50.0; 74214 47.7/48.4,
46.9/47.3. Only seed 74211 shows a large gap.

## Limits

- One matchup, against the heuristic only. About 49% means roughly even with
  it, not strong play.
- Pool versus GRU changes two reductions together (effect payloads and
  continuations).
- V7 only. Policy input v8 always reduces effect programs with the GRU to keep
  their token order, so this comparison does not apply to its effect path.
- Paired learners of a seed ran on different hosts.
- Pool/74212 was restarted from initialization; its weights differ from the
  lost run's by host QR rounding only (≤1.04e-6, receipt
  `init-comparison-typed_pool-74212.json`).
- Four seeds give weak power; the plateau verdict is about interval width, not
  a demonstrated continued rise.

## Attempts and costs

Every attempt stays in the denominator and the ledger.

- The first cohort was cancelled on 2026-11-04 before scoring, when measured
  concurrent GPU throughput invalidated its allocation budget.
- Scientific Run `e51f61b92931` ended **`lost / machine_lost`** on 2026-11-05
  after three of eight learners completed, with no curve or held-out endpoint.
  Its record and three completed learners are unchanged; the recovery
  amendment admitted those learners and restarted the other five from
  initialization on their seeds.
- Evaluation `664f44738f35`, paused pending the pool/74212 initialization
  check, stays an unscored attempt (`collect_failed`).
- Unscored endpoint attempts were repeated only for missing evidence: one host
  stall (two zero-decision timeouts), the cancellation above, and three Vast
  creates refused with HTTP 400 that Etude retained as unknown replies. Two
  learner hosts never booted and were restarted from initialization.

Recovery spent about **$6.75** in Etude estimates. With the **$12.14** accounted
before recovery, that is about **$18.89 of the original $25**, an operational
estimate rather than a settled invoice. No live rentals remain.

## Evidence

Manifest-published under
`experiments/data/state-representation-gpu-5da5f4/recovery-20261105/`:
`comparison.json` (frozen analysis, attempt counts, scores), `ledger/` (every
attempt with receipts), learner deliveries with checkpoints, unit allocations,
driver state and logs, `init-comparison-typed_pool-74212.json`, and
`retained-admission.json`.
