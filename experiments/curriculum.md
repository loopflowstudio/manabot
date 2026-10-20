# MTG-122 follow-up: simple-pool curriculum before Allies versus Lessons

**Result (2026-10-20): no useful curriculum effect.** Spending the first
quarter of training on L1 simple creatures and lands does not make Allies versus
Lessons (L3) learning faster or stronger than spending it on L3. It starts
behind and only catches up. Design, decisions and the frozen rules are in
[the design](../scratch/measure-learning-progress-across-simple.md); rules as
code in `experiments/runners/curriculum_analysis.py`, frozen plan in
[curriculum-plan.json](curriculum-plan.json).

## What ran

- **Arms**, six seeds (12251–12253, 12271–12273), the complexity recipe
  (64x2, 64 streams × 64 transitions, 128 iterations = 524,288 learner
  transitions). Both arms continue one learning state (weights, EMA, Adam,
  schedule clock) across iteration 32; only the first 32 iterations differ:
  - *curriculum (A)*: L1 `l1_swift`/`l1_heavy`, then L3 via a declared
    `source_match` setup transfer;
  - *scratch (B)*: L3 throughout, through the same boundary.
- **Training:** Etude run `8cc849175eab` on one L4, source `adcf54e3`, all
  12 fits in 5,371 s, $0.37 (SDK estimate). 222 outputs fetched with digest
  checks; every fit passed evaluation admission.
- **Scoring:** 180 cells × 400 games = 72,000 games on four rented RunPod
  cpu3c 8-vCPU pods ($0.24/h), 05:31–11:30 UTC. Held-out L3 matrix (all four
  deck-role cells, both play orders, 50 deals) against heuristic, greedy and
  random at iterations 0, 32, 64, 128, with same-deal frozen-initialization
  controls. Milestones 0/64/128 use the complexity cohort's L3 protocols
  exactly; 32 uses deals 1,227,132,000–049. Pods ran source `c0b65517` with
  `curriculum_campaign.py` from `112aff3f` (SHA-256 checked on each pod).
- **Parity:** the initialization cells for seeds 12251–12253 must repeat the
  Mini's complexity cells. 8 of 9 were compared (3,200 games across all four
  pods and four EPYC generations): 0 differing games
  ([parity.txt](data/curriculum/parity.txt)).
- **Failures, retained:** a first scoring attempt failed admission in every
  unit before any game (fetched fits were looked up under `<arm>/` rather than
  `<arm>-<seed>/`; fixed in `112aff3f`). Three pods placed in EU-SE-1 never
  opened SSH and were deleted after ≤13 minutes. Slower pod f was stopped
  after 32 units by its stop file; its 13 unstarted units ran on the other
  pods. All 180 cells completed with 400 games; none was rerun.
- **Cost:** GPU $0.37 + CPU ≈ $5.59 from pod receipts
  ([cpu-pods.json](data/curriculum/cpu-pods.json)) ≈ $6.0, within the plan's
  $4.35 GPU lease and $12 scoring allowance. MTG-122 inclusive ≈ $10.8 of $35.
  Launch tooling and receipts: `.runs/mtg122/curriculum-gpu/` and
  `.runs/mtg122/curriculum-cpu/` (a copy of MTG-135's ad-hoc pod scripts).

## Results

Curriculum minus scratch, paired by seed and deal, mean of the four matrix
cells, points; six-seed mean with 95% t interval (df 5), seeds positive.
Analysis: [analysis.json](data/curriculum/analysis.json).

| Anchor | 32 | **64 (primary)** | 128 |
|---|---|---|---|
| heuristic | −4.6 [−11.8, +2.5], 2/6 | **−2.3 [−5.6, +0.9], 2/6** | +0.4 [−4.1, +4.9], 4/6 |
| greedy | −7.5 [−15.2, +0.1], 1/6 | **−0.8 [−4.0, +2.5], 2/6** | +1.5 [−2.0, +4.9], 5/6 |
| random | −11.9 [−20.6, −3.2], 1/6 | −1.6 [−3.1, −0.1], 0/6 | +3.3 [+1.1, +5.4], 5/6 |

Frozen verdict at 64: **greedy — no useful effect** (interval inside ±5);
**heuristic — inconclusive** (lower bound −5.6), leaning negative. Neither
approaches the +5 "useful" criterion. The decision anchor is still to be set
by `anchor_validation` once MTG-153's heuristic-vs-greedy pairing exists; it
cannot change "not useful".

Gain over the same seed's frozen initialization (points):

| Anchor | Arm | 32 | 64 | 128 |
|---|---|---|---|---|
| heuristic | curriculum | +6.3 | +19.1 | +24.5 |
| heuristic | scratch | +10.9 | +21.4 | +24.0 |
| greedy | curriculum | +7.3 | +27.7 | +32.5 |
| greedy | scratch | +14.8 | +28.5 | +31.0 |
| random | curriculum | +9.0 | +28.5 | +29.7 |
| random | scratch | +21.0 | +30.0 | +26.4 |

- **Speed:** time to +10 points against greedy is iteration 32 for five of
  six scratch seeds but only two of six curriculum seeds; the rest reach it by
  64. At the boundary the L1-trained policy, playing L3 decks for the first
  time, has about half the scratch gain.
- **Ceiling:** by 128 the arms are level against heuristic and greedy. The
  random-anchor endpoint favours curriculum by 3.3 points [+1.1, +5.4]; scratch
  also dips from 64 to 128 there, so this is a small secondary signal, not a
  benefit claim.
- **Cells at 64:** curriculum is slightly ahead in Lessons mirrors and behind
  when playing Allies (heuristic: LvL +1.7, LvA +0.3, AvL −6.5, AvA −4.8).
- **Boundary check (B vs C):** split scratch matches the complexity cohort's
  continuous L3 fits on the same deals (seeds 12251–12253, e.g. greedy at 64:
  52.5/57.0/53.2% vs 53.0/55.8/52.5%), so the continuation itself costs
  nothing.
- **Compute:** at equal transitions the curriculum arm used less active time
  (L1 source segments 87–108 s vs 116–147 s for L3), so a seconds-matched
  comparison would give it slightly more L3 training; that would not close a
  gap that is absent at the endpoint and negative earlier.

## Conclusion

For this recipe and scale, an L1 warm-up is not a useful curriculum: it delays
L3 progress and yields no endpoint gain. Simple full games remain useful as a
diagnostic (MTG-122's complexity result), not as pretraining. Not tested:
L2→L3, other switch points or shares, weights-only transfer with a fresh
schedule, longer horizons where L3 learning may plateau, and mirror training.
None is selected by this result.
