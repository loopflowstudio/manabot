# Early learning in the batching and size grid

Observed 2026-10-17 under [frozen protocol013](protocols/grid-early-learning-013.md).

At131,072transitions, the small64×2 model at64streams retained substantial early learning:53.1% score against scripted greedy and84.4% against random, versus56.3% and84.4% at4streams. Its observed training duration was296.9seconds versus1778.0seconds. This makes64streams a useful longer-training candidate. It does not establish equivalence: each endpoint has only four paired deal blocks, and training ran once per cell on different CPU hosts.

Increasing to256streams further shortened small-model training to83.8seconds, but the endpoint scored21.9% against greedy and65.6% against random. The early tradeoff makes transitions/second insufficient for selecting a training recipe. Longer independent-seed comparisons must retain64streams, evaluate skill against both experience and elapsed training time, and compare larger batches without assuming the fastest short run learns best.

At64streams, greedy scores were53.1%,50.0%,28.1% for64×2,192×4,384×8 respectively. This short horizon shows no early advantage for the largest tested model. It cannot distinguish a lower ceiling from slower onset of useful learning. Both added architectures64×4 and384×2 are still missing.

## Complete observed cells

All cells use one training seed13571 and exactly131,072transitions. Each score is the mean of32games across four paired deals and eight seat/deck/mirror legs. Scores count win1,draw0.5,loss0; no draws occurred in this panel. Initialization scored15.625% against greedy and46.875% against random in every observed cell. Parentheses show improvement in percentage points from that cell’s own initialization.

| Model | Streams | Training seconds | Greedy score (change) | Random score (change) |
|---|---:|---:|---:|---:|
| 64×2 | 4 | 1778.0 | 56.2% (+40.6) | 84.4% (+37.5) |
| 64×2 | 64 | 296.9 | 53.1% (+37.5) | 84.4% (+37.5) |
| 64×2 | 128 | 129.2 | 37.5% (+21.9) | 56.2% (+9.4) |
| 64×2 | 256 | 83.8 | 21.9% (+6.2) | 65.6% (+18.8) |
| 192×4 | 4 | 1756.4 | 34.4% (+18.8) | 65.6% (+18.8) |
| 192×4 | 64 | 220.5 | 50.0% (+34.4) | 75.0% (+28.1) |
| 192×4 | 128 | 171.3 | 43.8% (+28.1) | 71.9% (+25.0) |
| 192×4 | 256 | 112.3 | 25.0% (+9.4) | 50.0% (+3.1) |
| 384×8 | 64 | 391.4 | 28.1% (+12.5) | 53.1% (+6.2) |
| 384×8 | 128 | 365.6 | 15.6% (+0.0) | 46.9% (+0.0) |
| 384×8 | 256 | 166.3 | 21.9% (+6.2) | 53.1% (+6.2) |
| 384×8 | 512 | 140.6 | 21.9% (+6.2) | 50.0% (+3.1) |

## Interpretation and limits

Batching changes training, not just hardware efficiency. These runs retain64transitions per stream:4/64/128/256/512streams produce512/32/16/8/4collection-and-learning iterations at the fixed experience endpoint. Actual Adam steps and filtered sample exposure also differ. Schedules were mapped to the reference endpoint, but that does not restore the missing parameter-update opportunities. The screen does not isolate which of these mechanisms explains the learning differences.

Twelve of25declared cells completed. Large384×8 at4streams exhausted its learner deadline at97,024transitions and is censored, not included at a shorter horizon. Small512,middle512 and all ten cells of64×4 and384×2 remain uncompleted. Pre-training infrastructure failures are retained separately and are not model outcomes.

Four deal blocks are correlated repeated measures, not32independent samples; one training seed gives no estimate of training-seed variability. The [machine-readable summary](data/grid-early-learning-013-summary.json) retains all paired block scores and descriptive exact4^4block-bootstrap intervals. These intervals are not adjusted for the grid comparisons and do not establish noninferiority or broad architecture rankings. This screen does not resolve the historical CUDA/NVML failures or finite-weight anomaly.

## Verification and retained evidence

All24evaluation commands,48checkpoint/opponent panels and1,536games completed. All895manifest entries verify by size and SHA256; exact checkpoint coordinates, source/runtime compatibility, frozen protocol, seat/deck/registration bindings, full trace hashes and replay receipts pass. Recomputing the ordinary monitor summaries reproduced stored scores. Local evaluation took1309.95seconds, with no GPU rental.

The raw local evidence is in`.runs/grid-007/early-learning-013/`; analysis and the verification receipt are in`.runs/grid-007/early-learning-013-analysis/`. The verifier is`.runs/grid-007/analyze_early_learning.py`. The raw manifest SHA256 is`f3425a5ce00053a357370aeda8e49ee95852d48c89aa2f7f79bf4fd7a1999973`; protocol SHA256 is`93001968b52f1624bb7128ab41bab80d15516689af6ae90d08aa462cfda3c08d`. Training source is`f057cfebb3e9be07addfff9ae7e04c0106b85ef3` for these12cells. Evaluation uses the separately validated local route012; its scores are not mixed with prior Linux GPU calibration scores.
