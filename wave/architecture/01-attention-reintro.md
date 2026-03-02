# 01: Re-enable Attention + A/B Testing

## Finish line

Attention mechanism is re-enabled with reduced padding. A/B test shows
whether it helps, hurts, or is neutral for the current task.

## Changes

### 1. Reduce observation padding

Based on profiling from first-light (stage 03), set realistic maximums.
The attention cost is O(n^2) in sequence length — going from 302 to ~72
objects is a 17x reduction in attention compute.

### 2. Re-enable attention with config toggle

`attention_on=True` in a new config. Run the same verification ladder
(beat passive, beat random) with and without attention. Compare:
- Final win rate
- Steps to reach 80% win rate
- Explained variance trajectory
- Training throughput (SPS)

### 3. Tune attention

If attention helps:
- Experiment with number of heads (1, 2, 4)
- Experiment with number of attention layers (1 vs 2)
- Try pre-norm vs post-norm transformer block

If attention doesn't help with simple decks, document that and move on.
It may become valuable when the card pool grows.

## Done when

A/B results logged to wandb with clear conclusion: attention helps / neutral / hurts.
