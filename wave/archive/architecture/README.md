# Architecture

## Vision

Re-introduce the MTG-specific model architecture — attention, memory,
richer observations — now that the training loop is proven correct and
basic learning is demonstrated. Every addition is A/B tested against
the baseline.

### Not here

- Distributed training (that's scale wave)
- New algorithms (NFSP, MCTS) — keep PPO, improve the model
- Card pool expansion — same simple deck, better model

## Goals

1. Attention mechanism demonstrably improves win rate over no-attention baseline
2. LSTM memory demonstrably improves play in situations requiring history
3. Flat observation pipeline achieves 2x+ throughput over dict pipeline
4. Every architectural change has a measured before/after comparison

## Risks

- Attention over-parameterizes for simple decks (no improvement visible
  until card pool grows)
- LSTM training is notoriously finicky (hidden state management across
  episodes, gradient flow through time)
- Premature optimization of the pipeline before the model architecture
  is settled

## Metrics

- Win rate delta vs no-attention/no-LSTM baseline
- Training throughput (SPS)
- Explained variance convergence speed
- Model parameter count and inference latency
