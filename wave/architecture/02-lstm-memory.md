# 02: LSTM Memory for Imperfect Information

## Finish line

Agent has optional LSTM layer that maintains memory across steps within
an episode. A/B test shows whether memory improves play.

## Changes

### 1. LSTM integration

Add an LSTM layer between the encoder and the policy/value heads.
Follow PufferLib's pattern for managing hidden states across vectorized
environments:

- Each env slot has its own hidden state (h, c)
- Hidden states reset when episodes end
- Hidden states are carried forward between rollout steps
- During PPO update, hidden states are recomputed from stored observations
  (or stored and replayed — tradeoff between memory and compute)

```python
class Agent(nn.Module):
    def __init__(self, ...):
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)

    def forward(self, obs, hidden_state=None):
        encoded = self.encode(obs)  # existing encoder
        if self.use_lstm:
            lstm_out, new_hidden = self.lstm(encoded, hidden_state)
            # Use lstm_out for policy/value heads
        ...
```

### 2. What memory helps with

In MTG, the agent currently has no knowledge of:
- What cards the opponent has played (information about their hand)
- What's in the graveyard from earlier actions
- How many lands it has already played this turn
- What happened on the stack earlier

Some of this is in the observation (graveyard is visible), but the
temporal pattern — the sequence of events — is lost without memory.

### 3. A/B test

Same setup as attention A/B. Compare LSTM vs no-LSTM on:
- Win rate vs random
- Win rate in scenarios where memory matters (e.g., opponent with
  removal spells — the agent should learn to play around them)

## Done when

A/B results with clear conclusion. If LSTM helps, it becomes the
default. If not, document why (maybe the observation already captures
enough state for simple decks).
