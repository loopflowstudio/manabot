# MTG-306 design assumptions (headless; reversible, simpler option taken)

- Exploratory, not a finding: a pilot whose numbers decide whether to open a design Task.
- Token model = `ModelConfig.token_stream="pooled"` in manabot (excluded when "tables",
  so default regimes, checkpoints and architecture identities are unchanged). Needed
  so checkpoints load in the arena and evaluation needs no special path.
- Departure from the brief's host conversion: tokens are packed on the device *after*
  the existing per-type embeddings and joins (links, facts, effects, continuation,
  catalog programs) — "per-type embedding over contiguous segments" — so references
  are already resolved and no host remapping exists. The pooled sequence holds
  players, cards, permanents, relation rows and legal actions. Facts and effect
  records stay joins onto their objects (their encoders are unchanged in both arms).
  The fill measurement still counts every entity row type, as asked.
- Pack/unpack round trip is the "loses nothing" check (unit test), not a host
  tokens->tables inverse.
- Pooled arm rejects belief rows and value_token aggregation (Elemental uses masked_mean).
- Learning: Elemental recipe (elemental_variants.elemental_recipe), w64l2, two seeds
  per arm shared across arms, matched updates; scored vs heuristic on the
  closure-lambda-elemental pairings and head-to-head pooled vs tables at matched updates.
- Timing captures come from a 3-update in-situ CUDA run of the tables arm (fresh
  init), so they describe early-training observations. Learner step timing uses a
  stand-in loss (policy log-likelihood + value MSE): same forward/backward cost
  class as the damped loss, stated in the report.
