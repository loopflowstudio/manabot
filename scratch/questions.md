# Assumptions taken (reversible)

- Rehearsal size: every stage function does one update with at most one game
  per stream, on CPU unless `device=` given; temporary store deleted. Chosen
  over "zero updates" so the learner, export and reload paths actually run.
- Stage names default to the function name numbered by call order
  (`self-play`, `self-play-2`); `stage=` overrides. Resume relies on this order.
- Resume re-enters the function from the top; completed stages replay their
  recorded results. Code between stages must be deterministic.
- Model names lowercase and unique; models are seeded from `run.seed` + name.
- Kept the `#regimes-and-stages` heading so existing links stay valid.
