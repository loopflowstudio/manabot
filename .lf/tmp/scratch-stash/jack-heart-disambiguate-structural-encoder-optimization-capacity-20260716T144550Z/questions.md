# W2-266 assumptions

- The original CPU gate is applied to the hot, semantics-identical cached
  catalog path because static typed-program relations can be built once per
  content hash. The original numeric thresholds remain unchanged, and both the
  legacy online path and model-only path remain mandatory reporting.
- The current lifecycle phase is kickoff. This commit is the human-reviewable
  pre-result contract; the machine-readable JSON contract, runner, and tests
  must land before any new training or timing result is generated.
- `relational_message_encoder_v1` is an architecture-path diagnostic, not a
  semantic expansion: it consumes only the eight relations already present in
  `structural-semantic-katas-v1`.
