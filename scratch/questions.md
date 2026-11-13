# MTG-337 decisions and questions

- Use Rust `payment_search_tests.rs` in ordinary debug CI. Reuse the benchmark
  counters under `debug_assertions` as well as `benchmark`; release wheels
  retain no instrumentation. A scoped thread-local limit stops search at the
  first call above 1,000 and restores itself on unwind.
- Exercise native `VectorEnv::step_isolated_into`, the implementation wrapped
  by Python's `step_isolated_into_buffers`. Compare its complete observation
  callback and decision frame with the scalar step. Python buffer plumbing
  is outside this search regression.
- Preserve the reporter script byte for byte and exclude it from Ruff, like
  the existing external fixture. Its historical install instructions remain
  evidence; current reproduction uses the Rust benchmark command in the docs.
- If rules later invalidate the recorded actions, do not rewrite this evidence
  or silently skip it. That requires a new reviewed fixture/outcome decision.
