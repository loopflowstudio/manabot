# Open Questions / Assumptions

1. **Parallel stepping removed.**
   Both `std::thread::scope` and rayon were benchmarked. Per-env step
   cost (~4.4µs) is too low for thread overhead to break even —
   multi-threading was performance-neutral to slightly negative across
   1–256 envs. Sequential `for_each_env` is simpler and equally fast.
   Revisit when per-step cost grows (complex cards, heavier encoding).

2. **Python verification incomplete.**
   Rust tests pass (`cargo test`, clippy). Python tests (`pytest`,
   `manabot.verify`) were blocked by sandbox tooling (no pytest, pip
   too old). Need to run on a properly configured host.
