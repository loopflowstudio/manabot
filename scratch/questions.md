# Open Questions / Assumptions

1. **No new Rust crates available offline in this sandbox.**
   - Assumed we must keep dependencies to those already in `Cargo.lock`.
   - Implemented `managym::VectorEnv` without `rayon` and without `numpy` crate bindings.
   - Current implementation writes into NumPy buffers via Python `__setitem__` calls (correctness-focused, not SPS-optimized).

2. **Threading knob is accepted but currently not used.**
   - `num_threads` is validated (`!= 0`) and kept in the API, but stepping is currently sequential.

3. **`VectorEnv.reset(seed=..., options=...)` compatibility behavior.**
   - Current Python wrapper ignores these arguments and delegates to Rust `reset_into`.
   - If deterministic reseeding per reset is required, API/design clarification is needed.
