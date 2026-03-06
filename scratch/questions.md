# Open Questions / Assumptions

1. **`rayon` crate availability.**
   - Sprint 02 requires adding `rayon` as a dependency. If the sandboxed build
     environment doesn't have it cached, this will fail. Fallback: vendored
     dependency or `std::thread` with manual work distribution.

2. **`Env` is `Send`.**
   - Assumed based on code inspection (no `Rc`, no `Cell`, no raw pointers).
   - If `Env` is not `Send`, Rayon parallel iteration won't compile. Fix would
     be to audit the `Game` struct for non-Send fields.

3. **Thread count default.**
   - Defaulting to `min(num_envs, available_cores - 1)`. The `-1` leaves a core
     for Python/PyTorch. This may need tuning on training machines (where PyTorch
     uses multiple cores for inference).
