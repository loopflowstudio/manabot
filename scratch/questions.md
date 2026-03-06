# Open Questions / Assumptions

1. **`std::thread` instead of Rayon.**
   Rayon crate was unavailable offline. Parallel stepping uses scoped
   `std::thread` workers. Public API (`num_threads` + `RAYON_NUM_THREADS`
   env var) is unchanged — swapping to Rayon later is mechanical.

2. **Python verification incomplete.**
   Rust tests pass (`cargo test`, clippy). Python tests (`pytest`,
   `manabot.verify`) were blocked by sandbox tooling (no pytest, pip
   too old). Need to run on a properly configured host.
