# Open Questions / Assumptions

1. **`rayon` crate unavailable in offline sandbox.**
   - `cargo test --offline` failed to resolve `rayon` from crates.io index.
   - Implemented sprint 02 parallel stepping with scoped `std::thread`
     workers instead of Rayon so the branch remains buildable in this env.
   - If/when crates.io access is restored, we can swap the worker loop back to
     Rayon without changing the public Python API (`num_threads` + env var).

2. **`Env` is `Send`.**
   - Confirmed by compile: scoped thread workers borrow `&mut Env` across
     threads successfully.

3. **Thread count tuning in real training workloads.**
   - Implemented precedence: explicit `num_threads` > `RAYON_NUM_THREADS`
     > auto default (`max(1, min(num_envs, available_parallelism - 1))`).
   - Auto uses `std::thread::available_parallelism()` (logical cores), not
     physical cores, because `num_cpus` could not be added offline.
   - Remaining question is empirical: whether `-1 core` is still best under
     CPU inference-heavy runs, or should be adjusted per host profile.

4. **Python done-when checks blocked by host tooling.**
   - Host only has system Python 3.9 with no `pytest`, and `pip` is too old for
     editable PEP 660 installs (`pip3 install -e managym` fails).
   - Rust-side verification is complete (`cargo test`, `cargo test --features python`,
     and `cargo clippy --all-targets --all-features -- -D warnings`).

5. **Git metadata writes are sandbox-blocked.**
   - Commands that write to the worktree index (e.g. `git checkout`, commit)
     fail because the `.git/worktrees/...` path resolves outside writable roots.
