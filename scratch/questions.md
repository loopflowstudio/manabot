# Open Questions / Assumptions

1. **Rust numpy crate unavailable in sandboxed build**
   - Assumption: using `pyo3::buffer::PyBuffer` + validated numpy ndarray contracts is acceptable for sprint 03 in this environment.
   - Reason: network-restricted sandbox could not fetch new `numpy` crate dependency from crates.io.

2. **GIL release around vector stepping**
   - Assumption: keeping the GIL held for `reset_all_into_buffers` / `step_into_buffers` is acceptable for now.
   - Reason: `Python::allow_threads` closure requires `Ungil`; the current `MutexGuard<VectorEnv>` capture is not `Send/Ungil`.
   - Follow-up: restructure ownership/locking so stepping can run in `allow_threads` while preserving safety.
