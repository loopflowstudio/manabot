# 01: Zero-Copy Buffers [SHIPPED]

**Status:** Complete. Shipped as part of the in-process vector env.

Buffer writes use `PyBuffer` protocol with `mutable_slice_from_buffer()` for raw `&mut [f32]` access. No `__setitem__` calls. No per-step Python allocations.

Key files:
- `managym/src/python/vector_env_bindings.rs` — `with_write_buffers`, `ObservationFieldSlices`, `WriteBuffers`
- `manabot/env/rust_vector_env.py` — pre-allocated numpy buffers, `torch.from_numpy()` views
