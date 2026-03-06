# SPS: Rayon Parallel Stepping

## Problem

`RustVectorEnv` steps N games sequentially in a for loop. On an M4 Max with 16 envs, stepping is ~7% of env-only time — but that percentage was measured when Python encoding dominated at 77%. Now that encoding is in Rust and buffers use raw `PyBuffer` slices, the sequential Rust step+encode loop IS the hot path. Sequential execution leaves cores idle.

The wave goal is >100,000 env-only SPS at 16 envs (wave goal #5). Parallel stepping is the path there.

## Approach

Add Rayon parallelism to `step_into_buffers` and `reset_all_into_buffers`. Each env steps + encodes into a disjoint buffer slice in parallel, with GIL released.

### Architecture

```
Python: actions tensor → .tolist() → Rust step_into_buffers()
                                        │
                                        ▼
                                  [GIL held] extract raw *mut pointers from PyBuffer
                                        │
                                        ▼
                                  py.allow_threads(|| {
                                      thread_pool.install(|| {
                                          envs.par_iter_mut()
                                              .for_each(|env| {
                                                  step game
                                                  encode obs into buffer[i]
                                              })
                                      })
                                  })
                                        │
                                        ▼
                                  [GIL reacquired] return to Python
```

### Key implementation details

**1. Send-able buffer pointers**

The current `with_write_buffers` acquires `PyBuffer` handles and creates `&mut [f32]` slices. These slices have lifetimes tied to `PyBuffer` and aren't `Send`. For Rayon:

- Extract raw `*mut f32` / `*mut i32` / `*mut f64` / `*mut u8` pointers while GIL is held
- Wrap in a `SendSlice(*mut T, usize)` struct that implements `Send + Sync`
- Safety invariant: each env writes to a disjoint row, pointers are valid for the duration of `allow_threads`, Python can't GC the backing arrays because `ObservationBuffers` holds `Py<PyAny>` refs

```rust
struct SendSlice<T>(*mut T, usize);
unsafe impl<T> Send for SendSlice<T> {}
unsafe impl<T> Sync for SendSlice<T> {}
```

Each env gets a sub-slice at offset `env_index * row_len` — no overlap, no synchronization.

**2. Rayon with dedicated ThreadPool**

Add `rayon` to `Cargo.toml`. Create a `rayon::ThreadPool` in `PyVectorEnv::new()` with thread count = `min(num_envs, num_cpus - 1)`. Store it on the struct.

Add `num_threads` parameter to `PyVectorEnv::new()` (optional, defaults to auto-detect).

**3. No Mutex needed**

`PyVectorEnv` already owns `VectorEnv` directly. PyO3's `&mut self` guarantees exclusive access. For `allow_threads`, restructure to operate on individual `Env` instances via the inner `VectorEnv`'s env vec, rather than going through `VectorEnv::step` (which takes `&mut self` and can't be split across threads).

Concretely: add a method on `VectorEnv` that exposes `&mut [Env]` plus the shared state (player_configs, seeds, opponent_policy) needed for per-env stepping.

**4. Fused step+encode per env**

Each Rayon task does: step game → auto-reset if terminal → encode observation → write into buffer slice. The `encode_into` function is pure Rust with no GIL dependency. This avoids a sequential encoding pass and intermediate `Vec<StepResult>` allocation.

**5. Auto-reset inside parallel section**

The current `VectorEnv::step` handles auto-reset (terminal → reset → skip opponent turns → return post-reset obs). Each env carries its own seed and player_configs clone, so reset is independent per env.

### Changes by file

| File | Change |
|------|--------|
| `managym/Cargo.toml` | Add `rayon` dependency |
| `managym/src/python/vector_env_bindings.rs` | `SendSlice`, extract raw pointers, `allow_threads` + Rayon, `num_threads` param |
| `managym/src/agent/vector_env.rs` | Expose per-env stepping (e.g., `par_step_into` or public `envs_mut()`) |
| `managym/__init__.pyi` | Add `num_threads` parameter to `VectorEnv.__init__` |

No Python-side changes needed — `RustVectorEnv` already uses `step_into_buffers`.

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| `numpy` crate for buffer access | Cleaner typed array API | Extra dependency. `PyBuffer` already gives raw access and is working. |
| Sequential step + parallel encode | Simpler — only encode needs Send | Game stepping (esp. opponent turns, resets) is the expensive part. Misses most of the gain. |
| `std::thread` instead of Rayon | No new dependency | Rayon's work-stealing handles variable step times (some envs hit terminal + multi-step reset). Manual pool would reimplement this. |
| Two-pass: parallel step then parallel encode | Cleaner separation | Intermediate `Vec<StepResult>` allocation per step. Fused single pass is simpler and avoids it. |

## Key decisions

1. **Fused step+encode, not two passes.** Each Rayon task does step → encode → write. Avoids intermediate allocations and maximizes work-stealing benefit (variable step times).

2. **`PyBuffer` raw pointers, not `numpy` crate.** The `mutable_slice_from_buffer` infrastructure already works. We extract the pointer before `allow_threads` and wrap it for Send.

3. **Dedicated `ThreadPool`, not global Rayon pool.** Prevents interference with PyTorch or other Rayon users. Thread count defaults to `min(num_envs, available_cores - 1)`.

4. **Auto-reset stays in Rust parallel section.** Terminal detection + reset + opponent turn skipping happen per-env. No GIL needed.

## Scope

- In scope: Rayon parallel stepping with GIL released, Send-able buffer pointers, thread count configuration, `num_threads` API
- Out of scope: removing `AsyncVectorEnv` (sprint 03), Python wrapper changes, training loop changes, `VectorEnv.step()` compat path parallelism

## Done when

```bash
cargo test
pip install -e managym && pytest tests/env/ tests/verify/ -v
```

- All tests pass (Rust + Python, including observation parity)
- Env-only SPS scales near-linearly from 1 to core_count envs
- SPS > 100,000 at 16 envs on M4 Max (advancing wave goal: "10x env-only SPS improvement on same hardware")

## Measure

**Before (sequential baseline):**
```bash
for n in 1 4 16 64; do
    python scripts/bench_breakdown.py --num-envs $n --steps 2048
done
python scripts/bench_breakdown.py --num-envs 16 --steps 2048 --with-inference
```

**After:**
```bash
for n in 1 4 16 64 128; do
    python scripts/bench_breakdown.py --num-envs $n --steps 2048
done
RAYON_NUM_THREADS=1 python scripts/bench_ab.py --a rust --b rust --num-envs 16 --rounds 5
```

"Better" = near-linear SPS scaling up to core count. Absolute target: >100k env-only SPS at 16 envs.
