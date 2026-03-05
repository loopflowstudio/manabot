# 01: Rust VectorEnv

## Finish line

A Rust `VectorEnv` struct owns N games, steps them in a single PyO3
call, and returns `Vec<Observation>`. Python wrapper replaces
`AsyncVectorEnv`. Training works. No zero-copy yet — that's sprint 03.

## Changes

### 1. `managym/src/agent/vector_env.rs`

```rust
pub struct VectorEnv {
    envs: Vec<Env>,
    num_envs: usize,
}

impl VectorEnv {
    pub fn new(num_envs: usize, seed: u64, skip_trivial: bool) -> Self;

    pub fn reset_all(
        &mut self,
        player_configs: Vec<PlayerConfig>,
    ) -> Vec<(Observation, InfoDict)>;

    pub fn step(
        &mut self,
        actions: &[i64],
    ) -> Vec<(Observation, f64, bool, bool, InfoDict)>;

    /// Reset only the envs that are done, using fresh seeds.
    pub fn auto_reset_done(
        &mut self,
        dones: &[bool],
        player_configs: &[PlayerConfig],
    );
}
```

Each `step` call processes all N envs sequentially (parallelism comes
in sprint 04). Games that terminated are flagged; the caller decides
whether to auto-reset.

### 2. PyO3 bindings

`managym/src/python/vector_env_bindings.rs` — expose `VectorEnv` to
Python. `step()` takes a list of ints, returns a list of tuples.

### 3. Python wrapper

`manabot/env/vector_env.py` — thin wrapper that:
- Creates the Rust VectorEnv
- On step: calls Rust, encodes observations using existing Python
  `ObservationSpace.encode()`, stacks into batch tensors
- Handles auto-reset (reset done envs, return post-reset obs)
- Implements the same interface the trainer expects

### 4. Trainer integration

`manabot/model/train.py` — swap `AsyncVectorEnv` for `RustVectorEnv`
behind a flag. Both paths coexist until sprint 03 validates parity.

## Verification

- `cargo test` passes
- Step 0-2 verification ladder passes with new env
- SPS improves (even without zero-copy, eliminating subprocess
  overhead helps)
- Observation output matches old path on 100 random game states

## Done when

Training runs with `RustVectorEnv`. Same learning curves as
`AsyncVectorEnv` (within noise). No regressions.
