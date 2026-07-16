# W2-208: ContentPack clone allocation boundary

## Scope

This is focused allocation-ownership evidence for Runtime State Foundation
KR1. It shows that exact clones copy mutable match storage while retaining
immutable `ContentPack` and `CardDefinition` storage by reference. It is not a
before/after W2-179 comparison, a clone-latency result, a step or rollout
throughput result, an RSS measurement, or a branching-representation decision.

The executable identity contract separately checks that independently reset
environments, exact root/sibling forks, and every retained `RolloutPool` slot
have the same `Arc<ContentPack>` pointer, schema version, and content digest.
Mutating one branch changes only its deterministic match-state hash and facts;
the source and untouched sibling retain their facts and shared definition
handles.

## Measurement record

- Captured: `2026-07-16T01:32:56Z`
- Measurement-code revision:
  `f7d5878b6be4b2276ac5f42f50dc5f390a6982ad`
- Build profile: Cargo `release`
- Toolchain: `rustc 1.96.1 (31fca3adb 2026-06-26)`;
  `cargo 1.96.1 (356927216 2026-06-26)`
- Hardware: Apple M4 Max, arm64, 16 logical CPUs, 137,438,953,472 bytes
  physical memory
- OS: macOS 26.0.1, Darwin 25.0.0
- Focused identity/isolation contracts:

  ```bash
  cargo test --manifest-path managym/Cargo.toml --test content_pack_tests matches_and_search_clones_share_one_immutable_content_pack -- --exact
  cargo test --manifest-path managym/Cargo.toml --lib agent::env::content_pack_contract_tests::content_pack_contract_covers_env_roots_siblings_and_rollout_slots -- --exact
  ```

- Allocation command:

  ```bash
  cargo test --release --manifest-path managym/Cargo.toml --test content_pack_clone_allocations -- --nocapture --test-threads=1
  ```

The checked raw record is
[`data/w2-208-content-pack-clone-allocations.json`](data/w2-208-content-pack-clone-allocations.json).

## Method and workload

The test installs a counting global allocator that records Rust allocation
requests and requested bytes only while an explicit measurement flag is set.
The process runs one test thread. Fixture construction, pack expansion,
serialization, digesting, warmup, output formatting, and deallocation are
outside the snapshots.

The mutable fixture is the deterministic `interactive-heavy-80-v1` game after
80 fixed seeded decisions: 120 cards, 28 allocated permanent slots, and 498
committed events. Each fixture is warmed with 64 exact `Game` clones, then
1,024 `GameState` clones and 1,024 exact `Game` clones are measured
sequentially. Reference bookkeeping retains 4,096 `Arc<ContentPack>` clones in
a preallocated vector and verifies that the strong count rises by 4,096, so a
zero allocation count cannot result from skipping the workload.

The baseline contains the admitted 58 definitions and 53,802 serialized
definition bytes. The controlled expanded pack preserves those definitions and
appends 256 unused definitions with 16,384 text bytes each. It contains 314
definitions and 4,417,834 serialized definition bytes: a 4,364,032-byte
increase, above the 4,194,304-byte minimum. The two game fixtures are cloned
from one root and differ only in `GameState::content`; their base card
definition handles remain shared.

## Raw result and failure threshold

| Boundary | Baseline allocations / bytes | Expanded allocations / bytes | Expanded - baseline |
|---|---:|---:|---:|
| Retain 4,096 `Arc<ContentPack>` references | 0 / 0 | 0 / 0 | 0 / 0 |
| Clone `GameState` 1,024 times | 36,864 / 51,653,632 | 36,864 / 51,653,632 | 0 / 0 |
| Clone exact `Game` 1,024 times | 37,888 / 51,719,168 | 37,888 / 51,719,168 | 0 / 0 |

The gate fails if either Arc workload allocates anything, if the mutable clone
workloads do not allocate, if the expanded digest matches the baseline, if the
serialized payload increase is below 4 MiB, or if either mutable clone workload
has a nonzero expanded-minus-baseline allocation or byte delta. There is no
tolerance: the required allocation and byte deltas are exactly zero.

The result passes. It proves that clone allocations do not include or scale
with the controlled immutable definition bytes at this source revision. The
executable test remains the current gate; this checked record is a historical
receipt and must be regenerated if the measurement-code revision is rewritten.
