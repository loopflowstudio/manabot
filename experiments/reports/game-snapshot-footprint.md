# Dormant game footprint

On the M4 Max, 10,000 forks of two shared game positions added **443.3 MiB** of resident memory. Clearing the characteristics and payment caches on both roots before cloning reduced the increase to **439.8 MiB**. Physical cards already share their storage. The largest heap costs were mutable object rows at both positions and undrained observation events on the later position.

## What was measured

The engine was main at `0b1cc9cfef9c1a98430c825b9764a49237182696`. The host was an Apple M4 Max, 16 logical CPUs, 128 GiB RAM, macOS 26.7.4. Rust and Cargo were 1.98.1, target `aarch64-apple-darwin`, profile `release`. The baseline engine budget binaries and Python extension used the default allocator without benchmark instrumentation. Clone diagnostics and constructed boards used the opt-in `benchmark` feature.

There was no native `GameSnapshot` at this revision. The dormant proxy is a normal `Game::clone()` taken after replacing the root's `DerivedCache` and `PaymentCache` with their defaults. It retains legal actions, pending choices, command bindings and viewer histories under the existing search-clone contract. A full witness comparison on disposable clones checked authority, legal actions, both viewer projections, acting observation and RNG before counting. A training capture that retains recording state or copies a hot game before dropping caches will have different costs.

Both roots came from `match_fixtures::build_fixture`. The 48-decision fixture drains observation events after each decision. The 80-decision fixture leaves them pending. Both have 120 objects and two legal root actions.

| Fixture | Permanent slots | Committed events | Authority BLAKE3 | Legal surface BLAKE3 |
|---|---:|---:|---|---|
| `interactive-midgame-48-w4-v1` | 15 | 340 | `bc54c4159f3c61d5dbb368fc25f05eec4a5ec1a5efec46b6c94164dbad387ccc` | `17b492e4c9efe30cf461ec097739ea7e7eb723948b07a28c8a0539f36684f600` |
| `interactive-heavy-80-w4-v1` | 28 | 499 | `f27e14b7bea2dc858a0a8e84c564bc75f4f1438bf848fa5cc371182c2dd918a8` | `fc4d7757b6b9116dd61252ead2bfff9c5a86f9c5ad1603faff63bd332f9829fb` |

Build from the repository root and run each cache mode in a fresh process. Each invocation writes JSON Lines for both fixtures. On macOS and Linux, `--bank-mib` also samples live RSS with `ps`. These commands assume 1 GiB available for the bank.

```sh
cargo build --locked --release --manifest-path managym/Cargo.toml --features benchmark --example bench_footprint
managym/target/release/examples/bench_footprint exact --bank-mib 1024
managym/target/release/examples/bench_footprint cold --bank-mib 1024
```

Omit `--bank-mib` for sizes and clone traffic alone. Before reserving the bank, the example checks that projected inline and heap bytes fit the allowance and stay below 1.5 GiB. Leave headroom for allocator overhead.

## Inline and heap bytes

| Rust type | Release inline bytes |
|---|---:|
| Game | 5,152 |
| TrackedState | 4,072 |
| GameState | 4,040 |
| Object | 96 |
| Permanent | 168 |
| DerivedCache | 96 |

Each cell below gives **allocation requests / requested bytes** for one clone. Counting begins immediately before cloning and ends while the result is still live. Setup, hashing, destruction and the inline destination are outside the count. Field rows are independent probes of the same source and are included in their parent.

| Clone | Midgame | Heavy |
|---|---:|---:|
| Game | 38 / 22,609 | 70 / 50,090 |
| Cold-cache Game | 36 / 22,369 | 68 / 49,642 |
| TrackedState | 35 / 22,305 | 67 / 49,578 |
| Physical roster (`CardVec`) | 0 / 0 | 0 / 0 |
| Object rows | 1 / 11,520 | 1 / 11,520 |
| Permanent slots | 1 / 2,520 | 1 / 4,704 |
| Zone manager | 10 / 2,296 | 9 / 2,256 |
| Last-known object facts | 15 / 2,246 | 24 / 2,535 |
| Observation events | 0 / 0 | 22 / 24,984 |
| Derived cache | 2 / 240 | 2 / 448 |
| Payment cache | 0 / 0 | 0 / 0 |

`CardVec` is a 16-byte handle over `Arc<[Card]>`. Another Arc around that handle would share no additional card bytes. Observation delivery drains the event queue in ordinary `Env` play, so the heavy fixture is useful for a fork taken before delivery. Dropping events to reduce that number would change the next observation.

## Resident memory

One fresh process per cache mode retained alternating clones of the two roots in a vector with capacity 10,000. Each process sampled its own current RSS via `ps -o rss=` before inserting any clones and with the bank live at both requested counts. The pre-bank sample includes fixture setup, witness checks and reserved vector address space. RSS includes touched inline storage, heap allocations and allocator overhead. These two roots share immutable data and history prefixes across the bank. Independent training episodes may share less.

| Contents | Zero (KiB) | 1,000 (KiB) | 10,000 (KiB) | Increase at 1,000 / 10,000 (MiB) |
|---|---:|---:|---:|---:|
| Ordinary Game clones | 14,704 | 60,000 | 468,624 | 44.23 / 443.28 |
| Cold-cache proxy | 14,800 | 59,760 | 465,200 | 43.91 / 439.84 |

## Play and search baselines

The default Engine Budget run completed 512 seeded UR Lessons versus GW Allies games, 123,517 decisions, zero capped games, in 3.404 s (36,281 steps/s). Decision digest `6877553498864102521`. Search-64 made 40 replies per seed. All four digests matched the committed budget. The timing is one local recording.

| Search seed | Total (s) | Worst reply (s) | Ordered-score digest |
|---|---:|---:|---|
| 1 | 29.589 | 2.185 | `12271650047222149123` |
| 2 | 18.781 | 1.912 | `10117444071094495583` |
| 3 | 18.179 | 0.989 | `4954990117603543659` |

The constructed boards reached attack, block and damage decisions. Their priority forks had the same allocation costs at each size. The difference emerged when damage resolved on the board with diverse static abilities.

| Workload, sizes 4 / 8 / 16 | Duplicate creatures | Diverse statics |
|---|---|---|
| Priority fork, requested bytes | 10,908 / 13,436 / 18,492 | 10,908 / 13,436 / 18,492 |
| Priority fork, median µs | 1.54 / 1.92 / 2.17 | 1.50 / 1.75 / 2.17 |
| Damage step, median µs | 18.7 / 24.3 / 47.4 | 30.9 / 104.8 / 426.8 |
| Damage step, requested bytes | 46,302 / 89,030 / 163,181 | 101,368 / 411,933 / 1,610,621 |

The diverse-statics damage step grows about fourfold per doubling of board size, in both time and requested bytes, while the duplicate-creature board stays near linear. Its work counter shows one layer-board rebuild per creature at each size (4 / 8 / 16 boards), so the quadratic term is each rebuild re-evaluating every static source. This is the largest superlinear cell among the measured states. It is outside dormant storage and is left for a layer-rebuild change.

Both the timing and the separate allocation pass returned digest `2f90bd126a98e659c8c17ca4f09e58aab97eb270055e9dc80591c8370e04d99b`. Reproduce these runs from the repository root.

```sh
cargo build --locked --release --manifest-path managym/Cargo.toml --example bench_steps --example bench_search
uv run --no-project scripts/engine_budget.py sample --candidate managym/target/release/examples --artifact scratch/budget-recording
cargo run --locked --release --features benchmark --example bench_states --manifest-path managym/Cargo.toml -- duplicate-creatures diverse-statics --sizes 4,8,16 --layers clone,step
cargo run --locked --release --features benchmark --example bench_states --manifest-path managym/Cargo.toml -- duplicate-creatures diverse-statics --sizes 4,8,16 --layers clone,step --allocations
```

## Four-worker collection

One CPython 3.13.15 process ran four persistent `ShardedVectorEnv` worker threads over 16 streams starting at seed 24. All streams reset to the authored UR Lessons versus GW Allies matchup and took 128 buffered steps each. Python's `Random(24)` chose uniformly from each row's legal action mask. The timer covered `step_into_buffers`, including the worker barrier and native observation encoding. Setup, action selection and hashing ran outside the timer.

The 2,048 decisions produced SHA-256 `6e1db496eca5346db1aa0546fa9145767bf20a5e28116361e3eb93b9211224c1` over the action rows, masks, chosen indices, terminations and rewards on every batch. The standalone baseline was 36,470 steps/s. This measures the CPU producer prefix; model inference and training updates are outside its scope.

Reproduce the prefix with the checkout's release extension in a fresh process.

```sh
uv run --no-sync python scripts/bench_sharded_collection.py --expect-allocator system
```

## Allocation candidates

The two widest optional continuation fields occupy 1,280 bytes for `suspended_decision` and 424 for `trigger_placement`. Both were absent at the two bank roots. A separate probe read the two fields before each of 160 ordinary decisions on curated seeds 0–7, using the budget's legal choice sampler. Across 1,280 decisions, 106 held a suspended resolution, 17 held trigger placement, and zero held both (count/choice digest `7640719225026218502`). Replacing both inline options with pointers would save up to about 16.1 MiB over 10,000 absent pairs before padding. That upper bound is 3.6% of the ordinary bank increase. One field was occupied at 9.6% of sampled decisions. Keep both fields inline to preserve their public Rust types and avoid an extra allocation when populated. The sampled cast-permission and effect tables requested zero heap bytes at both roots. Boxing them has little measured upside.

On 64-bit targets the engine checks six inline sizes at compile time. The ceilings are 96 bytes per `Object`, 168 per `Permanent`, 4,096 for `GameState`, 4,128 for `TrackedState`, and 5,248 for `Game`. The `DerivedCache` ceiling is 96 bytes in release and 104 with debug assertions. The debug scan count brings the observed `Game` size to 5,160 bytes. The clone probe measures heap costs separately.

## Matched allocator trial on macOS

A temporary Cargo feature selected `mimalloc::MiMalloc` as Rust's global allocator (mimalloc 0.1.52, libmimalloc-sys 0.1.49, default features disabled). Each variant had its own optimized binaries and CPython extension. Both used the same base engine. The Engine Budget sampler ran mimalloc, system, system, mimalloc on this host. The four-worker prefix ran separately in that order, in fresh processes. Values aggregate the two durations per variant using the budget harness's convention. The committed reference supplied the expected digests.

| Measure | System allocator | mimalloc | Change |
|---|---:|---:|---:|
| Headless decisions/s | 37,465 | 44,503 | +18.8% |
| Total search reply time | 64.751 s | 54.370 s | −16.0% |
| Worst search reply | 2.124 s | 1.806 s | −15.0% |
| Four-worker buffered decisions/s | 38,710 | 49,704 | +28.4% |

Total search times in execution order were 53.886, 64.565, 64.937 and 54.855 s. The corresponding four-worker times were 40.41, 55.06, 50.75 and 42.00 ms. Each mimalloc sample beat both system samples on both workloads. The budget comparison passed, including every search score digest and the step digest. All four collection digests also matched. All 11 debug `match_state_hash_tests` passed with the allocator feature. These runs establish an allocator effect on this host. They contain no lock-wait measurement from which to attribute the improvement specifically to contention.

Routing the counting allocator through mimalloc reproduced every inline size, per-field request count, requested byte total and fixture witness. Its ordinary bank read 15,344 / 56,464 / 460,528 KiB at 0 / 1,000 / 10,000 clones (increases of 40.16 / 434.75 MiB). The cold bank read 14,784 / 56,112 / 456,768 KiB (40.36 / 431.62 MiB). Each RSS point has one reading per mode. At this bank size the throughput gain came with a small observed reduction in resident memory.

The native Cargo feature `mimalloc` retains the allocator used in this trial. The system allocator remains the default. Build the Rust budget examples with `--features mimalloc` for timing, or build `bench_footprint` with `--features benchmark,mimalloc` for request counts. Its opening JSON line identifies the backing allocator.

Both release wheels reported the expected allocator with counting disabled and reproduced the 2,048-decision digest above. Requesting mimalloc with the system wheel loaded stopped the script before timing. These short macOS samples support the opt-in feature. A Linux training default needs matched Linux measurements. CI compares the default engine with the committed Engine Budget reference. [Allocator build and collection commands](../../managym/README.md#native-allocator)
