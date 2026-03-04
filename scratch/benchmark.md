# Throughput benchmark: Rust vs C++

Date: 2026-03-03
Environment: macOS arm64, Apple M-series, Python 3.12, Release builds

## Workload
- 2 players (`gaea`, `urza`) using identical 40-card decks
- Deck: Mountain=12, Forest=12, Llanowar Elves=10, Grey Ogre=6
- Policy: always take action index `0`
- 500 games per seed, `skip_trivial=True`, profiler disabled

## Summary

| | C++ (avg) | Rust (avg) | Factor |
|---|---|---|---|
| steps/sec | 95,600 | 438,100 | **4.6x** |
| games/sec | 1,355 | 6,260 | **4.6x** |

## Per-seed results

### Rust

| seed | steps | time (s) | games/s | steps/s |
|------|-------|----------|---------|---------|
| 0 | 34,000 | 0.0757 | 6,600 | 448,857 |
| 1 | 32,000 | 0.0647 | 7,725 | 494,404 |
| 2 | 19,000 | 0.0374 | 13,384 | 508,589 |
| 3 | 50,000 | 0.1175 | 4,257 | 425,659 |
| 4 | 27,500 | 0.0566 | 8,840 | 486,194 |
| 5 | 63,000 | 0.1747 | 2,863 | 360,697 |
| 6 | 54,000 | 0.1324 | 3,776 | 407,783 |
| 7 | 40,500 | 0.0939 | 5,323 | 431,122 |
| 8 | 30,000 | 0.0664 | 7,534 | 452,027 |
| 9 | 55,500 | 0.1515 | 3,299 | 366,239 |

### C++

| seed | steps | time (s) | games/s | steps/s |
|------|-------|----------|---------|---------|
| 0 | 36,003 | 0.4031 | 1,240 | 89,308 |
| 1 | 35,825 | 0.3691 | 1,355 | 97,062 |
| 2 | 35,358 | 0.3686 | 1,357 | 95,934 |
| 3 | 35,625 | 0.3659 | 1,367 | 97,372 |
| 4 | 35,396 | 0.3596 | 1,390 | 98,422 |
| 5 | 34,797 | 0.3574 | 1,399 | 97,365 |
| 6 | 36,110 | 0.3721 | 1,344 | 97,034 |
| 7 | 35,805 | 0.3661 | 1,366 | 97,797 |
| 8 | 35,451 | 0.3633 | 1,376 | 97,593 |
| 9 | 35,537 | 0.3685 | 1,357 | 96,448 |

## Notes

- No seed parity between engines — different RNG implementations produce
  different games for the same seed. Step counts differ accordingly.
- C++ throughput is very consistent (~95K steps/s). Rust varies more
  (360K–508K) because game length variance is higher.
- C++ built from `main` (commit fd5ad2b) with CMake + Ninja, Release mode.
- Rust built via `maturin develop --features python` (release profile).
- Both benchmarks run on the same machine, same Python venv, same script.
