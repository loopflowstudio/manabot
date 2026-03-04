# 03: C++ Removal and Cutover

Remove the C++ backend and make Rust the only maintained runtime path.

## Finish line

All four cutover gate items from the wave README are satisfied:
1. Rust engine + binding tests green (cargo test, pytest)
2. Throughput at least parity with C++ baseline
3. Rust/PyO3 is the default and only managym runtime path
4. C++ runtime/backend path and CMake-based Python module build removed

## Starting point

Depends on 02-pytest-green completing. At that point:
- Rust engine is functionally complete (game loop, priority, combat, SBAs)
- PyO3 binding surface matches C++ pybind.cpp exactly
- Python tests pass against Rust backend
- Profiler supports baseline/compare workflows

## Work needed

1. **Throughput benchmarking.** Run the agreed benchmark against both C++ and
   Rust backends. Document results. Rust must be at least parity.

2. **C++ code removal.**
   - Remove C++ source: `managym/agent/*.{cpp,h}`, `managym/state/*.{cpp,h}`,
     `managym/flow/*.{cpp,h}`, `managym/infra/*.{cpp,h}`,
     `managym/cardsets/**/*.{cpp,h}`, `managym/main.cpp` (46 files total)
   - Remove C++ test files: `tests/agent/*.cpp`, `tests/flow/*.cpp`,
     `tests/infra/*.cpp`, `tests/state/*.cpp`, `tests/main.cpp`,
     `tests/managym_test.{cpp,h}` (10 files)
   - Remove root `CMakeLists.txt`, cmake config, pybind11 dependency
   - Clean up any C++-only CI/build scripts

3. **Build system cleanup.**
   - Maturin is the only build path for the Python module
   - `pip install -e managym` is the canonical install command
   - Remove scikit-build references if any remain

4. **Documentation update.** README references to C++ build commands
   (`mkdir -p build && cd build && cmake ..`) need updating. Remove
   C++ test commands from README testing section.

5. **Final verification.**
   - `cargo test` still green
   - `pip install -e managym && pytest tests/env/ tests/agent/`
   - Training smoke test (short PPO run)

## Risks

- Throughput regression — Rust may be slower on specific workloads (GIL
  overhead, allocation patterns). Profile before removing C++.
- Missed C++ files — grep for `.cpp`, `.h`, `cmake` patterns after removal.
  Note: C++ source is co-located alongside Rust `src/` — careful not to
  remove the Rust code in `managym/src/`.
- Third-party code depending on C++ headers (unlikely but check).

## Done when

```bash
# No C++ source remains
find managym/ -name "*.cpp" -o -name "*.h" -o -name "CMakeLists.txt" | wc -l
# => 0

# Rust tests
cargo test

# Python tests
pip install -e managym && pytest tests/env/ tests/agent/

# Throughput parity documented
```
