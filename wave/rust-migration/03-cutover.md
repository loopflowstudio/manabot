# 03: C++ Removal and Cutover

Remove the C++ managym runtime and make Rust the only maintained backend.

## Finish line

- No `.cpp` or `.h` files in `managym/`
- No CMake, scikit-build, or pybind11 references in the build system
- `pip install -e managym && pytest tests/env/ tests/agent/` still passes
- Throughput is at least parity with C++ baseline on agreed benchmark

## Starting point

Requires 02-pytest-green to be complete (Python tests passing against Rust).

C++ source files still exist throughout `managym/`:
- `managym/agent/` — action_space, env, behavior_tracker, observation, pybind
- `managym/flow/` — game loop, turns, priority, combat, SBAs
- `managym/state/` — cards, players, zones, mana
- `managym/cardsets/` — card implementations
- `managym/infra/` — logging, profiling, info_dict

## Work needed

1. **Delete C++ source.** Remove all `.cpp`, `.h` files and any remaining
   C++ build artifacts (CMake fragments, pybind11 references).

2. **Clean up build config.** Ensure `managym/pyproject.toml` has no
   residual C++ toolchain references. Verify maturin is the only build path.

3. **Throughput verification.** Run the profiler baseline comparison to
   confirm Rust is at least as fast as C++ on the standard benchmark.
   Use `export_profile_baseline` / `compare_profile` through the Python API.

4. **Update documentation.** README references to C++ build commands
   (`mkdir -p build && cd build && cmake ..`) need updating.

5. **CI update.** Verify the CI workflow doesn't reference any C++ toolchain
   steps. It currently only runs Rust + Python gates, which is correct.

## Risks

- Deleting C++ before throughput parity is verified makes it harder to
  benchmark. Consider recording a C++ baseline before removal.
- Some Python code may import or reference C++ internals (logging config,
  profiler hooks). Grep for pybind11, CMake, and C++ file references.
