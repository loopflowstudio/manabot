# Testing

**NEVER intentionally induce load for testing.** Do not start CPU burners
(such as `yes` redirected to `/dev/null`), busy loops, stress tools, memory
pressure generators, disk/network floods, or extra workers whose purpose is
to create contention. This applies to local machines, CI and remote hosts,
including debugging, benchmarks, demos and failure reproduction.

Test load-dependent behavior with injected measurements, controlled clocks,
bounded fixtures and synchronization primitives. Measure performance using
the actual workload. Keep normal test execution focused and bounded; every
test-owned subprocess must be terminated and reaped on success, failure,
timeout and cancellation.

No test runs for five minutes: `tests/conftest.py` ends the run when a single
test reaches 300 seconds, printing every thread's stack.

CI runs the Integration groups with one pytest worker per logical core
(`pytest -n logical`, pytest-xdist), so tests run in separate processes, in no
fixed order, beside other tests. A test keeps its files under `tmp_path`,
binds no fixed port, relies on no state another test left in the module, and
does not assert on wall time that a busy neighbor could stretch. A
module-scoped fixture is built once in every worker that runs a test of that
module. Under `-n` the worker that reaches the five-minute limit exits and its
test is reported as a crash.

See [validation by change area](docs/development.md#validation-by-change-area) for required
checks and [AGENTS.md](AGENTS.md) for repository conventions.
