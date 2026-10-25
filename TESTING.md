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

See [validation by change area](docs/development/validation.md) for required
checks and [AGENTS.md](AGENTS.md) for repository conventions.
