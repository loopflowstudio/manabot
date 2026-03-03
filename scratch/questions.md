# Open questions / blockers

1. `pytest tests/env/ tests/agent/ -v` could not be run in this sandbox because
   required Python packages are unavailable and outbound network access is
   blocked:
   - `pip install ...` fails DNS resolution (`Could not find a version ...`).
   - `uv` panics in this environment (`Attempted to create a NULL object`).

2. `maturin develop --features python` is currently blocked for the same reason
   (`maturin` not installed, no network access to install it).

Assumption taken for this implement pass: validate behavior via Rust integration
tests and direct Python smoke scripts against the existing local `managym`
module, then leave full pytest + maturin verification for a network-enabled
environment.
