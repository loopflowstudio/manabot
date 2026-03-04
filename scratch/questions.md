# Open Questions / Follow-ups

1. Verification gap: this sandbox lacks Python test deps (`pytest`, `gymnasium`) and `uv run` panics under current restrictions, so full `pytest tests/ -v` could not be executed here.
2. Validation gap: the 100+ game truncation-warning check (and the larger Rust-first overflow/A-B protocol in the design doc) was not run in this environment.

Assumption made: proceed with code + test updates and leave runtime validation for CI or a fully provisioned local dev environment.
