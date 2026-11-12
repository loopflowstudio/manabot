# MTG-316 plan

Docs written for this change (the spec): manabot/README.md (Local-update
search: "Compact rollout evidence", "Native frontier encoding", "Inference
threads", "Budgets and failures"), docs/benchmarks/local-update.md
(bench arms), docs/development.md (parity tests).

Built:
- `LocalSearchConfig.rollout_evidence` full (v3, config_json bytes unchanged
  via `receipt_config_json`) | compact (v4: no hashes, unaudited session,
  `_validate_rollout_audit` reconciles counts with native counters in search
  and `from_json`).
- Registration: compact requires `branch_audit: false`; configs without the
  key keep validating.
- `managym.encode_observations_into(envs, buffers, capacity)` in
  `vector_env_bindings.rs` (step-state buffers now optional there);
  `LocalUpdateTeacher(native_encoder=...)`, `make_player` top-level key.
- bench: `--rollout-evidence`, `--native-encoder`, `--inference-threads`
  (inter-op no longer forced to 1, per the docs).

Checks: `pytest tests/sim/test_local_update.py tests/sim/test_local_update_sampling.py`
27 passed; buffer/observation/sim suites 161 passed; clippy --features python
clean; bench arms at 1–2 worlds exact parity (compact+native ≈2.1× on the
initialized opening fixture, debug-free uv build; not a claim).

Compress: frontier capacity read from the buffers (no separate counter); no
binary feature probe (managym is version-pinned); bench compares compact rows
by `replace` over the two omitted fields; the frontier checker no-ops for row
teachers. Rerun: local-update suites 27 passed; bench 4 arms at 1 world exit 0.

Remaining: gate (debug native suite in CI), a main-world 16-world measurement
on the Mac mini if a number is wanted.
