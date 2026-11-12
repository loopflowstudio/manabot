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

Realign (after main merge 5066d7b5d, which touched none of these files): the
extension (`managym/_managym.cpython-313-darwin.so`, 07:40) postdates the last
Rust edit (07:39). The Task brief asks for "cargo test in debug per CLAUDE.md";
only clippy has run, so the debug native suite is still owed (locally or CI).

Remaining: gate (debug native suite), PR to main with auto-merge per the brief
("Deliver this as a PR to main and land it"). No experiments or rentals, so a
16-world measurement is out of scope for this Task.

Concept review (headless): two concepts, matching the brief's "Defaults
unchanged; compact and native encoding opt-in". `rollout_evidence` changes
what a receipt retains, so it lives in `LocalSearchConfig`, the schema version
and registration identity. `native_encoder` changes only how frontier rows are
written, so it stays a teacher/`make_player` option outside receipts. Name kept
from the frozen source (`mirror-w13-study-frozen`). No design change; no open
product choice (evaluation opt-in stays a reversible note in questions.md).
Next: gate (debug native suite), then PR.

QA (scratch/qa.html): documented 16-world bench exits 0, exact parity,
compact+native 2.5× on the fixture; a full 95-decision game is identical under
full and compact+native (search 82 s → 28 s). Fixed: `--reference main` failed
because the reference teacher received today's config class (its config_json
gained `rollout_evidence`); it now gets its own class and matches main exactly.
Fixed: the compact README example omitted `decision_seconds` and would time out
at the 1 s default on real checkpoints. Still owed: debug native suite.
