# MTG-323 plan

## Docs written (intended behavior)

- manabot/README.md — "Rollout evidence" and "Native frontier encoding" sections
- docs/development.md — local-update validation paragraph
- docs/benchmarks/local-update.md — arm comparison paragraph
- .lf/skills/qa.md — local-update search QA note

## Behavior the docs commit to

- `LocalSearchConfig.rollout_evidence` defaults to `"compact"`; `"full"` stays selectable.
- `LocalUpdateTeacher(native_encoder=True)` default; `make_player` spec
  `native_encoder` defaults to True (`flat_mc.py` `spec.get(..., False)`).
  Missing `managym.encode_observations_into` raises; no fallback.
- Compatibility (must hold after the default flips):
  - `verify_replay` parses `receipt.config_json` (local_update.py:218). v1–v3
    receipts omit `rollout_evidence`; they must be read as full, not the new
    compact default.
  - `registration.py` ~163: a registration config omitting the key is
    historical full evidence (branch_audit true). Keep that reading; compare
    against a full config when the key is absent.
  - `receipt_config_json()` still excludes the field for full, so v3 bytes are unchanged.
  - bench `--reference` path builds the reference config from
    `current.receipt_config_json()`; references predating the field need the
    full-evidence arm (or the historical arms flag defaults: compare_arms
    `evidence=args.rollout_evidence or ["full"]`, `encoders=... or ["off"]`
    must become the new defaults: compact / on).
- Tests: keep parity tests comparing both paths; add v3-receipt and
  key-omitting-registration reads-as-full checks.

## Built

- `LocalSearchConfig.from_recorded(mapping)`: a config omitting
  `rollout_evidence` reads as full. Used by `LocalUpdateReceipt.from_json`,
  registration validation and `guidance._build_policy_player` (so a registered
  player whose config omits the key still plays full evidence; bare
  `make_player` specs get the compact default).
- Bench: arms baseline is `compact/on`; per-arm `speedup` became
  `seconds_vs_default`; `--reference` uses full evidence when the reference's
  config predates the field.

## Checks

- `pytest tests/sim/test_local_update.py tests/sim/test_local_update_sampling.py tests/sim/test_puct_compatibility.py` → 52 passed (conditional_search fixture failure is a missing experiments/data file, unrelated).
- `bench_local_update.py --worlds 4 --position-seeds 42 --rollout-evidence full compact --native-encoder off on` → parity passed; full/off 2.4× default seconds.
- `bench_local_update.py --reference main --worlds 4 16` → exit 0.
- compress: renamed the parity test's full config `full_config`; bench arm name is `_DEFAULT_NAME`. Focused local-update tests → 5 passed; `bench_local_update.py --help` ok.
