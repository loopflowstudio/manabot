# MTG-323 plan

## Outcome

Local-update search defaults to compact rollout evidence and native frontier
encoding; full evidence and the Python row encoder stay selectable. Historical
receipts and registrations keep their meaning.

## Built

- `LocalSearchConfig.rollout_evidence` defaults to `"compact"`;
  `LocalUpdateTeacher(native_encoder=True)` and the `make_player` spec's
  `native_encoder` default to True. A missing
  `managym.encode_observations_into` raises; no fallback.
- `LocalSearchConfig.from_recorded(mapping)`: a config omitting
  `rollout_evidence` reads as full. Used by `LocalUpdateReceipt.from_json`
  (v1–v3 receipts), registration validation (`registration.py`, which still
  compares key-omitting registrations without the key) and
  `guidance._build_policy_player` (registered players omitting the key play
  full evidence). `receipt_config_json()` still drops the field for full, so v3
  bytes are unchanged.
- Bench (`scripts/bench_local_update.py`): arms baseline is `compact/on`;
  per-arm `speedup` became `seconds_vs_default`; `--reference` uses full
  evidence when the reference's config predates the field.
- Docs: `manabot/README.md` (Rollout evidence, Native frontier encoding),
  `docs/development.md`, `docs/benchmarks/local-update.md`, `.lf/skills/qa.md`.
- Tests: parity test compares full/row against compact/row, full/native and
  defaults; v3 receipt round-trips as full; key-omitting registration accepted
  with `branch_audit: true`; incomplete-row rejection runs on both encoders.

## Checks (run during implementation)

- `pytest tests/sim/test_local_update.py tests/sim/test_local_update_sampling.py tests/sim/test_puct_compatibility.py` → 52 passed (a conditional_search fixture failure is a missing experiments/data file, unrelated).
- `bench_local_update.py --worlds 4 --position-seeds 42 --rollout-evidence full compact --native-encoder off on` → parity passed; full/off took 2.4× the default's seconds.
- `bench_local_update.py --reference main --worlds 4 16` → exit 0.

## Remaining

- Publish. Editing `local_update.py` changes its implementation source hash, so
  existing registered local-update players already refuse with "source drift"
  until re-registered; this is the existing rule, not new behavior.

## Concept review (2026-11-12)

No change. There is one concept: the evidence mode controls what a receipt
keeps, never what search chooses. An omitted field has one meaning for each
kind of input. A fresh spec or constructor means the current default
(compact). A recorded config (a receipt, or a registration and the player built
from it) means full, the only mode that existed when those records were
written. `from_recorded` is the only place that reads the recorded rule; its
three callers are every recorded reader on the branch (grep of `manabot`,
`academy`, `experiments`, `scripts`). Academy replay reads receipts through
`LocalUpdateReceipt.from_json`. A new registration that omits the field also
plays full, which is the safe direction, and the README tells new registrations
to state the mode. The native encoder is not part of identity, so its new
default needs no rule for recorded inputs.
