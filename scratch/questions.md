# Open questions

- `experiments/elemental_long_eval.py` (frozen) still takes `action_abi_sha256`
  from fresh defaults; it would fail serving only if a checkpoint's action
  shape differed. Left unchanged as frozen evidence; a follow-up Task if it is
  ever served on CUDA.
- Registrations for checkpoints now take `matchup_sha256` from
  `runtime_fingerprints` (`model_dump()`), which `manabot.eval.match` already
  checks against; previously `model_dump(mode="json")`.
