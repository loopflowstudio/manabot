# Open questions

- `experiments/elemental_long_eval.py` (frozen) still takes `action_abi_sha256`
  from fresh defaults; it would fail serving only if a checkpoint's action
  shape differed. Left unchanged as frozen evidence; a follow-up Task if it is
  ever served on CUDA.
- Registrations for checkpoints now take `matchup_sha256` from
  `runtime_fingerprints` (`model_dump()`), which `manabot.eval.match` already
  checks against; previously `model_dump(mode="json")`.
- Compress run: `tests/eval/test_experiment_files.py::test_a_tracked_self_play_run_draws_every_named_metric`
  hit `TimeoutError: training resource wall deadline exceeded` inside a
  32-minute local batch. At gate it passed alone in 3.4 s, and the whole module
  passed in 7.6 s: the wall deadline was exceeded under concurrent local load,
  not by this change (it registers no checkpoint).
