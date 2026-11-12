# MTG-310 plan

Docs changed: `docs/evaluation.md` (Shared inference → **Checkpoint identity.** paragraph).

Built: `register_checkpoint` takes `role` and `artifact_prefix` (id is
`<prefix>/<digest>`, so callers don't rehash) and returns the loaded `space`;
`experiments/tla_frontier.py` and `experiments/learning_signal.py` register
checkpoints through it (code players keep fresh-default hashes). Test:
`tests/eval/test_batched_eval.py::test_registered_non_default_checkpoint_is_admitted_by_serving`
(max_facts=256 checkpoint admitted by `PolicyStore`; fresh-default hash refused).

Check: focused pytest (batched_eval -k non_default, learning_signal, experiment_files,
layer_input, local_evaluation) passed.

Compress: `artifact_id` → `artifact_prefix` (all callers used `<prefix>/<digest>`).

Other registration sites (checked at realign): `manabot/experiment/study.py`
and `attack_execution.py` take ABI hashes from the training run's
`identities`, which describe the space the checkpoint was trained with;
`experiments/mtg186_eval.py` and `elemental_grid.py` already pass the loaded
space; `card_evaluation.py` registers code players only. None needed a change.
`elemental_long_eval.py` stays frozen (see questions.md).

Remains: CI (rerun of the timed-out self-play test noted in questions.md),
then a GPU evaluation pass that admits a frontier checkpoint, run by MTG-267.
