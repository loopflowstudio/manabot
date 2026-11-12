# MTG-310 plan

Docs changed: `docs/evaluation.md` (Shared inference → **Checkpoint identity.** paragraph).

Built: `register_checkpoint` takes `role` and returns the loaded `space`;
`experiments/tla_frontier.py` and `experiments/learning_signal.py` register
checkpoints through it (code players keep fresh-default hashes). Test:
`tests/eval/test_batched_eval.py::test_registered_non_default_checkpoint_is_admitted_by_serving`
(max_facts=256 checkpoint admitted by `PolicyStore`; fresh-default hash refused).

Check: focused pytest (batched_eval -k non_default, learning_signal, experiment_files,
layer_input, local_evaluation) passed.
