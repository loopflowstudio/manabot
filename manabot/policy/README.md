# manabot.policy

The policy contract every framework backend shares, and the host work that
does not depend on a framework. **This package never imports an array
framework** (PyTorch, MLX, JAX); it uses NumPy only
(`tests/test_package_boundaries.py` enforces it).

- `prepare.py` turns one admitted observation batch into a `Prepared` batch:
  definition rows, record-vocabulary lookups, legal-action coordinates, packed
  GRU schedules, shared-effect routes and inference column compaction. A
  backend's graph then holds only gathers, scatters and arithmetic.
- `checkpoint.py` is the checkpoint container: one safetensors file of
  canonical float32 parameters, lookup buffers and the checkpoint's
  configuration, world binding and source record. It validates the world
  binding as `manabot.model.checkpoint.load_checkpoint` does and refuses
  configurations the backends do not implement (`UnsupportedPolicy`).
- `protocol.py` states what a backend provides (`BackendPolicy`,
  `BackendLearner`, `Backend`) and the learner's per-minibatch inputs
  (`Targets`, `Settings`). Values are `[B]`.

Backends live in [`manabot/backends/`](../backends/README.md). The PyTorch
model in `manabot.model` reads `.pt` checkpoints directly and does not yet
consume this package.
