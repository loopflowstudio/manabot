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
  binding as `manabot.backends.torch.checkpoint.load_checkpoint` does and refuses
  configurations the backends do not implement (`UnsupportedPolicy`) and
  files that are not containers (`NotAContainer`).
- `buckets.py` chooses the sizes bucketed shapes trim a batch's variable
  axes to (`bucket_sizes`). Admission measures them on the host; each backend
  trims its own arrays.
- `returns.py` holds the returns and advantage-selection recursions over host
  NumPy rows (`transition_gae`, `selection_mask`, `quantile`,
  `terminal_distances`), bit for bit what the torch backend computes. The
  update diagnostics read them, and a backend with no better form wraps them.
- `protocol.py` states what a backend provides (`BackendPolicy`,
  `BackendLearner`, `Backend`) and one captured minibatch's constants and
  coefficients (`Targets`, `Settings`). Values are `[B]`.

Backends live in [`manabot/backends/`](../backends/README.md). The PyTorch
model in `manabot.model` reads `.pt` checkpoints directly and does not yet
consume this package.
