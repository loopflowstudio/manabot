Never add "Co-Authored by:" tags in commits

## Naming

- **Manabot** — the whole project.
- **manabot.manabot** — the training system within the project: agents,
  models, learning, search, and evaluation. You train *a manabot*.
- **manabot.academy** / **Academy** — the web app for playing Magic against
  manabots, including play, replay, and Study.
- **managym** — the rules environment and authoritative game engine.
- **Etude** — the separate experiment library at
  [loopflowstudio/etude](https://github.com/loopflowstudio/etude), distributed
  on PyPI as [etude-sdk](https://pypi.org/project/etude-sdk/). It provides
  experiment launching, tracking, checkpoints, and reporting; it is not the
  name of this project or its play app.

The dotted names above identify project components, not a promise that those
Python import paths exist in every historical checkout. In this checkout,
training lives in `manabot/` and the Academy server still lives in the legacy
`etude/` directory. The SDK also imports as `etude`; that namespace collision
must be resolved during integration, not hidden by changing documentation.
Using the SDK does not replace managym or the learning algorithms.

Older material calls this project Etude or Etude Fantasia. Preserve historical
quotes, frozen experiment IDs, receipts, schemas, artifact paths, W&B history,
and actual commands/environment names (including `ETUDE_PLAY_*`) until their
implementations migrate. Use the current names in new prose. Documentation
changes alone do not establish SDK integration or authorize experiment spending.

## Style

- Package style guides live in each package README (`manabot/README.md`,
  `managym/README.md`). Avoid transient comments that denote changes; pay
  attention to file headers and README content; propose small, iterative
  changes.

### Python structure and documentation

- Design around the domain's data structures and public APIs. Keep the core
  types and entry points easy to find; separate orchestration, learning rules,
  and persistence when they have different responsibilities.
- Give substantial modules a short opening explanation of their purpose,
  principal types and entry points, and non-obvious ownership or data flow.
  Use section comments to group related code when that helps navigation.
- Explain intent, invariants, units, tensor shapes, and algorithmic choices
  beside the code that depends on them. Training code should explain what an
  estimator measures, where gradients flow, and which samples a loss uses.
  Cite a method when claiming to implement it; document meaningful departures.
- Keep docstrings concise, but include non-obvious contracts, side effects,
  and failure behavior. Do not repeat signatures in `Args:` / `Returns:`
  boilerplate or narrate obvious statements. Update comments with behavior.
- Put imports at the top; use explicit imports and underscore-prefixed private
  helpers. Keep one implementation and one owner for each fact; use git for
  old versions rather than `_new`, `_old`, or `_backup` variants.

### Python typing

- Write fully typed Python: annotate parameters and return values on all
  functions and methods, including private helpers and tests; use `-> None`
  where appropriate. Type model fields and collections with their element
  types. Local variables can rely on clear inference.
- Represent structured domain data with named types: dataclasses, existing
  Pydantic models, `TypedDict`, or small protocols as appropriate. Avoid
  passing unstructured dictionaries or positional tuples across API boundaries.
- Prefer precise unions, literals, and callable signatures to `Any`. At
  untyped library, JSON, or native-extension boundaries, validate or narrow
  values once and expose a typed interface. Keep necessary `Any`, casts, and
  type-checker suppressions local, with a reason when it is not evident.
- Use Python 3.12 forms (`list[T]`, `T | None`). A nullable annotation means
  `None` is a real possible value; a default determines whether an argument
  can be omitted. Return `None` for expected absence only when the API defines
  it; do not disguise failures as empty results.
- Apply these conventions to new and changed code incrementally. Do not claim
  repository-wide type safety without a passing configured type check, or
  weaken types merely to silence one.

## Commands

- **All Python commands run through uv.** This repo's environment is
  uv-managed; bare `python`, `uvicorn`, `pytest`, `maturin` etc. will miss the
  venv or pick the wrong interpreter. Always `uv run <cmd>` (or
  `.venv/bin/<cmd>` when a script must reference the interpreter directly).
  Never emit an un-uv'ed Python command in docs, scripts, or instructions.
- Python is pinned to 3.12 (PyO3 caps at 3.13; the venv is 3.12). Fresh venvs:
  `uv venv --python 3.12`.
- After changing Rust under `managym/src`, rebuild from the repository root so
  uv uses the pinned root environment instead of creating a nested venv:
  `uv run maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`,
  then place the cp312 `.so` from the wheel at
  `managym/_managym.cpython-312-darwin.so`.
- Play against the bot: `./scripts/play` (installs locked local dependencies,
  starts backend + frontend, and stops both on Ctrl-C).

## Testing

- Remote credential tests require the `artifacts` extra. Validate dependency
  changes in an isolated environment so locally installed extras cannot hide
  missing CI dependencies: `uv run --isolated --extra dev --extra artifacts pytest tests/remote/ -q`.
- **CI runs `cargo test` in debug, so validate in debug before landing.**
  `cargo test --release` alone is not enough: the engine guards its invariants
  with `debug_assert!`, which compiles out of release entirely, so a test can
  pass green in release and still fail CI. `create_token` is the live example —
  it `debug_assert!`s that the name resolves to a token definition, so passing
  an ordinary card name is release-silent and debug-fatal. Clippy does not
  catch this either.
