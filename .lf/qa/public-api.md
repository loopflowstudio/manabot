Read when the change touches the published library API: `__all__` in `PUBLIC_MODULES`, `managym/__init__.pyi`, `scripts/public_api.py` or `tests/public_api.json`.

- Contributor path: `uv run python scripts/public_api.py check` (under 1 s).
  Drive it with real temporary edits and revert each with `git checkout`:
  rename a required parameter, drop a name from `__all__`, drop a config field,
  and edit `managym/__init__.pyi`. Drop a field no validator names
  (`MatchConfig.decklists`); dropping `hero_sideboard` fails at import inside
  Pydantic instead. Renaming a `@field_validator` method must still match. Each should print `BREAKING …` and make
  plain `record` refuse. An optional parameter prints `ADDED …` (check exits 1
  until `record`).
- Deprecation lifecycle: rename a published class, convert **every** caller
  (`grep -rlw <Name> manabot experiments scripts tests --include='*.py'`;
  quote the glob in zsh), and keep the old name as a module `__getattr__`
  `DeprecationWarning` alias still in `__all__`. Run `record` under
  `-W error::DeprecationWarning` to prove no internal caller still uses the
  alias. Removing the alias on 0.2.x gives `DEPRECATED_REMOVED`; after bumping
  the root `pyproject.toml` to `0.3.0`, plain `record` accepts it and resets
  `line`/`breaks`. Bumping the version makes uv rebuild the editable install;
  that's harmless and reverts on the next `uv run`.
- Opaque handles: renaming a field of `Collector` (or `Prepared`, `Prefix`,
  `MemoryArchive`) must still print `public API matches`.
- Installed consumer: build `uv build --wheel --package manabot -o .runs/<dir>`
  and pair it with an engine wheel that holds the `.so`. A
  `managym/target/wheels/*cp313*` wheel of about 40 KB has no extension, so
  check with `unzip -l … | grep .so`; the cp312 wheel usually holds it, so a
  `--python 3.12` venv covers the second interpreter too. From that venv,
  `venv/bin/python -I <checkout>/scripts/public_api.py check` compares the
  installed modules with the snapshot (it reads the stub and `pyproject.toml`
  from the checkout). Never `uv run --python 3.12` in the checkout: it replaces
  the shared `.venv`. Install both into a fresh `/tmp` venv with
  the CPU Torch index (`--extra-index-url https://download.pytorch.org/whl/cpu
  --index-strategy unsafe-best-match`). Run everything from `/tmp` with
  `python -I`, because the checkout's cwd shadows the installed `managym`.
  Walk `pkgutil.walk_packages(manabot.__path__)`, collect modules defining
  `__all__` and `hasattr` every name with `DeprecationWarning` ignored (the
  `0.3.0` aliases warn on access). Expect the 15 `PUBLIC_MODULES` plus
  `manabot.sim.search.flat` and managym's own modules seen through the
  `manabot.managym` alias. Extract the first
  ```python block under `## Library API` in `manabot/README.md` (it stops before
  `## Local-update search`, whose example writes to a checkout-relative
  `.runs/`) and run it with `-W error::DeprecationWarning`; it prints
  `2 1.0 <value>` in about 4 s.
