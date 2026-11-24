Read when the change touches packaging, releases or installed use (wheels, `scripts/release`, the clean-machine proof).

- First-time clone: copy tracked plus untracked non-ignored files
  (`git ls-files -co --exclude-standard`) to an empty `/tmp` directory and run
  `ACADEMY_CLEAN_API_PORT=… ACADEMY_CLEAN_FRONTEND_PORT=… ./scripts/verify-clean-machine`
  there; it builds cold, runs the browser proof and writes
  `.cache/clean-machine/proof.json` (`launch.elapsed_to_playable_ms`). Shared
  uv/Cargo/npm download caches stay warm locally, unlike the CI reference.
- Library consumers: follow `docs/development.md#build-and-verify` into an
  ignored `.runs/` output (the engine staging is `<staging>/managym`), then
  `uv build --wheel --python 3.12 <managym sdist>` for the other interpreter
  (reuse a `CARGO_TARGET_DIR`). In a fresh `/tmp` venv with PYTHONPATH unset and
  a PATH holding only a `uv` symlink plus `/usr/bin:/bin` (no Cargo), install the
  engine wheel alone and run `scripts/release/engine_smoke.py` and the
  `docs/engine-python.md` commands verbatim; then add both wheels at the minimum
  Torch/Etude pins and run a copy of `scripts/release/smoke.py` (CPU training,
  about 20 s). Also `uv init` a project and `uv add` both `file://` wheels for
  latest resolution. Returning users: install PyPI `manabot==0.2.0` in a
  `--seed` venv and upgrade with
  `uv run --no-project --python <venv>/bin/python -m pip`
  (wheel paths plus `--find-links`
  and the CPU Torch index), since uv's one-step upgrade hides file-ownership
  problems that pip's install order exposes.
- Release installed-Academy proof (`scripts/release/academy_smoke.py`, the
  Release packages job): `./scripts/verify-installed-academy` builds the staged
  sdist, its wheel, a no-build-tools venv, the smoke and the wheel Playwright
  spec (about 10 minutes under load). It deletes its directory on exit; to rerun
  an older smoke against the same wheel, copy the script into `.runs/` with the
  `trap` line replaced by `echo $proof_dir`. Socket readers must skip
  `work_status` and early accepted `command_outcome` receipts.
- Installed Academy (`managym[academy]` wheel): build as above, then in a fresh
  `/tmp` directory `uv venv --python 3.13 --seed` (without `--seed` there is
  no `pip`) and
  `uv run --no-project --python <venv>/bin/python -m pip install '<wheel>[academy]'`,
  with `node`/`npm`/`cargo`
  shadowed by failing stubs on PATH. Re-running that on an already installed
  identical wheel file exits 0 silently and skips the extra; install by name
  with `--find-links <dist>` instead. When the candidate's version is already
  on PyPI, `uv run --no-project --python <venv>/bin/python -m pip install
  --find-links <dist> manabot` takes PyPI's copy (no
  `managym.academy.bots` entry point); `--force-reinstall --no-deps` the local
  wheel. Never patch files inside the installed `managym`: manabot's training
  refuses a changed source bundle. Supervise `managym academy` from a Python
  `Popen(start_new_session=True)` that writes `p.wait()`, delete the old log
  before relaunching, and wait for its `ACADEMY_PLAY_READY`. `uv run etude
  train manabot/recipes/local.py` with `ETUDE_HOME` in that directory gives
  trained opponents under `track/<experiment>/out/policies/` within a minute. Another project's server may hold `*:8000`; use
  your own port.
  With a current installed policy, set `ACADEMY_PLAY_CHECKPOINTS=<absolute path>`
  before launch. Read `/api/opponents` for its digest, then open
  `/?hero=ur_lessons&villain=gw_allies&opponent=<digest>` for the local recipe's
  bound decks (Elemental UR/GW are different lists and correctly refuse it).
  Click Start game, then the legal Play Island offer and Pass Turn. Wait until
  an action offer is enabled and automation has settled before recording
  `main[data-frame-revision]`; reload must restore that revision. A merely
  visible offer can still belong to an automated pending transition.
- Python library changes: build manabot with `uv build --wheel --package
  manabot -o .runs/<dir>`. When the checkout has unreleased engine changes,
  stage the engine with scripts/release/prepare.py as in docs/development.md
  and build its wheel from the staged managym directory. Install both local
  wheels in the fresh venv, including the evaluation extra for corpus SQL.
  The PyPI engine can carry the same version and lack modules used by the
  checkout. Run the user script from /tmp with
  `uv run --no-project --python <venv>/bin/python python -I <script>`.
- Standalone statistics/recording: stage and install an engine-only wheel into a
  fresh venv, then run the two Python examples in `docs/engine-python.md` outside
  the checkout. Play the same seeded authored game with neither option, statistics
  only, recording only and both; compare observation digests and summary counts.
  Exclude `generation` from cross-Env summary comparison: it intentionally names
  each distinct game. Check terminal drain-once and replaced partial summaries.
  Copy a journal and run `uv run --no-project --python <venv>/bin/python python -m
  managym.replay <copy> --seconds 30 --statistics`; confirm source bytes remain
  unchanged. Empty/missing/corrupt inputs must fail, and a torn final append must
  warn while recovering the last complete boundary. `scripts/release/engine_smoke.py`
  additionally checks that manabot, Torch and Etude are absent.
