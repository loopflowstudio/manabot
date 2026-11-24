Read when the change touches training (`manabot/training`, recipes, models, backends, collection or decision memory).

- Model or training-path changes: use `uv run etude train manabot/recipes/local.py`
  for the local CPU smoke. Inspect update rows for per-phase collection and
  learning seconds: `etude.train.read.read(<home>/track, <experiment>,
  <run>).points(metric, clock="updates")` (`.metrics()` lists names). To try a
  learning field the recipe does not expose, copy `local.py` into `.runs/`
  with config fields that `model_copy` `settings(config).learning`; such a
  single-job file refuses `--seed` other than 0. When saved training state gains a field, check that
  emitted `coordinates/` values are scalar numbers. When adjusting input or
  embedding scales, perturb the feature at initialization and measure the
  resulting change in the full object token with its usual streams enabled.
  For installed-wheel QA, `scripts/release/smoke.py` trains one policy on the native
  lethal-target root through `etude train`, reloads its
  `out/policies/00000002-raw.safetensors`, and discovers that exact path
  through `managym.academy.bots`. Run from outside the checkout with
  `uv run --no-project --python <venv>/bin/python <script>`. Play
  checkpoints with `uv run manabot sim --hero-deck-file <f> --villain-deck-file
  <f> --set sim.hero=<checkpoint.safetensors>
  --set sim.villain=<checkpoint.safetensors> --set sim.num_games=10 --set sim.num_threads=1`. The deck files must equal the
  checkpoint's `world_binding["setups"]`, written as `N Name` lines with a
  `Sideboard` section, because the default sim match is refused. Fixed seeds
  make equal win, timeout and mean-step counts a check that every sampled
  action matched. Ten games took about two minutes on a loaded laptop. Sim
  logs a model failure as `Error getting action`, plays a random move and
  still prints win rates: count those lines before trusting a result. The
  regime's decks are `run.json` `regime.match` (`hero_deck`, `hero_sideboard`).
- MLX backend (`manabot.backends.mlx`, Apple Silicon with the `mlx` extra only):
  run it on the Mac that owns the measurement, not on a shared laptop. The
  published `experiments/data/mlx_pilot/capture/initial.pt` is now refused by
  the parity CLI's world binding (written at `b62ac3127`). Make a current
  capture instead with `manabot.backends.torch.capture.capture(out, regime,
  seed, experiment=...)`, using the regime of
  `tests/model/test_backend_capture.py` with `hidden_dim=384`,
  `attention_layers=8`, `num_attention_heads=8`, 8 streams and 32 transitions
  (about 2 minutes on CPU; `hidden_dim=64` takes 20 s). Convert its
  `initial.pt` with `uv run python -m manabot.backends.convert`, load it with
  `resolve("mlx").load_policy` and step a `VectorEnv` with `MatchConfig(world=WORLD_SETUP)`,
  preparing each batch with `policy.prepare(obs, inference=True)` and evaluating
  with `policy.evaluate` (logits `[B,A]`, values `[B]`); untrained-policy argmax
  differences are exact ties. `uv run python -m manabot.backends.parity <initial.pt>
  --capture <capture dir> --backend mlx --precision <dtype> --minibatches 4`
  takes seconds; `--json` keeps per-minibatch `loss_torch`/`loss_backend`, which
  show when a relative-loss failure comes from a near-zero loss.
- Tracked MLX training (baseline, local, Elemental with `backend: "mlx"`):
  `uv run --extra mlx etude train manabot/recipes/local.py --config
  '{"backend":"mlx","device":"mps"}' --until updates=1` takes about 11 s on an
  M4 Max under load, then `--resume --until updates=2`; Elemental with
  `{"updates": 2, "streams": 4, "backend": "mlx", "device": "mps"}` about 27 s.
  Cross-read every `out/policies/*.safetensors` of a Torch and an MLX run on
  both `resolve("torch")` and `resolve("mlx", precision="float32")`: encode a
  scalar `managym.Env` observation with `encode_observation(space, raw)` and add
  the batch axis yourself (`{k: v[None]}`), or admission refuses
  `known_library`. Agreement was below 1e-7 on 153 decisions. Seat an MLX-written
  Elemental policy in Academy with `--policy`, `--deck elemental_ur
  --opponent-deck elemental_gw`. For Ctrl-C on a pty, poll the master with
  `select` (the run prints nothing between "local" and its end, so a blocking
  read never sends `\x03`). `manabot training-report` hides `local` runs
  (kind `smoke`) until "Include labeled benchmarks…" is checked.
  No shipped recipe exposes `precision`. For bf16 MLX, subclass `LocalConfig`
  in a `.runs/` recipe with `precision` (and optional `pretrained`) fields and
  `model_copy` them into `settings(config)`. bf16 needs MPS, and `SelfPlay`
  refuses it on CPU inside `track`. A resume with a changed precision must
  refuse with `run_identity_conflict`. Read `execution/precision` from the
  run's `metrics/*.permanent.json` labels.
- Training guide (`docs/training-loop.md`): run its example command verbatim
  (about 20 s on CPU), save and reload the policy as it says, call both
  maintained rules directly on a collected batch, and compare the tables'
  signatures with `inspect.signature`.
- Training recipes: run `uv run etude train manabot/recipes/local.py --experiment smoke`.
  Inspect `~/.etude/track/smoke/` (or `$ETUDE_HOME/track/smoke/`):
  `run.json`, logged rows, `out/state/current.json` and the save folder it
  names (`state.json` with the framework state files), and
  immutable `out/policies/`. A second new run with that name must refuse.
  For a planned stop, use a larger config update count with `--seconds 1`;
  expect exit 75, then `--experiment smoke --resume` to finish the same run. To extend a
  finished run, use `--experiment smoke --resume --until updates=N`; without a later target
  it must refuse. `etude continue --run RUN --until updates=N` makes a child
  run for a launched parent and leaves the first unchanged.
  Device and other recipe fields are passed with `--config JSON`; the seed
  is the run's `--seed`, not a config field. Check raw/EMA checkpoint tensors
  with matched small inputs. Fetch artifacts through Doppler without printing
  credentials. `manabot train`, its `--out` identity and `state.pt` are retired.
  After a train and a `--resume`, run `uv run --extra reports manabot
  training-report --source $ETUDE_HOME/track --out <f>.html` and read the
  page: learner transitions and transitions per active second must be numbers,
  not "unavailable" (they come from the loop's declared clocks). Screenshot it
  with the `academy/ui` Playwright script above (`npm ci` first in a fresh
  worktree). A run without a `training_seconds` clock, such as belief's
  sampler, honestly shows "unavailable" with four warnings.
  Set `ETUDE_HOME` to an ignored `.runs/` folder so QA runs stay out of
  `~/.etude`. `etude smoke experiments/elemental.py --keep --revision HEAD
  --updates 3200 --hours 11.9 --usd-per-run 9` trains for about 3 minutes, then
  evaluates for about 30 minutes on a laptop (600 one-game matches). A Python
  error in a game fails smoke eval within seconds, naming the error; only native
  failures become invalid rows, so still read the eval rows:
  `etude.eval.query.statsdb("elemental", "<kept>/track")`, then group `games`
  by `unit`, `valid` and `failure` (duckdb over
  `<kept>/track/elemental/*/blocks/*/games.parquet` also works; there is no
  pyarrow). The `failure` column is omitted when every value is null. Rerun only eval and report with
  `--step eval --step report --source <kept>/track`. That eval-only report has
  no training runs, so its heuristic panels stay empty; for the full picture,
  copy the training and eval run folders into one track and run
  `etude report experiments/elemental.py --source <it> --out <dir>`. Local eval
  evidence goes to `.runs/elemental/arena/<match-key>/attempt-*`; a retry adds
  an attempt beside earlier failed ones, so nothing needs deleting. To see a policy from `etude train` in Academy, start
  `./scripts/play` with the same `ETUDE_HOME`; discovery searches Etude's local
  track.
- TLA Limited decks (the MTG-282 pool corpus) overflow default observation
  capacities within a few dozen games; copy the `observation` block and the
  `game_errors` policy (`recover_capacity: true`) of
  `experiments/regimes/tla-frontier-c5fba4-w64.json`; `experiments/tla_frontier.py` trains that record on the current loop.
- To check combat metrics and their Etude panels, run the five-update
  `probe-demo` in [the training guide](../../docs/training.md#combat-probe) with
  `ETUDE_HOME` under `.runs/`. Read the update row with
  `updates_from_run(read(<home>/track, "probe-demo", "probe-demo"))`.
  Render it with `uv run etude report manabot/recipes/local.py
  --experiment probe-demo --source <home>/track --out <dir>`. Locate the
  attack and block panels by title when taking a Playwright screenshot.
  For chunk-size parity on the same positions and weights, run
  `tests/training/test_combat_probe.py`.
- Training game journals (`diagnostics: detailed`): run `etude train manabot/recipes/local.py --experiment NAME --config '{"updates": 200, "diagnostics": "detailed"}'` from a supervised `Popen(start_new_session=True)`, poll `game-journals/operation.json` and `os.killpg(..., SIGKILL)` while `collection` is pending after update 1; copy `vector-N.jsonl` and replay it with `manabot.training.state.game_errors --attempt 1`. Kills land at `ready` boundaries; a pending native step is reachable only through the `os._exit` child fixtures. Compare on/off runs by safetensors tensor bytes (metadata differs by `run_id`; the dev venv has no `safetensors` package, so read the little-endian u64 header length, the JSON header and each tensor's `data_offsets` slice directly); `SelfPlay.workers=2` gives sharded files.
- Ctrl-C a run as a terminal does: fork it on a pty (`pty.fork()`, then
  write `\x03` to the master). The first SIGINT is Etude's saved stop (exit
  75); a second is a hard interrupt. Without a tty, uv 0.9.22 forwards a
  process-group SIGINT to its child, which gets two and ends `failed` with no
  save, so `os.killpg` on `uv run etude …` does not show what a person sees.
  Signal `.venv/bin/etude` directly, or use the pty. A job started with `&`
  from a non-interactive shell ignores SIGINT.
- Training corpus and checkpoint ancestry: use a fresh ETUDE_HOME for
  `uv run --extra evaluation etude train manabot/recipes/local.py --config
  '{"statistics":true}' --until updates=1`. Resume with the same home and
  recipe, `--resume --until updates=2`, omitting the config. Use the guide
  in docs/evaluation.md on the resulting checkpoints and compare Island
  draws and choices with terminal records from
  `read_statistics(home / "track/local/out/statistics")` at capture update
  one less than the outcome update. For ancestry, train a one-update recipe
  in the same home with SelfPlay.pretrained set to the update-one policy in
  the parent's Etude checkpoint record. The child corpus should contain the
  parent's first update and the child's first update. Query historical
  frontier settings against the sealed run declaration as well as its counts.
  Run the guide as written: extract the four `python` blocks under
  "Querying the training corpus" with a regex and replace only the track
  root (and, to run every query on the file-form corpus, drop the three
  frontier `read` lines), so an import or query the guide no longer supports
  fails. The first block starts from a policy file,
  `<home>/track/<run>/out/policies/0000000N-raw.safetensors`; try each file a
  run wrote, a moved copy, a string path, a storage root without the run, and
  a file from an earlier pass's `.runs/` (older files record no experiment
  and take `experiment=`). For a consumer outside the
  checkout, build both wheels from the warm target
  (`uv run --extra play maturin build --release -i .venv/bin/python -m
  managym/Cargo.toml -o <dist>` and `uv build --wheel --package manabot -o
  <dist>`, under a minute), install the engine wheel plus
  `manabot[evaluation] @ file://<wheel>` in a `/tmp` venv with the CPU Torch
  index, and run the same script with `<venv>/bin/python -I` from `/tmp` on
  absolute storage paths. `<venv>/bin/etude train
  <site-packages>/manabot/recipes/local.py --until updates=1` trains there.
  An engine wheel built before a merge from main does not match the merged
  Python; rebuild it.
  `etude train <any file>.py` runs a recipe file from an ignored `.runs/`
  directory too, named by its stem: copy `manabot/recipes/local.py`'s `train`
  and `model_copy` the settings for a child run (`pretrained=<parent
  policies/0000000N-ema.safetensors>`, `match` with `decklists="closed"`), or
  rebuild `SelfPlay(**dump)` with `match=None`, a `UniformDecks(...,
  decklists="mixed")` deck source and `model.world = managym.WORLD_SETUP` for
  per-game disclosure. Runs written by an earlier commit of the same branch are
  returning data; read them again after a table rename.
- Collector changes: scripted environments in `tests/training`
  (`collector._env = ...` in `test_credit_assignment.py`,
  `test_two_seat_learning.py`, `test_game_errors.py`) stand in for the vector
  environment, so an attribute the collector newly reads fails there, not in
  the tests of the change. Run those files; CI does not run
  `test_game_errors.py`. Read a failed job while its run is still in progress
  with `gh api --allow-escape-sequences
  repos/<owner>/<repo>/actions/jobs/<job-id>/logs`.
- Snapshot-aware self-play (`SnapshotConfig`, `docs/training-loop.md#snapshot-aware-self-play`):
  extract the guide's `snapshot_recipe.py` block into `.runs/` and run its three
  commands verbatim with `ETUDE_HOME` there (about 25 s each on CPU). Read
  `snapshot/*`, `source_selection/<source>/*`, `decisions_per_game` and
  `optimizer_exposures` per update with `etude.train.read.read(...).points`;
  the bank is `extra.snapshot_bank` in `out/state/save-*/state.json`. With
  four streams and the guide's defaults the first group starts at update 3.
  Also try `{"horizon": null}`, `{"group_size": 1}`,
  `{"replay_fraction": 1.0, "fresh_floor": 0.0}` (fresh rates must be absent),
  `--seconds 8` then `--resume`, and a pty Ctrl-C. With `{"record_games": 1,
  "statistics": true}`, `read_records(Path(out / "games"))` (a `Path`, not a
  string) must hold only fresh games: a continuation's 15-action prefix
  replay is a defect, as is an empty `reset` record repeating a fresh seed
  (16 complete plus one zero-action in-flight game per stream after 4
  updates). `points(...)` returns `Points` with `.x`/`.y` arrays. `workers`
  is not a recipe config key: check the startup refusals (`workers=2`,
  `wdl_entropy` on a scalar critic, `action_value_spread`) with a `.runs/`
  copy of the recipe that `model_copy`s `SelfPlay`/`SnapshotConfig` fields.
- Decision memory in self-play (`ModelConfig.decision_memory`): the smallest
  real run is `tests/training/test_memory_self_play.py`'s `settings()`: Gray
  Ogre decks, the four `max_decision_*` capacities, and
  `SelfPlay.{collection,bootstrap,learner}_rows=2`. Train a
  `decision_memory="none"` control, then start a `"gru"` branch from its
  checkpoint with `SelfPlay.pretrained`; the branch's gates read 0.0 and its
  shared tensors equal the control's. A control without the log capacities is
  refused as a trunk. Test reliance by seating the same checkpoint with
  `config={"blank_decision_memory": True}` through `manabot.eval.pairing`.
