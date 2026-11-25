Read when the change touches checkpoint loading, input readings or files an earlier version wrote.

- Checkpoint compatibility (loader and Academy listing): put the
  checkpoints in one directory, plus broken copies made with `torch.load` /
  `torch.save` after editing one `world_binding` field (`world_digest`,
  `learning_schema_sha256`, `naming_branching`). Start
  `MANABOT_RUNS_DIR=<dir> ACADEMY_FRONTEND_BUILD=… uv run managym academy
  --no-open --port <unused>` (bare `academy`; `watch` requires `--player`);
  `GET /api/opponents` lists each entry's `available` and `reason`, and the
  Seat 2 bot select's "Registered policy" shows them as disabled options.
  An unexplained-key copy: set `semantic_cards.record_token_keys[0]` in a v11
  `.pt` state dict to `(3 << 32) | 12345`. In the browser, select
  `[data-testid=opponent-select]` → `policy`, then the
  `[data-testid=checkpoint-select]` option, and press "Start game" (the setup
  form has no "New Game" button); the end dialog offers "Rematch".
  `uv run python -m manabot.backends.convert` must accept both a
  PyTorch zip and a container saved under a `.pt` name (`save_checkpoint`
  always writes a container); convert a v11 PyTorch zip and load its `.safetensors`. A v9 or
  v10 checkpoint must report v0.2.6 as its last supporting release. The repository
  no longer holds one: take the v9 pilot from the last 0.2.x tag
  (`git show v0.2.6:tests/fixtures/pinned_pilot/v9/pilot.pt > .runs/…/v9-pilot.pt`)
  and check `load_policy`, `manabot.eval.local` as either seat and
  `/api/opponents` (listed with `available: false` and the reason).
  `manabot sim` cannot seat `heuristic` as villain; seat the checkpoint twice.
  Write returning-user checkpoints on `main` as in "Files a returning user
  wrote". For a column append, cover both `deck_world` and `checkpoint_world`
  bindings. The Elemental recipe writes `checkpoint_world`. Score the same
  seeded games on both trees and compare decision counts and logits within
  1e-5 to verify saved-prefix compatibility. Open one as a Seat 2 opponent with
  `/?hero=ur_lessons&villain=gw_allies&opponent=<sha256>` and press Start
  game. Admission changes also reach `tests/model/test_attack_choice.py`
  (old-world fixtures with an empty state dict), which CI's model job runs.
- Files a returning user wrote before a change: `git worktree add --detach
  /private/tmp/<x> main`, build its release extension with its own
  `CARGO_TARGET_DIR` and copy the `.so` into that worktree's `managym/`. Then,
  from inside that worktree with `PYTHONPATH=/private/tmp/<x>`, call this
  checkout's `.venv/bin/etude train manabot/recipes/local.py --experiment demo
  --config '{"record_games": 1}' --until updates=1` and `.venv/bin/python -m
  manabot.eval.local ... --deal-seed` four times with `--seconds 45` (print
  `managym.__file__` first). That leaves a run to `--resume`, recorded games
  for `managym academy replay` and an incomplete panel for `--status` /
  `--continue`, all written by main. Copy untracked `experiments/data`
  fixtures into the worktree before comparing a test there. Earlier passes'
  `.runs/qa*/` hold older checkpoints, panels and `traces/play.sqlite` too.
- Input readings (`manabot/policy/readings.py`, `input_readings` in a
  checkpoint): the returning user is a checkpoint written before a change.
  Train `manabot/recipes/local.py` in a base worktree (its own release
  extension, this checkout's `.venv/bin/etude`, `PYTHONPATH` at the
  worktree), then play the same seeded games with `scripts/pinned_pilot.py`'s
  `play` on both trees (the v11 pilot's decks) and diff probabilities (381
  decisions matched within 3e-8 in MTG-404). Warm-start a `.runs/` recipe
  file with `pretrained=<that checkpoint>` and read the child's
  `input_readings`: they must equal the parent's legacy readings. A copy with
  one reading renamed must show in `/api/opponents` as unavailable, naming
  the column.
