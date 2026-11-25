Read when the change touches experiment launches or reports (`experiments/`).

- Experiment reports and figures: arrive as a new reader with
  `MANABOT_DATA_CACHE` set to an empty directory, then `doppler run --project
  manabot --config prd -- uv run --extra artifacts python -m manabot.data fetch
  <paths>`. In zsh, pass a path list as an array (`"${P[@]}"`), since zsh does
  not word-split `$P`. Fetch only the files the figure code reads; whole
  evidence folders can exceed 1 GB. Redraw twice and compare digests, because a
  figure that changes bytes on every redraw also changes its manifest entry.
  Render SVGs with `rsvg-convert -z 2 -b white` and look at them.
  `fetch` skips files that already exist in the checkout ("already present"),
  whatever the cache holds. In the authoring checkout, fetch in a `git worktree
  add --detach /private/tmp/<x> HEAD` instead, running `uv run --project
  <checkout>` there, then `cmp` its files with the checkout's. Recompute the
  report's table cells from the fetched CSVs.
- Staged experiment launches (`etude launch ... --quote`): run each documented
  command verbatim under `doppler run -p etude -c prd -- doppler run -p manabot
  -c prd --preserve-env=RUNPOD_API_KEY --`; the locked client reads the launcher
  DB (2026-11-17). A quote prints the selected runs and the committed ceiling,
  then refuses or prices; confirm with `etude runs --experiment <name>` that
  nothing launched. `etude report` exits 1 until the experiment has a run.
  An experiment-file edit while runs are launched must keep their resume
  identity: for each `etude.launch.client.default().standing(<experiment>)`
  record, call `read_training_jobs(Path(file), given)` with its
  `launch.params` as `--kebab-name value` arguments and compare the request's
  `job.config` with `record.launch.spec.config` byte for byte (a JSON string).
  `params` alone differ by launch arguments and prove nothing.
- Live experiment report from the mini's evaluation: rsync
  `mini-heart.local:etude-eval/<study>/track/` into `.runs/`, then
  `ETUDE_TRACK_DIR=<copy> doppler run -p manabot -c prd -- uv run --extra
  reports --extra artifacts etude report experiments/<file>.py --source
  r2://manabot/jobs/<experiment> --out <dir>` (about a minute). A stale local
  track renders a page of "names tournament …, which nothing holds" warnings
  with empty panels; that is missing data, not a broken report.
