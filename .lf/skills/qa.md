---
requires: a change or behavior to assess
produces: a change that works for the people who will use it, with what you saw and what is still wrong
action_style: procedural
---
Use the change the way the people it is for will use it, and fix what breaks.

You are its first user. Nobody should open a demo or merge a PR and find it broken
on the first click; if they would, this step failed.

## Use the real thing

Run the real product, not its tests: the built app, the installed CLI, the live
page in a browser, the API over the network, the simulator or device. Drive it
with computer use, a browser, a terminal or whatever reaches it the way a person
would. Internal setters, test-only endpoints, mocks and fixtures do not count.

Getting there is your job. Build it, start its services, create accounts and
realistic data, sign in, set up whatever state a real user would already have.
When no route exists, make one. Hand back only what truly needs a person, such
as their own credentials, a physical device, a payment or a production approval,
and say exactly what that left unchecked.

Make the scenario as real as you can reach:

- Arrive the way a person arrives: a fresh install or first visit, then a
  returning user with existing data from before this change.
- Do the main job end to end, then the things people actually do: wrong input,
  empty and large data, going back, interrupting and resuming, a slow or failed
  network, two things at once.
- Check what it leaves behind as well as what it shows: files, rows, messages,
  processes, and what another part of the product now reads.
- Look at it. Read the output, check the screenshot, notice what is confusing,
  slow or ugly, not only what is wrong.

## Fix and repeat

Repair what breaks within the change's scope, then go through the experience
again from the start. Keep going until it works for the people it is for. A
change to what the product should be is not yours to make; describe it
instead. When asked for a read-only review, report instead of
fixing.

Run tests only where they tell you something the real path cannot reach, and
keep any you add about what a person would notice.

## Keep what you learned

How to launch and drive this product belongs in this skill. When you had to work
out a step it does not describe, or a step it describes no longer works, write
the working version here with `lf skill customize qa`, replacing what was wrong.

Show what you did in `scratch/qa.html`: each scenario, what you did, what you saw
with screenshots or transcripts, and what you fixed. Finish with what works now,
what you could not reach and why, and what is still wrong, worst first. Do not
publish, land or choose the next Graph step from here.

## This repository

- Loopflow workflow changes (`.lf/graphs/`, `.lf/skills/`): the user surface is
  `lf graph list`, `lf graph show <name>` (and `--json` for nodes and
  `returns_to` edges), `lf help <skill>` for the text an agent receives, and
  `uv run python scripts/check_loopflow_workflows.py`. Check every command,
  flag and doc anchor a skill names against the real tool (`--help`, `rg`).
  Launching the graph starts agents and possibly benchmarks; inspecting and
  compiling it is sufficient for workflow-only QA. When the Task authorizes an
  actual-workload measurement, run the guide's benchmark example as written into
  an ignored `.runs/` path, then read the receipt fields the skills name. The
  default CPU `bench_training_update.py` took about 13 minutes and 8 GB resident
  in one 16-CPU Mac reading; budget for the workload before starting it. Keep its
  process supervised through completion or cleanup. Measurements require the
  Task's scope and resource authority; never create artificial contention to
  test them (see `TESTING.md`).
- Academy table: `./scripts/play --port <unused> --frontend-port <unused>`; wait
  for the `ACADEMY_PLAY_READY` line (it names Python, native ABI and module
  path), open the frontend URL, press New Game, play a land and Pass Turn.
  Stop with SIGINT to the `scripts/play.py` process; it stops both servers.
- First-time clone: copy tracked plus untracked non-ignored files
  (`git ls-files -co --exclude-standard`) to an empty `/tmp` directory and run
  `ACADEMY_CLEAN_API_PORT=… ACADEMY_CLEAN_FRONTEND_PORT=… ./scripts/verify-clean-machine`
  there; it builds cold, runs the browser proof and writes
  `.cache/clean-machine/proof.json` (`launch.elapsed_to_playable_ms`). Shared
  uv/Cargo/npm download caches stay warm locally, unlike the CI reference.
- Library consumers: `uv run --no-project python scripts/release/prepare.py <staging>`, `uv build --sdist
  --no-sources <staging>`, then `uv build --wheel --python 3.12|3.13 <sdist>`
  (reuse a `CARGO_TARGET_DIR`). In a fresh `/tmp` venv with PYTHONPATH unset,
  install the wheel with `--torch-backend cpu` at the minimum pins and run a
  copy of `scripts/release/smoke.py`; also `uv init` a project and `uv add
  "manabot[<extras>] @ file://<wheel>"` to see the latest resolution.
- Rental availability and quotes (free reads, no rental):
  `SSL_CERT_FILE=$(.venv/bin/python -I -c "import certifi;print(certifi.where())") doppler run --project manabot --config prd -- doppler run --project etude --config prd -- uv run --no-sync etude availability <supply> --timeout 30`.
  Without `SSL_CERT_FILE`, the macOS python.org interpreter fails TLS
  verification and Etude reports it as `GET catalog/gpus: no reply`; that is
  local, not a provider outage. Confirm with `curl` before recording one.
- Experiment report CLIs (`scripts/report_collection_*.py`): run the guide's
  commands in a `git ls-files -co --exclude-standard` copy under `/tmp`,
  fetch with Doppler `manabot/prd` and an empty `MANABOT_DATA_CACHE`, and
  `cmp` regenerated outputs with the published bytes.
