Read when the change touches Loopflow graphs, skills or QA notes (`.lf/graphs/`, `.lf/skills/`, `.lf/qa/`).

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
