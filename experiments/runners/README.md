# experiments/runners/

One-off experiment drivers, each written to produce a specific `exp-NN` report next door.
`manabot/` is a library of reusable instruments; anything that exists only to generate one report lives here instead.
Run one from the repo root: `uv run experiments/runners/<name>.py --help`.

Note: reports written before 2026-07-10 reference these runners at their old
`manabot/verify/run_*.py` / `python -m manabot.verify.run_*` paths; those lines are
historical provenance and were left as written.

`distributed_benchmark.py` supervises the disposable ETU-108 CPU probes in
`distributed_workloads.py`. See [research and launch limits](../../docs/distributed-rl.md).
Inventory and supervisor checks need only Python 3.12; Torch/native workloads
remain untested and require a separately available project environment.
