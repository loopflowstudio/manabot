# MTG-329: the prototype and the Etude API it wants

Historical prototype and pre-EXP-56 gap analysis. Current implementation and
remaining release work are in implementation.md; APIs and filenames below
describe the earlier stub, not the current code.

## The prototype (manabot, runs today against a local stub)

| File | What it is |
| --- | --- |
| `manabot/training/loops/self_play.py` | `SelfPlay` settings, `self_play(settings, out, *, seed, updates, run=None) -> Path`, `new_state`/`load_state`/`save_state`, `save_checkpoint`/`save_final`, `Tracker`, `Stopped`. Data plus functions: `step(state, ...) -> (state, row)` inside one visible `while` loop; resume = call again. |
| `manabot/recipes/elemental.py` | `ElementalConfig` + `elemental(config)`: `with track(...) as run:` around one `self_play` call; leaving the block finishes the run (failed when it raised). |
| `manabot/recipes/belief_sampler.py` | `belief(config)`: `self_play` → `collect_belief` → `train_belief`, passing paths; each step returns at once if its output exists. |
| `manabot/belief/collect.py`, `manabot/belief/train.py` | The two belief steps. |
| `manabot/recipes/builders.py` | `elemental_self_play()`, `gray_ogre_self_play()`. |
| `manabot/recipes/proposed_etude.py` | **The stub**: Etude's real `track`/`Run` plus `out` and `stopping()`. Deleted when Etude ships them. |
| `manabot train --recipe elemental\|belief` | Sets `ETUDE_OUT` from `--out`, calls the recipe; rerun resumes. |

Run output (`/tmp` smoke, 4 streams, 2 updates, 79 s CPU): `state.pt`, `final.pt`,
`checkpoints/0000000{0,1,2}-{raw,ema}.pt`, `game-attempts/0000.jsonl`,
`etude/<run>/{run.json,metrics,checkpoints,artifacts}`.

## What the recipe asks of Etude

```python
with track(clocks=[Step("updates", ...)], seed=..., config=...) as run:  # exists (Run is a context manager)
run.out            # NEW: the run's directory
run.log(row, updates=n); run.log_table(name, rows, updates=n)   # exists
run.checkpoint(path, updates=n, tags=...)                        # exists
run.stopping()     # NEW
training(elemental, config=ElementalConfig(...), max_hours=, max_usd=)  # exists
etude continue RUN --until updates=N                                     # exists, needs run.out
```

### Gaps (against etude main `be7c71b`)

1. **`Run.out`**, one directory per run, not per attempt. Etude carries it to
   storage as the recipe writes it and restores it whole before every attempt.
   Today `ETUDE_OUTPUT_DIR` is per attempt, and a retry gets back only the run
   records and the latest checkpoint. Locally, `out` is the directory the
   caller names (`manabot train --out`); the stub reads `ETUDE_OUT`.
   - Open: Etude's run record (`run.json`, `metrics/`) should live inside
     `out` (the docs show it there; the stub puts it in `out/etude/<run>/`).
   - Open: `run.checkpoint(path)` copies the file into the run folder. When the
     path is already inside `out`, Etude should record it by reference and not
     copy it twice. Today every milestone's bytes are stored twice.
2. **`Run.stopping()`**: true once the stop file exists, or once the deadline
   less the slowest interval between logged steps has passed. Today a job reads
   `ETUDE_STOP_FILE`/`ETUDE_DEADLINE` itself and guesses its own reserve. The
   stub measures the interval between `log` calls.
3. **Call again after a planned stop.** The loop saves its state and raises
   `Stopped` (nonzero exit). Etude should treat that like a lost machine:
   another attempt, same `out`, within hours, dollars and attempts. Distinguish
   it from a failure in the record (exit code or an `etude.Stopped` the recipe
   raises?).
4. **`etude continue` from `run.out`**: the child starts from a copy of the
   parent's `out` (so `state.pt`, with Adam and EMA), not from its latest
   checkpoint file. The child declares a new run, and its `updates` target is
   the only changed config field. The manabot worker refuses `etude continue`
   today.
5. **Checkpoint idempotence.** A recipe rerun after its last step records
   `run.checkpoint(sampler)` again. Etude could ignore a checkpoint whose digest
   and clocks it already holds.

6. **Where a resumed attempt's rows begin.** `track`'s docstring: a resumed
   run's rows replace earlier attempts' "from its latest checkpoint on" unless
   the process passes `resumed_at`. The loop resumes from `state.pt`, usually
   later than the last milestone, so rows between them would be dropped. The
   docs promise "later rows replace earlier ones from the resumed update".
   Either the first `run.log(..., updates=n)` of an attempt sets that point,
   or the loop reports it (`resumed_at`) after `load_state`; the first needs
   nothing from a recipe author. Read from the docstring, not exercised.
7. **Experiment name outside a launch.** `track` with no experiment raises
   `experiment_not_named` unless `ETUDE_EXPERIMENT` is set, and naming one in
   the recipe refuses launches from other experiment files. Docs omit it from
   recipes; `manabot train` supplies it (questions.md 11).

### What moves to Etude, what stays

Moves (existing Etude pieces cover them once 1–4 land): attempts, leases,
retry lineage, deadlines and stop reserve; run records, artifact references,
host facts, source hash, provenance and cost; checkpoint evaluation during
training (`etude eval` passes on recorded checkpoints); studies (several
`training(...)` jobs per experiment file).

Stays in manabot: `SelfPlay` and the loop, `state.pt`, world-bound
checkpoints, game-error ledger and failure capture, the belief steps, `play`
for evaluation, reports.

## Parity (checked)

`tests/training/test_self_play_loop.py`: with seed 197 and 2 updates, the
loop's raw state and final EMA equal the executor's `self-play-raw.pt` and
`self-play-ema.pt` bit for bit, for PPO (`Learning`, EMA 0.9) and damped
policy. Same seed streams: init `s`, collection `s+10000`, sampling `s`,
minibatches `s+20000`. Differences from the executor: no initial-raw
`run.checkpoint` unless `checkpoint_every` is set; no adaptive sampling, frozen
opponents, diagnostic roots, `initial`/`learning_state`/`initial_weights`
imports or `active_seconds` endpoints.

### Added 2026-11-12 (functional pass)

8. **A local launch.** `manabot train --recipe X --out DIR` must tell `track`
   the run directory and experiment as a launch does. The stub offers
   `local_launch(out=, experiment=)`, a context manager that sets `ETUDE_OUT`
   and `ETUDE_EXPERIMENT` for the block. Etude could ship that, or let `track`
   take `out=`; either removes environment handling from the CLI.
9. **Stop reserve.** The stub's `stopping()` measures the longest interval
   between two `stopping()` calls (one per update), not between `log` calls.
