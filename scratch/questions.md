# MTG-329 current choices and SDK questions

The complete previous questions and exact attribution are preserved at
`bd14801a72d48ad38eeca18d08fc1c9e7f99612a:scratch/questions.md`; the current design is design-train-api.md.

## Reversible choices in the user docs

- Retire all of manabot train (including --recipe/--list), with Etude migration
  guidance. Device is a config field with its recipe default; no automatic
  CUDA choice or local CLI time cap. This CLI deletion is implemented.
- World-bound policies go in out/policies; Etude owns out/checkpoints folders.
- Counts includes completed updates and active collection + learning seconds.
  Keep initial policy publication: nightly evaluation needs update 0.
- Etude owns save cadence, so SelfPlay.save_every goes. Declare CLOCKS once.
- Belief splits into belief_policy.py (PolicyConfig/train/match) and
  belief_sampler.py (BeliefConfig/train/runs returning both jobs). Sampler
  declares a steps clock and restarts collection/fitting after interruption.
- The guide names f577362135562e569033fe33e48c7ef0411a829d as the last reader
  of branch-only state.pt folders (reachable through PR #500).
- The installed smoke writes a recipe file importing LocalConfig/train, then
  invokes etude run. It uses ETUDE_HOME to isolate output and reads
  track/local/smoke/out/policies/00000002-raw.pt. This follows the intended
  docs layout; it needs validation when EXP-56 implements that command.

## Spellings taken from EXP-56's draft docs (dda6b17), 2026-11-13

The user docs now follow `etude.exp-56-give-a-training/docs/training.md` and
`launcher.md` where they are more exact than the design. Nobody decided these
for Manabot; they are that draft's current spelling and move if it moves.

- Saves: `out/state/current.json` names `out/state/save-NNNNNNNN/`, which
  holds `state.json` and the framework files (design, "How a save is stored").
- `extra` is changed in place: "Etude saves that object as it stands, so
  change its fields in place". So the documented loop is
  `row = step(settings, learner, ema, collector, counts)` and `step` updates
  `counts`; the design's `counts, row = step(...)` would save stale counts.
  This answers contract 1 below for the `extra` case only.
- A run id is the job's name and seed; `--name` names the job. The docs drop
  `--name elemental-197` and use `elemental-seed-197`. The design's command
  table calls `--name` the run's name.
- `--resume RUN --until updates=N` takes the same run further, finished or
  not (design Open 2, "Both"); `--resume` on a finished run with no later
  target is refused; `etude continue` (`--until`, optional `--config`) makes a
  child run. A save is due five minutes after the last.
- Updated in code, not executed: `scripts/release/smoke.py` and
  `.github/workflows/manabot-integration.yml` use `--name smoke` and now read
  `track/local/smoke-seed-0/`.

## Contracts still missing from EXP-56

1. Resolved by source at dda6b17: resume(extra=...) returns and mutates that
   Pydantic object; saves read it in place. The no-extra form returns a folder,
   not clocks. Minimal now registers a Progress Pydantic object as well.
2. start=policy where policy is an earlier Job: how does it select that run's
   final checkpoint of the same seed? The docs assume that spelling.
3. Whether etude run accepts a start; docs run belief through smoke/launch.
4. input("start") for a path checkpoint: docs expect a policy file.
5. etude run root and experiment naming: docs assume
   $ETUDE_HOME/track/<file stem>/<run>, with no generated-id promise.
6. Benchmark after removing the untracked path: docs say a fresh local Etude
   run per reading, no --out. Use the released local-run contract, not a stub.

Missing APIs are a dependency, not a reason to add Manabot execution machinery.
No SDK changes are made in this checkout. Its new config/seed checks also own
refused admission and interruption status: source 105abdd can overwrite an
already finished run with failed on a later rowless exception.

## Retained review obligations

Current-game recovery, exact uninterrupted equivalence and source-repair
recovery are removed. Historical frozen declarations include Elemental
long-183, two-seat-181, uniform-174, ur-gw-186 and TLA c5fba4-w64.
A local checkpoint follower is still a product question; none is built.
PPO schedules use a fixed Learning.horizon, clamped beyond it; target extension
never changes recipe learning settings. Preserve seeds/paired deals/replay.

Reversible implementation choices: linked belief jobs explicitly use names
`policy` and `sampler` to match the documented paths. Baseline evaluation reads
completed optimizer counts from policy metadata, avoiding a reader for Etude
private save layout. Benchmark readings use an isolated temporary Etude home
and its real metric reader; no untracked loop or output-directory option.
