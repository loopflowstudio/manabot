# Review choices and remaining questions — MTG-329

Earlier numbered questions and prototype transcripts are retained in git at
`f09f5c2919b8b1a5cf6d72c4f4f4721dac68f653:scratch/questions.md`.

## Recovery and frozen workflows

Jack Heart, on the list including collector snapshot/restore: "do em all in
this diff". Current-game recovery, exact uninterrupted equivalence and in-run
source repair are gone. Saved regime fields remain readable; execution names
revision f4e1822faf46bbc9cdc43909293f8cbc8de4a300. Older exact collector
snapshots require revision 439379869. Frozen recovery declarations include
Elemental long-183, two-seat-181, uniform-174, ur-gw-186 and TLA c5fba4-w64.
Review still needs to assess that loss; no frozen run was rebound or rerun here.

The simpler cutover retires frozen staged/adaptive/frozen-opponent launchers
explicitly instead of translating their methods. Their records, evaluation and
reports remain. Current local/baseline/Elemental/belief training and nightly
baseline evaluation use the new loop. New local comparison scheduling was not
added: experiments return Etude jobs; historical comparison records stay readable.

Checkpoint following is an Etude evaluation pass. Whether a local follower is
wanted remains a product question; the old monitor worker is deleted.

## Reversible interface choices

- Existing experiment `runs()` names stay: Etude discovers them. Ordinary
  recipe entry points are `train(config)`, with a required Pydantic config.
  Jack Heart: "i liekd it better when there were explicit eval(), train() and launch() functions".
- `--recipe` selects explicit experiment functions, including `file:function`.
  CLI configs need seed and device, the fields passed by the CLI. The launch owns the experiment name.
- Default out is `.runs/<recipe>-<seed>`; local alone uses a fresh timestamp
  so a second default smoke actually trains. Same out resumes deliberately.
- Planned stops exit 75, matching Etude. Target extension uses run.until and
  does not mutate recipe config. Damped and PPO rates both read a horizon
  fixed in the recipe's learning settings (PPO: concept review 2, below;
  implemented and validated).
- Device, workers and threads may change on resume. Other learner settings
  refuse. RNG/native streams restart at the saved update; Adam/EMA/counts resume.

## Corrected contradictions

The 2026-11-12 belief smoke changed policy/final.pt (7ab8b51c → b063a85d)
while retaining old histories and sampler. There is now no final.pt. Policy
checkpoints are immutable; downstream belief filenames carry the policy stem.
The real SDK stop/resume/extension check verifies old bytes stay unchanged.
Selection output caching additionally binds policy, population, match, reference
and step cap. Historical selection records remain readable with unknown
collection identity; they cannot be reused as a new collection cache.

## Delivery dependency

EXP-56 PR #103 was open and SDK 0.2.16 lacks local_launch. Tests use an isolated
source override; no package version was invented and no local path was pinned.
Release/pin/lock and installed-consumer proof remain before publication.
Actual provider courier restore/retry and service continuation remain unproved
from Manabot. No paid launch was performed for this cutover.

## Concept review — 2026-11-13

Reversible, simpler option kept: the CLI passes only seed, device and (for
local/baseline) learning seats and padded shapes to a recipe config. Other
fields are unreachable, so `manabot train --recipe elemental` on a laptop runs
256 CPU streams; the two-stream smoke config exists only in `runs()`. Whether
the CLI should accept any field of the recipe's Pydantic config is not settled
by the recorded direction and was not built. It matters because the README
advertises that command as the way to run the current recipe locally.

## Implement choices — 2026-11-13

The hidden CLI alias `--steps` warns that removal is in Manabot 0.3.0 (the next minor release after current 0.2.3); no release bump is part of this pass. Both target flags together refuse, avoiding ambiguous precedence.

## Concept review 2 — 2026-11-13

Reversible, taken without asking: PPO schedules read a horizon fixed in the
recipe's learning settings, not the launch target (plan.md, item 1). The
alternative, refusing a changed target for PPO, was not chosen because it
removes extension for one learning rule only. Still open, unchanged: whether
the CLI should accept any field of a recipe's Pydantic config.

## Docs for concept review 2 — 2026-11-13

Reversible, written into docs/training.md without asking: a PPO run extended
past its `Learning.horizon` trains on at the final rates (the clamp both
schedules already have); it is not refused. Implemented choice: `SelfPlay.learning` is required, avoiding an unusable
implicit PPO default. `gray_ogre_self_play` requires an explicit horizon,
provided by the belief recipe's own update count.

## Concept review 3 — 2026-11-13

Open, for demo; reversible and additive, so it does not stop the Graph (the
simpler option, no new flags, stays): should `manabot train` set fields of a
recipe's Pydantic config, at least its update count? Why it matters, beyond
Elemental's 256 CPU streams: `runs()` on the Etude path sets
`ElementalConfig.updates`, while the CLI can only set the stop target, so the
same word means the recipe length under `etude launch --updates` and only
where to stop under `manabot train --updates`. For local and baseline that
changed what `--steps N` did on main (plan.md, concept review 3, item 2).
Also still open for demo, nothing built: a local checkpoint follower.

Implemented the reversible choice: drop the loop's `final` checkpoint tag
in favor of Etude's `TrainingRun.final`. The alternative, untagging the old
checkpoint on extension, keeps a second owner of one fact.

## Concept review 4 — 2026-11-13

Reversible, taken without asking: per-update monitoring rows the executor
logged (phase split, CUDA/MPS peaks, RSS, host load) go back into the loop's
row, and the update benchmark runs the loop, rather than deleting the panels,
the guide's claims and the script. The other option, dropping them, was not
taken because the branch's own docs and two skills still send people to them.
Not asked of anyone: nobody was told these rows were lost; the loss reviewed
under "do em all in this diff" was recovery. Raise it at demo with the
recovery loss. Open, small: whether loop runs need the `transitions` clock
back; nothing reads it today that was checked here.


## Monitoring implementation choices — 2026-11-13

Etude has a Time clock. Active time is collection plus learning for completed
updates, excluding setup, persistence and failed partial updates. Persist it
in Counts; no transitions clock is needed by checked readers. MPS exposes
current tensor/driver allocation, not peaks. Benchmark uses a bounded in-memory
Tracker, with no Etude run, and requires fresh output so cumulative counters
cannot be mistaken for this measurement. Retire the frozen throughput driver
rather than translate its TrainingRun contract; delete its exclusive execution
tests while preserving machine setup and packing evidence tests.

## Main sync — 2026-11-13

Keep the cutover's executor and exclusive bucketed-execution test deletions.
Main's automatic precision resolution survives in retained regime records;
live SelfPlay keeps explicit float32 defaults. Preserve the BF16 rejection
report and artifact manifest. The newly merged frozen precision-ladder launcher
refuses with pinned main revision `f4e1822faf46bbc9cdc43909293f8cbc8de4a300`,
which contains its executor and original builder. Its report/evaluation readers
remain. This follows the existing frozen-workflow cutover rather than adapting
the scored experiment to a different loop. Branch scratch plans stay the active
context; MTG-324's scratch history remains on the pinned main commit.

## Token-card main sync — 2026-11-13

Keep this Task's review notes as the working context. Main's token-card naming,
Scryfall facts and fixture assumptions remain at
`22ed4a733484d27db2fd51274d1b4e23245a6741:scratch/questions.md`.
The incoming token-card implementation is preserved without changing the
execution cutover or its outstanding delivery obligations.

## Concept review 11 — 2026-11-13

Reversible, taken without asking: a local resume at a different source
revision is documented, not refused (plan.md, review 11, item 2). Refusing
would make a local bug fix start a new run; each checkpoint already names the
revision that wrote it. Raise with the recovery loss at demo.

## QA implementation choice — 2026-11-13

Keep interruption handling in the CLI: show the last-save resume guidance and
exit 130. Do not convert KeyboardInterrupt into Stopped, whose contract says
state was saved, or write Etude's state.json from Manabot. Etude owns the
admission/interruption status fix (plan.md). The release remains pending;
continue testing with the real SDK source override without committing a path.

## Docs for the train-API rebuild — 2026-11-13

Written against `design-train-api.md`; nothing here was run, and the Etude
calls do not exist yet. The docs settle these details the design leaves
unstated. Each is reversible; the simpler option was taken without asking.

- The whole `manabot train` command goes, not only `--out`: it exits naming
  `etude run FILE`. No `--recipe` names or `--list`; a recipe is a file with
  `train`. An installed package without a checkout has no documented smoke
  path; the guide's recipe example is the file to write.
- Device is only a config field with the recipe's default (`cpu` local and
  Elemental, `cuda` baseline). The CLI's "CUDA when available" selection and
  the local recipe's 120-second cap are gone with the CLI.
- Manabot's world-bound policy files move from `out/checkpoints/` to
  `out/policies/`, because the design gives `out/checkpoints/updates-N/` to
  Etude's own checkpoint folders.
- `Counts` gains the completed update count, and the loop publishes the
  untrained policy when `counts.updates == 0`. The design's loop listing
  publishes no update 0, but `baseline.evaluate` compares initial with final.
- `SelfPlay.save_every` goes; Etude decides when a save is due (design, open
  item 5).
- `CLOCKS` is defined once in `loops/self_play.py`.
- Belief: `manabot/recipes/belief_policy.py` (`PolicyConfig`, `train`,
  `match()`) and `belief_sampler.py` (`BeliefConfig`, `train`, and `runs()`
  returning both jobs). The sampler run declares a `steps` clock and keeps no
  state: an interrupted one collects and fits again.
- Old `state.pt` directories: the guide names `f577362135` as the last
  revision that resumes one. That commit is this unmerged branch (the
  `state.pt` loop never reached main), so it is reachable only through PR #500.

Open, for EXP-56 or a person; the docs show one spelling of each:

1. What `run.resume(..., extra=Counts())` returns. The design's call table
   says the clocks at the last save; its Manabot loop assigns the result to
   `counts`. `docs/training.md` follows the loop (`extra` comes back);
   `docs/training-loop.md`'s minimal loop, which passes no `extra`, reads
   `["updates"]` from the result.
2. How a job names another run's final checkpoint. The docs write
   `training(train, ..., start=policy)` where `policy` is the earlier job,
   meaning its run of the same seed. The design says only
   `start=<checkpoint of another run>`.
3. Whether `etude run` can give a run a start (the design lists no such
   option). The docs run the belief recipe only through `etude smoke` and
   `etude launch` on its job file.
4. What `run.input("start")` returns for a checkpoint recorded by path. The
   docs treat it as the policy file.
5. Where `etude run` writes and what it names the experiment. The docs say
   `~/.etude/track/<file stem>/<run>/` (tracking's default today) and do not
   state the format of a generated run id.
6. How `scripts/bench_training_update.py` runs once the loop has no untracked
   path. `docs/workflows.md` says each reading is a new local Etude run and
   drops `--out`.
