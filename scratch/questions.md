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

## Contracts still missing from EXP-56

1. resume(extra=Counts()) return type: the design's call table says clocks,
   while its loop assigns Counts. The minimal guide expects a clock mapping
   when no extra is supplied. Also settle how replaced Counts values reach
   run.save: registration of the original extra alone would save stale counts.
   Do not implement a local adapter to conceal this mismatch.
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
