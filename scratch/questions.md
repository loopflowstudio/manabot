# Review choices and remaining questions — MTG-329

Earlier numbered questions and prototype transcripts are retained in git at
`f09f5c2919b8b1a5cf6d72c4f4f4721dac68f653:scratch/questions.md`.

## Recovery and frozen workflows

Jack Heart, on the list including collector snapshot/restore: "do em all in
this diff". Current-game recovery, exact uninterrupted equivalence and in-run
source repair are gone. Saved regime fields remain readable; execution names
revision f09f5c2919b8b1a5cf6d72c4f4f4721dac68f653. Older exact collector
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
