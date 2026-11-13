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
  CLI configs need seed and updates. The launch owns the experiment name.
- Default out is `.runs/<recipe>-<seed>`; local alone uses a fresh timestamp
  so a second default smoke actually trains. Same out resumes deliberately.
- Planned stops exit 75, matching Etude. Target extension uses run.until and
  does not mutate recipe config. Damped rates have a fixed recipe horizon;
  PPO rates still depend on the requested update count, documented as such.
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
