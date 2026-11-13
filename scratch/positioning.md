# Latest direction — 2026-11-13

Earlier exploration and exact quotes: `bd14801a72d48ad38eeca18d08fc1c9e7f99612a:scratch/positioning.md`.
The accepted design incorporates the relevant decisions.

These notes, `ontology.md` and `etude-readme-draft.md` were removed from this
directory by the MTG-329 run at about 12:22 and restored from commit
`bd14801a72`; the entries above were re-entered from the conversation.

Jack Heart, 2026-11-13: "is mtg-329 running too?" State at 12:40: MTG-329's
Graph stopped as blocked after retiring the training CLI (it did not write the
loop against the unbuilt API); EXP-56's Graph is running, its worktree is in
an unresolved merge with main, and it found that an interrupted `run.save`
through the real delivery leaves model from save 2 with optimizer and
metadata from save 1.

Jack Heart, 2026-11-13: "can you help resolve the questions with exp-56"
(EXP-56's session had reported the design's five open points unresolved).
Asked points 2–5 again as a dialog. He answered 2 "Both (Recommended)"
(`--resume RUN --until` extends the same run; `continue` for changed config
makes a child) and 3 "Show how to keep them"; gave no answer to 4 or 5; and
said "i thought we clarified a lot of this". What had been said before on the
five: only point 1 (`tracked_run?`, `track_run?`), overtaken by "ok,
RunRequest works". Design's Open section rewritten with this status; 4 and 5
carry my defaults, not his answers.

Jack Heart, 2026-11-13, on the public example (open point 4; also bears on
explore question 2, how wide multi-agent is): "I think we shoudl be open about
the idea that etude was built to support manabot, and use it as a flagship
example. but otherwise i think we should look at trying to support pufferRL's
games and work well with them, as well as softmax's". Design point 4 updated.
What "support" means for PufferLib and MettaGrid is not specified; from
memory, unchecked: PufferLib ships its own trainer and multi-agent
environments, and MettaGrid is Softmax's multi-agent environment.
On filing that as its own Task: "yeah thats fine, it cna be an unstarted
task". Correcting my "MettaGrid": "and not mettagrid, coworld". Filed EXP-62
(PufferLib and Softmax Coworld examples), unstarted; EXP-56 steered with the
correction. I did not look up what Coworld is.

Jack Heart, 2026-11-13: "https://softmax.com/hanabi is of interest to me,
maybe we could find another implementation of hanabi", then linked
https://github.com/google-deepmind/hanabi-learning-environment. Findings
posted to EXP-62 (Softmax Hanabi is a Coworld for four LLM-piloted players
with a shared score; DeepMind's HLE is Apache-2.0 but archived since 2023;
JaxMARL has a maintained JAX Hanabi).
On which Hanabi implementation: "JaxMARL sounds good". Posted to EXP-62 with
the consequence that JaxMARL is JAX while EXP-56 delivers a PyTorch writer
first.
On EXP-62 starting with the write-it-yourself save form and reporting the JAX
writer back, leaving EXP-56 unchanged: "yeah thats fine".
On EXP-62 starting with the write-it-yourself save form: "yeah thats fine".

Jack Heart, 2026-11-13, on EXP-56's state-publication questions (1 unchanged
files: a shared content-named files, b self-contained saves, c a copy path; 2
who deletes old remote saves: a launcher, b nothing, c lifecycle rule; 3 keep
the previous save locally; 4 running_hours from the save): "1b is fine, my
other feedback was that extra stuff locally we can cleanup automatically is
ok, but extra uploads are not". EXP-56 steered and restarted. In that steer,
marked as mine: upload less often than local saves; two alternating remote
slots so no delete right is needed; launcher cleanup as fallback. 2 and 4
were not answered by him.
Then, on question 2: "2a is the best of thse options" (the launcher deletes
replaced remote saves). EXP-56 steered; my alternating-slots suggestion
withdrawn there. 4 still unanswered.
On question 3 (keeping the previous save locally): "i dont think it should
need to save if its confirmed published online, but i also think we can have
some grace in our cleanups". EXP-56 steered; my earlier reading ("keep the
previous save locally as a fallback") corrected there.
Asked what counts as confirmed and how long the grace is: "dunno, dunno".
EXP-56 told to choose the simplest correct thing and list both as unanswered.

Jack Heart, 2026-11-13, on MTG-329 having stopped as blocked before writing
the loop: "i still think we should restart mtg-329 to match the proposed API,
and just have it wait after its done writing to block then".
