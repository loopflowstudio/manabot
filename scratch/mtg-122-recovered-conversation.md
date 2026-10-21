# MTG-122 conversation recovery — 2026-10-21

Jack Heart requested continuing the unavailable curriculum conversation here.
Current conversation: `session_a9f3f0dc744b40bbad9112cd103eab66`.
Recovered conversation: `session_de4ae59d0e964ff2944d0fb76ade392a`,
“MTG-122 Curriculum experiment design”. Its native thread is
`01a15292-673d-77d0-a798-bf91f28a8851`; the saved transcript is under
`/Users/jack/.codex/sessions/2026/10/18/`.

The recovered transcript ends during design on October 18 local time:

- Draft a separate experiment for L1 simple-pool pretraining followed by
  full-world Allies versus Lessons training.
- Include a matched boundary-reset control to distinguish pretraining from
  restarting optimizer state and schedules.
- Validate semantic checkpoint transfer; shared catalog dimensions alone
  do not establish transfer safety.
- At that time, the remote worker rejected pretrained starts. This was an
  implementation requirement to investigate, not a verified current limitation.
- The recovered session explicitly treated its work as design-only. Recovery
  does not authorize a training launch or replay historical operational steps.

The former checkout
`/Users/jack/src/fantasia.measure-learning-progress-across-simple` is absent.
The old workspace had a browser tab pointing to its
`experiments/data/curriculum/report.html`. That report was not recovered in
this pass. The similarly named exp-11 opponent-curriculum report in this
checkout is a different experiment and must not be substituted.

The original terminal diagnostic log was empty. The workspace/report connection
is evidence of related MTG-122 work, not proof that the design conversation was
the terminal's last session. Use fresh Task and delivery reads before advancing.

## Delivery continuation — 2026-10-21

Later inspection recovered completed results in PR #322, rather than only the
interrupted design. Both complexity and curriculum studies are documented under
`experiments/`. Jack Heart accepted fixing CI, updating the PR, landing it and
completing MTG-122. No further scientific cohort is selected.

This checkout now uses the existing Task branch. `lf task checkout` could not
restore the missing old checkout because existing conversations remain live or
unresolved; those processes were left alone. Current main was merged through
`lf sync --manual` and its conflicts resolved here. PR #322 was updated and
published at `65b5e81a`, with auto-merge off pending validation.

Repairs distinguish Etude tracker declarations from TrainingRun exports in
offline notebook discovery, preserve frozen w6 recipe construction on current
runtimes while retaining execution admission, and use the current world only
for explicitly declared CPU test fixtures. Fixed-source report/runtime tests:
23 passed. Play/catalog/SDK compatibility: 21 passed. Package boundaries: 2
passed. Full learnability diagnostic passed locally and on GitHub.

GitHub found default catalog digest drift from globally registering the three
research vanilla cards. The pending native fix isolates those registrations to
the research compiler's presentation registry, with Scryfall coverage retained.
Full debug validation uses `CARGO_PROFILE_DEV_OPT_LEVEL=1`, matching CI and
retaining debug assertions. See `/tmp/mtg122-rust-optimized-tests.log` and
`/tmp/mtg122-isolated-native-build.log` for ongoing verification.

Study tests passed 50 cases; two scoring fixtures rejected a source fingerprint
change because native source was edited during that run. Rerun the curriculum
fixtures with fixed source and the rebuilt extension. The prior unoptimized
debug runs were stopped, not counted as passes.

Local notes and Wave configuration were temporarily stashed for publication
and restored. Recovery stash `7951c22d7d4804ea321415a8748ed716278a99d9` is retained
until delivery finishes; remove only this known backup after verifying restored
files. Other stashes belong to other work.
