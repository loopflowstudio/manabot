# Academy shipping — current handoff

Jack Heart approved "otherwise lg2m, ship it" after requesting board-first play,
primary legal choices, collapsed logs/tools, Scryfall art, frontend relocation,
separate battlefield groups and a bounded battlefield. All are implemented.

Task 295863, PR https://github.com/loopflowstudio/manabot/pull/342.
Current branch jack-heart/295863-redesign-academy-around. Main's Elemental w10
change 6d7fee86 was merged and conflicts resolved without rewriting frozen packs.
The Elemental manifest is relocated byte-identically into academy/ui. Dead
academy/learn_demo.py remains removed. Current demo/navigation fixtures are new
*-w10.json files; w9 and earlier files are retained. All 106 cached art images
(69 new Elemental identities, 37 existing) are local ignored assets; Docker fetches
both pack sets, with credits. No hosted deployment or training was launched.

Published head 0933be9b5155815c8e1b54b41cbb9982a5f5ee1d.
Auto-merge is temporarily DISABLED to prevent the reference-generation workflow's
check names from permitting merge before images are committed. Re-enable with
lf arm only after the generated Linux references are committed.

Active Linux CI generation run: 39699491549, Visual Reference Gate job
119565003887. A single gh run watch process is running, exec session 85522,
log /tmp/academy-linux-generation-watch.log. This is not a live lf Flow.
The earlier ship Flow 8cb05b00-2cf5-4c5d-958f-76a130d97949 stopped because its
wait-and-fix launcher treats this primary root Process a5e9759c-ebcc-4ae2-92ac-2d317821112a
as a competing checkout owner. Direct lf land also has that bug. Do not relaunch
an automatic ci-fix worker: apply the ci-fix skill directly, using lf sync --manual,
lf commit, and lf arm. No worktree or new agent is needed.

Local checks after w10 reconciliation: 106 UI tests; Svelte zero errors/warnings;
production build; 22 focused surface/demo/navigation Python tests; 19 Rust protocol
checks; all 19 affected browser cases have passing focused evidence (18 in stable
selection plus repaired terminal-focus case). Prior gate has 289 rules and 30
arena/monitor passes. /tmp/academy-w10-*.log contains results. The release matrix
was refreshed by running its identical action policy against actual GameSession:
145 and 566 exact Commands in opposite seat directions, all 10 prompt families.
The local macOS browser completed the first match and reached all 10 families
in the second; it was intentionally stopped to defer full completion to Linux.
No full macOS release pass is claimed. The 20-minute limit accommodates explicit
mana/payment choices, retaining every exact sequence, legality, keyboard,
privacy and accessibility assertion.

New reference corpus v8; v7 PNGs plus archived original matrix remain unchanged.
Generation uses the installed CI workflow's update_visual_references=true input.
It runs the release suite, Python matrix/navigation contract, then comparison.
When job succeeds, download visual-references-v8 artifact from run 39699491549,
inspect representative rendered images, place its PNGs in
academy/ui/e2e/visual-references/v8, run the Python release contract, commit via
lf commit, and arm. Then wait for the normal PR comparison and all required checks
on that final exact head, repair failures if needed, verify actual GitHub merge,
run lf pr reconcile to record delivery and lf task complete 295863, reread status.
Do not use the generation success alone as evidence committed references compare.
The normal PR CI on 0933 was cancelled as redundant because its missing references
cannot pass; its successor final head needs a complete normal PR CI run.

A historical public-commitment receipt regeneration failure reproduces at unchanged
base (revision 18 attack offer2 absent). Current rules now have separately versioned
w10 authored receipts. Strict old test assertions and frozen artifacts are unchanged;
passing current readers/runtime does not certify regeneration of the old producer.
This limitation is disclosed in PR copy. Do not weaken those tests or rewrite
historical receipt bytes to force a pass.
