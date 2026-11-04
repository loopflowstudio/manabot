# TLA pilot continuation — reconciled 2026-11-04 17:45 UTC

Jack Heart's accepted outcome is preliminary TLA self-play and paired evaluation
on the frozen 134-deck trophy population. GPUs and enabling repairs are authorized
under a cumulative $10 cap, with each allocation <=$1. The corpus, runners,
Vast integration and GPU learning milestone are delivered. V5 completed three
updates, then failed at 256 legal actions versus capacity 64 before exporting a
trained checkpoint. The pilot remains incomplete.

[Accepted decisions](questions.md) own scope and provenance;
[the report](../experiments/tla-trophy-pilot.md) owns measured results, costs and
historical validation. Its v5 analysis accounts for 32 valid initialization games
(5 wins / 27 losses) and 256 missing games; no trained-policy or improvement claim
is supported. All 114 v5 evidence files are manifest-published under
`experiments/data/mtg213/gpu-v5-20261104/`.

## Source and repair boundary

PR #420 merged at 08:15:17 UTC as
`0a5f2ad0fb55030f2b8caa2d7e403e3a3876dd8a`. V5 Run `e8c76bc2c06f` executed
frozen source `97a2cfb963d7e3a3b2816f31c45dda72f34cdddd`, input v7 and
`max_cards_per_player=128`, with fresh initialization. Its source, protocol,
failed attempt and initialization evaluation remain frozen.

The 09:58 verification, reconfirmed at 10:09, found executable source, protocols
and tests equal to main `5dd11cec4100aa0e1f4a23c063dfd5778e2ba28f`. SSH fetch
failed (`Permission denied (publickey)`); authenticated GitHub API and local
diffs established the merge. T6 widened player/card rows to 73/56 from v5's
67/55 and fresh action capacity to 512 from v5's explicit 64. Runtime and shape
guards reject substitution. Synthetic scheduling/analysis fixtures pin historical
identity and widths; they never authorize real replay, encoding or training
under that identity.

The frozen-source replay matches stream 103 / seed 32172 after 76 actions:
scalar, vector and Python encoders all reject 256 legal actions at capacity 64.
Diagnostic Python configuration at 512 retains all 256; frozen native buffers
still enforce 64. A configuration-only retry is insufficient. Review current
main's capacity handling before implementing the narrow repair: native scalar
and vector buffers must honor the selected ObservationSpace action axis, with
Python parity. A successor needs an explicit source/world/protocol/regime binding,
fresh initialization/evaluation and the same 134 entries. No successor or paid
retry has been selected by this diagnostic.

Remaining acceptance: an admitted trained checkpoint, complete paired schedule
and a report of learning and full cost relative to Elemental. The next execution
step owns fresh inventory, billing and quote reconciliation within both caps.
At the 17:33:51 settlement all pilot rentals had ended, with no Vast instances,
RunPod pods or unsettled hardware. Cumulative billed-plus-estimated cost was
$0.32710677365251912; conservative reservations remain $3.9139916049382716/$10.
Vast billing is still unavailable; these are dated receipts, not a fresh balance.

## Operational handoff

Steer c38da514 assigned GPU continuation to Graph 3529715b. The launcher
`.runs/mtg213/v5-continuation/launch-vast.py` used Etude's existing provider/supply
owner, published isolated `etude-sdk[s3]==0.2.14`, and frozen worker 0.2.12. It
submitted v5 once on `mtg213-vast5090-25382cc374b5`. Milestone comment 028c09c4
reported learning and failure; its approximate 17:32 observation time is corrected
to 17:33:51 UTC. No retry, watcher or background service remains from the launch.

Vast support originated in Etude EXP-35 (loopflow://etude/task/EXP-35), delivered
by PR #79; PR #80 supplied release 0.2.14 from source
`4c28da88cbd79b3f1e75669d7a84993a8da0f763`. The primary conversation owns release
completion. At 17:30, workflow 41876921350 had successful PyPI upload but remained
in progress; its release process (exec session 15617 in `/Users/jack/src/etude`)
was waiting and the tag was absent. The installed published client was verified;
final release/tag settlement was not. Do not start a duplicate release.

The managed lf binary recorded on the ship process is
`/Users/jack/.lf/bin/lf-f631bbe3771ffecf9ef045006c9d63f348165220b70fb2de7e6baacd556eda2c`
(0.15.1+ac8e088f5); the app binary lagged the shared-store wave_removal schema.
An earlier Task comment failed with “no Loopflow registry on this machine”; no
auth repair was performed. The later milestone comment succeeded.

Frozen-source local evaluation initially stopped before games with
`identity_path_without_files`. Explicit ETUDE_SOURCE_DIR/ETUDE_COMMIT identified
the extracted archive and allowed the retained initialization evaluation.

## Reduction and validation

Delete — do not maintain: separate `scratch/vast-provider-followup.md` and
`scratch/etude-release-status.md` handoff/status notes; their live facts are above.
No code deletion targets. Frozen protocols and evidence remain reproducibility
records. Wave memory predates the v5 result; its next authorized maintenance
step should reconcile it against the report. Uncommitted pre-edit notes/report
are preserved under
`.runs/mtg213/context-before-compress-20261104-1745/`; the prior blocked decision
is at `.runs/mtg213/v5-continuation/verify-before-20261104-1720.md`.

Check: `uv run --extra dev --extra artifacts --extra evaluation pytest tests/eval/test_tla_pilot.py tests/eval/test_experiment_files.py -q` — 43 passed at 09:58 UTC; reused for unchanged executable code. Gate owns affected acceptance checks; design/implementation owns the prospective action-capacity repair.

Check: `git diff --check` — clean; all 114 new manifest entries match retained SHA256/byte counts and existing entries are unchanged. `lf context` confirms memory and scratch are within both limits. No executable code changed.
