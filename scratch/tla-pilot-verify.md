# TLA pilot continuation — reconciled 2026-11-04 after the GPU result

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
still enforce 64. A configuration-only retry on that source is insufficient.

The 2026-11-04 source inspection confirms that current main's native default is
512, but `input_buffer_config` in
`managym/src/python/vector_env_bindings.rs` still derives only card and fact
capacities. Scalar `encode_observation_into` and vector binding share that
helper; neither derives action, permanent or focus capacities from supplied
buffers. Vector event rows follow configuration only for public history; other
events use the default, while scalar input does not infer event rows. Shared-effect
and relation axes are version-fixed, not configurable capacities. Merely using
512 actions would agree with today's default without fixing that contract.

Primary operator comment bbc34e15 (2026-11-04 17:49 UTC) selects the routine
buffer repair under Jack Heart's existing enabling-fix and $10 authorization;
no new product choice is needed. Remaining implementation is to derive every
configurable axis at the shared native boundary, validate companion shapes,
and establish scalar/vector/Python parity on the retained 256-action position
and controlled non-default capacities, including overflow rejection. Historical
configurations and all legal actions must remain intact. The implementation cut below now resolves that buffer boundary, with focused
parity and overflow checks; it is not evidence of completed GPU acceptance.

Prospective v6 now binds the current world/content and tensor shapes, 512
actions and the unchanged 134 entries in its new regime/protocol. Exact
executable source freezing and fresh initialization/evaluation remain before
scoring. The executed v5 identity cannot be
reused for the repair or silently rebound to main's different world and widths.

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
`4c28da88cbd79b3f1e75669d7a84993a8da0f763`. Release completion is verified:
workflow 41876921350 succeeded at 17:40:58 UTC,
and tag v0.2.14 resolves to that source. The published client imported VastAPI
successfully. Provider configuration, the registered account SSH public key and
SDK release are delivered, superseding the earlier launch-integration blocker.

Frozen-source local evaluation initially stopped before games with
`identity_path_without_files`. Explicit ETUDE_SOURCE_DIR/ETUDE_COMMIT identified
the extracted archive and allowed the retained initialization evaluation.

## Evidence and validation boundary

The unchanged executable source retains the 09:58 UTC result of 43 passing
pilot/discovery tests. That result does not cover the unfinished buffer repair.
The selected Training Wave memory now includes the v5 learning/failure result;
there are no immediate child memories. Earlier notes remain at
`9e19d83a0da7a3abe3fc47b22ec1d6e6e665f4ea:scratch/tla-pilot-verify.md` and the
release completion receipt at the same commit's `scratch/etude-release-status.md`.
Uncommitted pre-compression evidence is preserved in
`.runs/mtg213/context-before-compress-20261104-1745/`.

Check: release extension rebuilt/atomically installed; 50 buffer regressions and focused v6 identity/discovery checks pass, as does the debug Rust `learn_outside_focus_is_complete_and_capacity_errors_are_explicit` test; Ruff and diff checks pass. Gate owns affected suites, full debug/Clippy, current-world admission, training checks and CI collection.

## Current implementation cut

Delete — do not maintain: remove `event_buffer_config` and its vector-only
capacity/default selection. `input_buffer_config` remains the single native
owner of configurable axes for scalar/vector buffers; relation/shared-effect
widths and historical input semantics remain fixed. Validate all companions
before scalar copies, preserving explicit overflow and public-history bounds.
Add controlled non-default capacity and retained 256-action parity fixtures.
The prospective pilot keeps v7, 128 cards, 40 permanents, 128 facts, 512 actions,
2 focus objects and 32 legacy events; no population or recipe change beyond
current runtime binding and the repaired action capacity. Publication and
landing remain the caller Graph's delivery steps.

The active experiment entry moves from `tla_pilot_v5.py` to `tla_pilot_v6.py`
with its analysis/discovery consumers. Delete the v5 entry rather than keep a
second live launcher; frozen v5 protocols, regimes, data and source
`97a2cfb963d7e3a3b2816f31c45dda72f34cdddd` remain the historical execution path.

Fresh read at 2026-11-04 18:10:23 UTC (`.runs/mtg213/v6-continuation/inventory.json`):
no Vast instances, RunPod pods or unsettled hardware; accounted full rental
cost $0.32710677365251912, conservative reservations $3.9139916049382716.
The same $1 allocation fits the cumulative $10 cap. A fresh quote/submission
still owns its exact source and hardware reservation.
