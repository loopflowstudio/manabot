# TLA pilot continuation — reconciled 2026-11-04 18:44 UTC

Jack Heart's accepted outcome is preliminary TLA self-play and paired evaluation
on the frozen 134-deck trophy population. GPUs and enabling repairs are authorized
under a cumulative $10 cap, with each allocation <=$1. The corpus, runners,
Vast integration and GPU learning milestone are delivered. V5 completed three
updates, then failed at 256 offered actions versus capacity 64 before exporting a
trained checkpoint. The native buffer and X-cost repairs are implemented; v7
is frozen but unsubmitted because both bounded GPU quotes refused capacity.
The pilot remains incomplete.

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
failed attempt and initialization evaluation remain frozen. The later rules
investigation established that its 256 X offers were illegal.

The 09:58 verification, reconfirmed at 10:09, found executable source, protocols
and tests equal to main `5dd11cec4100aa0e1f4a23c063dfd5778e2ba28f`. SSH fetch
failed (`Permission denied (publickey)`); authenticated GitHub API and local
diffs established the merge. T6 widened player/card rows to 73/56 from v5's
67/55 and fresh action capacity to 512 from v5's explicit 64. Runtime and shape
guards reject substitution. Synthetic scheduling/analysis fixtures pin historical
identity and widths; they never authorize real replay, encoding or training
under that identity.

The frozen-source replay matches stream 103 / seed 32172 after 76 actions:
scalar, vector and Python encoders all reject 256 offered actions at capacity 64.
Diagnostic Python configuration at 512 retains all 256; frozen native buffers
still enforce 64. A configuration-only retry on that source is insufficient.

Main `5dd11cec` used a native default of 512, but its shared
`input_buffer_config` derived only card/fact capacities and handled events
differently between scalar/vector paths. The repair now reads all configurable
axes from supplied arrays. Shared-effect and relation axes remain version-fixed.
Non-default fixtures establish this contract independently of today's defaults.

Primary operator comment bbc34e15 (2026-11-04 17:49 UTC) selects the routine
buffer repair under Jack Heart's existing enabling-fix and $10 authorization;
no new product choice is needed. The implementation derives every configurable
axis at the shared native boundary, validates companion shapes, and verifies
scalar/vector/Python parity and overflow. Historical configurations and complete
action spaces are preserved. These checks resolve the buffer boundary, not GPU
acceptance.

Prospective v7 binds the repaired X-cost rules, current world/content and
73/56/38 widths, 512 actions and unchanged 134 entries. V6 Run `bcde93e9ff8c`
(source `ad2c3a010f53`) was frozen but never submitted or scored. Operator
comment a85a89e8 (18:11 UTC) required examining its 256 offers before training;
replay identified illegal X=0..255 for an airbend {2} cast of Wan Shi Tong.
Announcement/payment now share the selected base cost, fixing X=0 here;
any-type mana preserves X and free casts preserve additional waterbend X when
the printed cost has no X. Current source replay has no illegal X choice.
Historical evidence stays frozen; no training may use the prepared v6 Run.
The separately observed Vast rental 59143644 / session d123328bc691 belonged to
`state_representation_gpu`, Run 9c7c8e6841a4, source 27a12642200a in Task
5da5f4's checkout. It was left to its owner and was absent by the fresh 18:39 read.

Remaining acceptance: an admitted trained checkpoint, complete paired schedule
and a report of learning and full cost relative to Elemental. V7 Run
`af3661790ac5` is frozen at `b5406bb33abd98870af6ca07c3beed63a92b7c1c`;
`.runs/mtg213/v7-continuation/prepared-run.json` retains its typed Run. The local
evaluation archive is `.runs/mtg213/runtime-b5406bb33`, bound to those bytes and
the rebuilt extension. It supplies no empirical training/evaluation result.

Fresh 18:44:18 UTC reconciliation found no Vast instances, RunPod pods or
unsettled hardware. Accounted whole-rental cost remains
$0.32710677365251912; conservative historical reservations remain
$3.9139916049382716/$10. The RTX 5090 quote refused price above $0.81/hour
(the next offer read was $1.1481481481481481/hour). The existing L4 supply
briefly reported available at $0.59/hour, then refused the actual quote with
no machines available. Neither quote submitted or rented anything. No training,
initialization evaluation, watcher or background process remains. These are
external availability/price refusals, not missing authorization; continuation
still requires a fresh quote/inventory within <=$1 allocation and $10 cumulative.
Source, inventory and refusal receipts are manifest-published under
`experiments/data/mtg213/gpu-v7-preflight-20261104/`.

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

The selected Training Wave memory now includes the v5 learning/failure result;
there are no immediate child memories. Earlier notes remain at
`9e19d83a0da7a3abe3fc47b22ec1d6e6e665f4ea:scratch/tla-pilot-verify.md` and the
release completion receipt at the same commit's `scratch/etude-release-status.md`.
Uncommitted pre-compression evidence is preserved in
`.runs/mtg213/context-before-compress-20261104-1745/`.

Check: `uv run --extra dev --extra artifacts --extra evaluation pytest tests/verify/test_native_buffer_capacities.py tests/eval/test_tla_pilot.py tests/eval/test_experiment_files.py -q` — 96 passed with the rebuilt, atomically installed release extension; Rust formatting and Ruff pass. Earlier directed debug X-cost checks remain at `3424e0543:scratch/tla-pilot-verify.md`. Gate owns affected suites, full debug/Clippy, current-world admission, training checks and CI collection.

## Current implementation cut

Delete — do not maintain (completed): `event_buffer_config`, incoming/default
configuration overrides in `input_buffer_config`, the printed-cost-only X
announcement predicate, and predecessor `tla_pilot_v5.py`/`tla_pilot_v6.py`
launchers. No remaining deletion targets.

`input_buffer_config` derives configuration solely from supplied arrays, using
one row/width reader for ordinary, relation and fact buffers. Callers validate
complete companion shapes before writes. `casting_mana_cost` owns the selected
cost for announcement and payment. Tests retain non-default parity, overflow,
historical 256-offer transport and repaired live-prefix parity. Protocol tests
share the historical contract fixture and assert all preserved recipe fields
together; real replay uses the explicit pilot allocation without a default override.
The prospective pilot keeps v7, 128 cards, 40 permanents, 128 facts, 512 actions,
2 focus objects and 32 legacy events; no population or recipe change beyond
current runtime binding and the repaired action capacity. Publication and
landing remain the caller Graph's delivery steps.

Only `tla_pilot_v7.py` is active, with its analysis/discovery consumers. Frozen
v5 protocols, regimes, data and source `97a2cfb963d7e3a3b2816f31c45dda72f34cdddd`
remain the historical execution path. Simplification does not rebind prepared
v7 Run `af3661790ac5` from its frozen `b5406bb33` source.
