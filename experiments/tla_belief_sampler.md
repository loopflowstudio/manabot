# TLA frozen-player belief sampler (MTG-357)

Exploratory. Collection and fitting have not started. There is no trained sampler,
held-out comparison or public prediction example yet. The experiment CLI described
in [the guide](../docs/tla-belief-sampler.md) remains to be implemented.

## Producer verification, 2026-11-14

The ordinary producer records establish that the staged Small checkpoint is the
raw endpoint of Etude run `59718fe34945`, not the experimental adoption used in
MTG-349. The retained R2 keys are relative to `r2://manabot/`:

| Record | Key | SHA-256 |
| --- | --- | --- |
| Endpoint checkpoint record | `jobs/59718fe34945/checkpoints/learner_transitions-82370334.json` | `7f8659083542eb36ea9bc181aef309466395300fb7669f2958011c642dc16f18` |
| Ordinary producer export | `jobs/59718fe34945/attempts/10d3740cb3a7/training/run.json` | `57910103734db654370fcbd36f766b4351b52fdadcc1464d77f82fa3fa34dc70` |

The checkpoint record names ordinal 80, 1,952 updates, 82,370,334 learner
transitions and tag `regime:tla-frontier-c5fba4-w64`. The producer export's
`self-play` stage names the same updates/transitions and raw artifact; its
`selected_artifact` agrees. All three point to 3,945,456 checkpoint bytes with
SHA-256 `c95c61aa298528e5c2782430d5e3f90929d3e9b7ac8b1fc45f560ed6ab4bbaf8`.
The staged `probe/data/final.pt` matches that identity.

The producer reports source `6cd54d349ccbbaf1f5f2aa10831ccfb3ecf4653b`, whose
tree `f2a131012ea8319b9cd00ac11d43f9d30b62ce04` equals the reported training
revision `ab2b757d9`. Preserve the original absence of `checkpoint_source`.
The producer export reports `paused` at its wall bound; the historical launcher's
`succeeded` outcome does not mean that the original update horizon was reached.

An isolated archive of `ab2b757d9` built with its locked CPython 3.12/Torch 2.10
runtime on the mini. The original `manabot.model.checkpoint.load_checkpoint`
loaded those unchanged bytes. The retained observation configuration is input
v9, card capacity 128 per player, permanent capacity 96, 512 facts, 32 events,
64 actions, `first-legal`, and two focus objects. Staged maintained capture
modules imported in that runtime. This is a loader/import proof; it supplies no
TLA game, sampling-validity or throughput evidence. The complete staged bundle
and native manifest still need durable admission receipts before collection.

## Current implementation

The public frozen-policy collector delegates to one scheduled capture core.
It preserves strict public-call behavior and supports the configured native
action prefix without remapping offer IDs. Scheduled capture supports a fixed
public schema, whole-game splits, bounded label-independent decision selection,
explicit unsupported-row coverage, per-game private shards and attempt receipts.
Continuation does not retry started slots; saved completed evidence is verified
before reuse. The existing fitter and policy loader are unchanged.

Validation on the mini uses an isolated copy of this checkout and the existing
CPython 3.13 native runtime for current-world collector tests. These controlled
and tiny-game checks are software validation, separate from experiment admission.
The historical-runtime import proof above does not establish an end-to-end
historical-runtime capture round trip. The changed test file passed 12 checks;
a subsequent explicit repeated-deal-seed fix passed its new reset check and all
three affected real-collector cases. Ruff and whitespace checks passed.

## Execution dependency

The available Etude 0.3.2 and original 0.2.15 clients both refused the mini's
shared launcher database while reading machine inventory:

> `/Users/jack/.etude/launcher.db was written by another version of etude; move or delete it to start again`

The shared database was preserved. No experiment job, rental, admission game or
sampler fit was launched. No launcher cost receipt exists for this experiment.
A compatible read of the existing launcher/controller inventory is required to
establish available shared capacity and launch under the cumulative USD25 cap.
The shell process snapshot alone is not a launcher inventory.

Next: provide that compatible launcher path, finish the frozen worker bundle and
experiment CLI, then run the separately accounted eight-slot admission under
USD2. Freeze cohort size and fit allocation from its ordinary pace before main
collection. The proposed 1,024-slot schedule and three fit seeds remain planning
choices, not an executed freeze or permission to ignore missing attempts.
