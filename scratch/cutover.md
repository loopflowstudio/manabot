# Execution extraction: implementation direction

2026-11-13, Jack Heart: "lets go" after the recipe-structure clarification in
`plan.md`. The current implementation and remaining acceptance checks are in
`plan.md`; this note preserves the extraction direction.

## Outcome

Delete `manabot/training/execution.py` by moving its general ML operations
responsibilities to reusable Etude tools and keeping a readable Manabot
training loop. Preserve the overall structure of recipe and experiment files,
with explicit training, evaluation and launch functions. Do not turn a recipe
into a nested `ElementalConfig.self_play` settings wrapper. Inspect the existing
entry points and preserve their roles; do not mechanically rename existing
`runs()` launch declarations just to fit a newly invented interface. Small
typed configs for actual varying inputs are fine.

The prototype's parity evidence, learner/state functions and game-error
handling are useful. Its imposed recipe/config structure is not the target.
Do not replace the large executor with a renamed orchestrator. Keep world
bindings, replay evidence, paired deals and training seeds intact. Saved
regimes and frozen experiments must stay readable or name their last runnable
revision; do not silently rewrite frozen evidence.

## Etude dependency observed 2026-11-13

At implementation handoff, EXP-56 had a live code Graph at `pr-publish`.
Realign rechecked its PR #103: still open at `105abdd`; no release is implied.
The source in `/Users/jack/src/etude.exp-56-give-a-training/src/etude/` provides:

- `train.run.local_launch(out=, experiment=)` for local process context.
- `Run.out`, `Run.stopping()`, and `Run.until(clock, default=)`.
- Planned-stop `Stopped`, training exit 75, retries within attempts/budgets.
- Run out carriage/restoration and continuation from parent out.
- First logged rows establish retry replacement cutoffs.
- In-out checkpoints are recorded in place and must be immutable.
- `jobs.training(train, config=, smoke=, **options)` calls an ordinary typed
  function. The transport config is a Pydantic value; this does not require a
  nested self-play config or a new recipe framework.

Local tracking files are `DIR/etude/BASENAME(DIR)/`, not top-level `DIR`.
Launched out is `RUN/out/`. A continuation's target comes from
`run.until("updates", default=...)`; do not change recipe config just to extend
it. Find the exact `Stopped` defining module before importing it.

EXP-56's publication notes say its copied Manabot local smoke/resume passed,
but the real Manabot launch cutover is unverified. Check current dependency
delivery/version before pinning. An unreleased API can be tested locally with
an isolated dependency override; do not commit a path into another checkout
or claim a shipped SDK contains it. If release is still needed, make that a
specific dependency at review while completing independent implementation.

## Implementation and verification

Reconcile docs to this scope first. Switch actual callers to Etude primitives,
remove the local Etude stub, migrate current entry points, and delete obsolete
executor/job/recovery plumbing only after checking its responsibilities are
covered. Resolve multi-step belief output caching so extending training cannot
reuse histories or a sampler from the old policy. Published checkpoint paths
must not be overwritten after Etude records their bytes.

Read current main and preserve its changes (including MTG-304's learning
modules); use `lf sync` for managed synchronization. Run focused checks per
AGENTS.md and docs/development.md, with all Python through uv, no artificial
load and no test over five minutes. Exercise fresh run, rerun, planned stop,
resume, target-only continuation, and the belief outputs' policy provenance.
Use bounded fixtures for launcher behavior. No remote paid run is necessary
to begin the cutover. Report any missing integration proof precisely.

Earlier prototype code Graphs all stopped at compress after documentation and
implementation succeeded; the last failure was the provider usage limit.
This run revisits docs and implementation because the recipe scope changed.
Existing local naming edits (`verify_checkpoint`, AGENTS.md) and their notes
belong to this Task and must be retained.
