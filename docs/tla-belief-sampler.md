# Belief sampling from player games

Inspect a saved TLA sampler and its held-out comparison from the repository root:

```sh
uv run --extra artifacts python experiments/tla_belief_sampler.py demo --artifacts experiments/data/tla-belief-357
```

Supply a downloaded evidence directory containing the dataset, completed seed-357
fit and its receipts. The command reloads the sampler, checks its recorded byte,
data, schema, producer and world bindings, and writes reports and a viewer-safe
text/JSON prediction under `demo/`. It prints the JSON example's path. It performs
no collection or fitting. If the files are missing, restore the exact saved
artifacts before retrying; a directory name alone does not identify a candidate.

The sampler predicts hidden-hand card counts for gameplay search, correcting
native physical-deal probabilities using public constraints and public action
commitments. Actual hidden hands supply private training targets. Held-out
prediction quality measures this sampler; search integration and playing-strength
comparisons are separate work.

The snapshot data APIs, equal-game sampling, saved-fit admission and offline
reports are implemented. Player-collector export, native capture before autoreset
and the Etude jobs described below remain implementation work. There is no
collector export switch or connected handoff command available yet; `demo` is
the current experiment CLI.

Local producer publication state is implemented in
`manabot.belief.sampling_publication`. It retains acknowledged membership, the
fixed split plan, an output frontier and pending/verified snapshot references
in `producer.json`, alongside immutable `games/` and `snapshots/` bytes.
Connecting these outputs to the maintained SelfPlay loop and Etude's saved-output
restoration remains implementation work; local fixtures do not prove remote
durability or learner recovery.

The reusable APIs are `seal_game`, `seal_snapshot` and `publish_snapshot` in
`manabot.belief.sampling_snapshot`. `read_dataset` admits the self-contained
snapshot; `fit_saved_sampler` takes that file and an attempt output directory.
Run fitting only on the admitted remote worker. Snapshot checkpoints reload
with `expected_snapshot_binding`; single-policy checkpoints retain
`expected_policy_identity`.

## Reuse completed player games

Enable private belief-game export on the player-training collector. Export is
off by default. Dedicated frozen-player collection produces the same game data.
Both sources retain complete games, per-seat behavior identities, replay
provenance, selected private labels and coverage. Enabling export preserves the
player's action sampling and learning batches; publication adds elapsed time.
It does not enable tolerance of game failures.

For example, select the completed games from a player's first update and publish
snapshot A. Give A to a belief fit while player training continues. After the
next update, publish snapshot B with the additional qualifying games. The fit
and later reload of A still use A's original membership and bytes. B is a new
input and requires an explicitly selected fit of its own.

Select producer runs and update/checkpoint ranges explicitly. Each seat records
its actual raw/EMA weights generation and digest, with a checkpoint reference
when one exists. A game that spans updates retains its ordered behavior
segments; every segment must qualify for the selected range. An unsaved
intermediate generation is still identified, but is not a published checkpoint.
Earlier-player games and final-player games describe different distributions.
An all-history selection must not be reported as final-player data.

A snapshot records the selection, publication watermark, game membership,
shard digests and sizes, producer identities, world/schema, splits and coverage.
It includes only sealed shards with successful completion records. A completed
game with no representable rows remains in coverage. Partial files, orphan rows
and failed or interrupted games supply no successful-game labels. Other games
may still be running when a snapshot is sealed.

## Deliver a snapshot and fit it

Run collection and fitting on an admitted remote worker. Offline reload and
reporting can run locally. Use the locked uv environment and inject artifact and
provider credentials through Doppler; see [development setup](development.md#environment-and-commands).
Verify private destination access before sending hidden labels.

Publish a self-contained snapshot file containing its manifest and selected
shards through the immutable artifact store. Publication is synchronous and
bounded at update boundaries. A snapshot becomes available only after storage
verifies its bytes. A metric or log message describing it is not a delivery
acknowledgement. Etude supplies the selected file to a fit as a named `Input`
with its exact URI, digest and size. The fitter reads those bytes rather than a
changing producer directory. Missing, corrupt or mismatched input fails admission.

The producer must retain its acknowledged game index, fixed split plan,
publication frontier and pending snapshot identity with its saved outputs.
Record the verified storage reference before making a snapshot available to
fits. Job declarations use that retained reference; they cannot infer successful
publication from logs or reconstruct membership by scanning files.

The internal producer API fixes bindings with `create_producer`, acknowledges
successful closures with `retain_game`, and retains already-persisted failure
evidence digests with `retain_failure`. `prepare_snapshot` derives the cut from
that index and retains exact pending bytes. `publish_pending_snapshot` reconciles
the same content key after a lost reply and saves its verified reference before
returning it. Fit declarations read only `read_producer(...).verified`.
These functions supply no game loop or launcher. The producer output directory
must survive independently of learner saves; reopening fixes neither a missing
output index nor missing shard bytes by scanning for files.

Declare Etude training jobs in `experiments/tla_belief_sampler.py` for the player,
snapshot fits and frozen-player reference. Each fit selects an already published
snapshot. Etude owns execution, work deduplication, attempts and costs; the
producer owns completed games and the snapshot owns fit membership. Use the
experiment report's admitted jobs and caps for connected execution. The
saved-artifact `demo` command above is the inspection entry point.

The fitter samples a nonempty training game uniformly, then a saved row uniformly
within that game. Thus a game with eight saved rows has the same total training
weight as a game with two. Zero-row games supply no gradient. No completed games,
or an empty required split, makes fitting unavailable; it does not yield a zero
loss.

Assign seeded game/deal-group splits before capture and retain them across
snapshots. Both viewers and swapped legs of a deal stay together. Validate group
membership across all selected snapshots: a test group in A cannot become a
training group in B. Repeated numeric seeds in independent producer runs do not
by themselves identify the same game or dependent group. Game identity includes
the producer run, original attempt, stream and game ordinal.

Retain each candidate's training snapshot, producer selection, fit configuration,
seed, checkpoint digest, schema and world. Reload verifies these bindings.
Reports also identify the evaluation producer distribution and mark a shift from
training. Older single-policy datasets and checkpoints retain their original
bytes and binding meanings.

## Collect the original TLA reference

The first TLA reference uses the unchanged Small/raw/u1952 player from Etude run
`59718fe34945`. Keep its runtime isolated from current-world player training.
Admission requires the ordinary producer export, original checkpoint bytes and
digest, saved configuration, native manifest and compatible source tree. The
checkpoint lacks `checkpoint_source`; preserve that absence and verify its
provenance through producer records. The deck-builder's adopted player is a
separate identity.

Keep the player's saved observation capacities and deterministic first-64 legal
action support. Native legality supplies every choice; overflow diagnostics
record the policy's restricted support. The capture vocabulary covers the
original manifest's public card, token and face names.

Before main collection, use eight frozen-runtime admission slots to measure
capture, fit, evaluation and serialization pace. The initial planning ceiling is
1,024 TLA attempts, up to eight decision moments per game, three fit seeds and
27 reserved deck IDs. Freeze the final schedule and any prospective reduction
in the [experiment report](../experiments/tla_belief_sampler.md) before main
collection. Admission games and the current-world training-game handoff proof
remain separate from the TLA reference data.

Rows are selected by a seeded, label-independent reservoir while both viewer
histories advance at every transition, including built-in continuation. Capture
retains terminal game observations before autoreset. Native trivial decisions
remain skipped; the 2,000-decision limit fails a game. Reserved deck IDs are held
out from sampler fitting; the frozen player trained on all 134 field entries.

Fit seeds are 357, 358 and 359. Initial settings are 512 optimizer steps, batch
size 32, hidden size 64, learning rate 0.001, history dropout 0 and 16 evaluation
samples during fitting. Admission pace determines the final allocation within
the remaining cap. Test scores never choose a candidate or change the allocation.
Seed 357 supplies the demo regardless of score.

All setup, handoff proof, admission, attempts, fitting, evaluation and teardown
share the cumulative USD25 ceiling. The USD2 setup subcap is included. Check
shared launcher inventory, writer ownership, costs and outstanding commitments
before launching. Size job caps for startup as well as execution. A per-launch
`--max-usd` does not account for earlier passes: derive the remaining ceiling
from the launcher's retained costs and commitments. Failed inventory leaves
capacity and commitments unknown. Preserve the shared launcher database and
resolve compatible access before further execution.

## Read the comparison

Saved reports compare learned predictions with the compatible physical-deal
prior at 16 and 64 samples. Joint hand NLL is in nats. Card-inclusion Brier,
ten-bin inclusion calibration error, conjunction Brier and support violations
measure different properties. Lower NLL and Brier are better; negative
learned-minus-prior differences favor the learned sampler for these losses.
The physical-deal prior is not a claim of uniform sampling over every distinct
valid world.

Read equal-game means separately from pooled row-weighted measurements.
Intervals resample whole declared deal groups, keeping available swapped legs
and both viewers together. Variation across fit seeds is reported separately;
fits sharing one snapshot are not independent data-generation seeds. Inspect
same-field and reserved-deck panels, expected/attempted/completed-game counts,
zero-row games and omission reasons alongside the scores.

To include grouped panels in the saved demo, add `--cohort <saved-cohort.json>`.
This reads a saved fit-cohort declaration and reports without fitting; its report
paths resolve from the evidence directory. TLA panels derive expected games from
the closed finite schedule, including unattempted and zero-row games. Every test
slot is assigned to `same-field` or `reserved` before collection.

The model uses square-root-normalized, order-independent public commitment
counts. Its GRU decodes card counts, not game time. It omits event order and most
board context. Count-only sampling cannot represent retained-selection and
foretold-slot constraints: those viewer rows have explicit omission reasons
while the game and histories continue. Metrics therefore condition on
representable rows in completed games. Failures and omissions can bias results.
Native materialization checks on replay-backed observations are separate from
count-support checks; neither establishes an exact posterior.

The public example uses the first saved test row with nonempty public history,
without selecting on its predictions. It shows normalized commitments, prior
and learned inclusion probabilities, and predictions with history zeroed.
If no such row exists, the example is unavailable. History zeroing illustrates
sensitivity and does not isolate history's causal value. The example exposes no
true hidden hand, private seed or replay prefix.

## Failures and continuation

Successful game closure and its private shard are persisted together before
acknowledgement. Republishing identical bytes for the same game is idempotent;
changed bytes under the same identity are corruption. Collector reconstruction
retains original game identities and must not publish replayed prefixes as new
games. Strict batch failures cannot publish an unverified successful subset.
Export errors stop before the next optimizer update while preserving completed
prior work.

Restoring an older learner checkpoint must preserve later acknowledged game
outputs and failure evidence. Reconcile those outputs with the restored cursor;
never advance learner state from a publication frontier. If the maintained loop
starts fresh native streams on resume, retain abandoned games as incomplete and
give newly started games distinct identities. That is not exact game continuation.

An interrupted upload may resume the same content key after verification. It
does not authorize more games or a new fit attempt. Etude owns attempt identity
and acknowledged saved state. An interrupted fit without a verified completed
candidate remains unavailable; never adopt an orphan candidate or blindly
restart the same seed. Saved artifacts remain inspectable without live launcher
access.

For finite frozen collection, every scheduled slot ends completed, failed,
interrupted or explicitly unattempted at the recorded stop. Preserve failure
causes and replay prefixes; there are no replacement games. Close this accounting
before the reference fits. An ongoing player producer can publish a snapshot of
closed games without declaring its entire run complete.

Identity mismatches, infrastructure errors, malformed constraints, unexpected
schema names, capacity overflow and unclassified native errors stop the job.
Recognized retained-selection and foretell states omit only the affected viewer
rows. The frozen capture adapter recognizes native panics and the known
`no legal targets` active-step failure, including its wrapped native cause;
other native errors remain fatal. Inspect retained diagnostics and restore the
required runtime, producer or artifact before continuing untouched work. Keep
fatal causes distinct from interrupted-slot status. Do not relax bindings or
substitute a newer checkpoint to make evidence load.

Keep hidden labels, seeds, prefixes and raw failures private. Follow the
[artifact workflow](../experiments/README.md#data) to publish result bytes to a
verified private destination, record their manifest and keep data files out of
git. Retain all three fits and their exact bindings. This workflow ends with the
saved report and demo for review.
