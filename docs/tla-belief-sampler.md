# TLA frozen-player belief sampler

Implementation status (2026-11-14): scheduled capture is implemented and the
original policy loads in its isolated runtime. The experiment CLI below is the
planned interface and is not available yet. Admission, fitting and saved demo
artifacts remain outstanding; see the [current report](../experiments/tla_belief_sampler.md)
for the launcher compatibility dependency.

Inspect the saved sampler and its held-out comparison from the repository root:

```sh
uv run --extra artifacts python experiments/tla_belief_sampler.py demo --artifacts experiments/data/tla-belief-357
```

The demo reloads fit seed 357 against its recorded dataset, schema, policy and
world hashes. It reproduces the held-out statistics and emits a public text/JSON
example with visible commitments, prior and learned card-inclusion probabilities,
and learned predictions with history zeroed. The example contains no true hidden
hand or private game seed. Zeroing history illustrates sensitivity; it does not
isolate the value of history in a controlled comparison.

This exploratory workflow trains a hidden-hand sampler from complete TLA games
played by the frozen Small u1952 player from Etude run `59718fe34945`. A negative
comparison is a useful result. Prediction scores do not establish better search
or playing strength.

## Prepare the frozen worker

Collection and fitting run serially on an admitted remote worker. Check its
existing work before using shared capacity. Offline reload and reporting can run
locally. Use the repository's locked uv environment and artifact credentials
injected through Doppler; see [development setup](development.md#environment-and-commands).
Keep the original policy runtime isolated from the current source environment.

Admission requires the ordinary producer export identifying Small/raw/u1952,
checkpoint bytes and digest, the original configuration and native manifest, and
the compatible source tree. A launcher specification alone does not establish
the checkpoint's producer. The original checkpoint has no `checkpoint_source`;
retain that absence and verify provenance through its producer records. The
experimental adoption used for deck-builder work is a separate identity and
cannot substitute for this player.

The capture schema covers all card, token and face names in the original native
manifest. Its vocabulary comes from public definitions, never hidden labels.
Keep the player's saved observation capacities and deterministic first-64 legal
action support. Native legality still supplies every choice; collection records
overflow and preserves the retained choices' native offer IDs.

## Admit, then collect and fit

Run admission on the remote worker before freezing the main schedule:

```sh
uv run --extra artifacts python experiments/tla_belief_sampler.py admission --out experiments/data/tla-belief-357 --max-usd 2
```

Admission verifies producer bindings and the frozen execution bundle, accounts
for eight game slots, and records collection, fit, evaluation and serialization
pace. Use those measurements to freeze the final schedule and work allocation
in the experiment report before main collection. The admission subcap counts
toward the USD25 whole-experiment ceiling, including setup, failures, fitting,
evaluation and teardown. The launcher owns the cost receipt.

With producer inputs and the final schedule frozen, run:

```sh
uv run --extra artifacts python experiments/tla_belief_sampler.py run --out experiments/data/tla-belief-357 --max-usd 25
```

The command collects the declared schedule, fits three independent samplers and
retains their artifacts for reporting. Its cap is cumulative with admission;
it does not add USD25 to an earlier allocation. Collection shards and each fit
are separate durable jobs so completed evidence can be delivered before later
work starts.

The initial allocation is at most 1,024 game slots, with swapped assignments
kept together in a single split, and up to eight decision moments per game for
both viewers. The final report records any prospective reduction needed to fit
the cap. Whole games are assigned to train, validation or test before play.
The test panels distinguish new deals from reserved deck IDs: those deck IDs
are held out from sampler fitting, but the frozen player trained on all 134
field entries.

Rows are selected with a seeded, label-independent reservoir while viewer
histories advance at every transition. Actual hidden hands are private
supervision only. Only complete games contribute fitting rows. Native trivial
decisions remain skipped; reaching the 2,000-decision limit fails the game.

Fit seeds are 357, 358 and 359. Initial fit settings are 512 optimizer steps,
batch size 32, hidden size 64, learning rate 0.001, history dropout 0 and 16
evaluation samples during fitting. The frozen report specifies the final
allocation; test scores never choose a checkpoint or change it. Seed 357 supplies
the demo regardless of its score.

## Read the comparison

Saved reports compare learned predictions with the compatible physical-deal
prior at 16 and 64 samples. Joint hand NLL is measured in nats; card-inclusion
Brier, ten-bin inclusion calibration error, conjunction Brier and sample support
violations describe different aspects of prediction quality. Lower NLL and
Brier are better. Learned-minus-prior differences therefore favor the learned
sampler when negative for those losses.

Read equal-game means separately from pooled decision-weighted measurements.
Whole-game intervals describe uncertainty across games; variation across the
three fit seeds is reported separately. Inspect both test panels, attempted-game
denominators, row-selection coverage and omissions alongside the scores.

The model's history features are order-invariant, square-root-normalized public
commitment counts. They omit event order and most board context. Count-only
sampling cannot represent retained-selection and foretold-slot constraints:
those viewer rows are explicitly omitted while the game and both histories
continue. Metrics condition on representable rows from completed games;
failures and omissions can bias the comparison. Native materialization checks
on fixed replay-backed observations are reported separately from count-bound
support checks. Neither establishes an exact posterior.

## Proposed continuation contract

Use one private evidence directory for admission, `run` and `demo`. Admission
and main collection occupy separate subdirectories; admission games never enter
the main dataset. The directory retains references to Etude jobs and their
launcher cost receipts, so the cumulative USD25 limit survives continuation.
This interface remains proposed; the collector alone does not implement it.

Repeat the same `run --out` command to reconcile existing jobs and continue
untouched work. Unknown live jobs or costs stop new dispatch. Verify completed
outputs before reuse; never replay started games or silently restart a failed
fit. An interrupted fit without a verified complete candidate stays unavailable;
other declared, untouched fit seeds may proceed within the remaining cap.

Before fitting, close collection accounting: every slot is completed, failed,
interrupted or explicitly unattempted at a recorded stop, and no worker may
still write. Save one immutable dataset for all fits and reports. Three
nonempty splits alone do not establish closure. A completed game with no saved
representable rows remains in coverage, even though it supplies no fit rows.

For example, if a worker stops after writing a game shard but before its result
receipt, retain the shard as interrupted evidence and exclude its rows. Reconcile
the old job, restore any fatal prerequisite, then continue untouched slots.
A checksum-valid shard does not prove completion. Keep a fatal stop's cause
private; marking its slot interrupted cannot erase the reason collection stopped.

## Failures, continuation and private evidence

Each scheduled slot has a receipt before reset and ends completed, failed,
interrupted or unattempted. A finished game's private shard is persisted before
the next slot. An isolated game failure discards both viewers' rows from that
game and retains its reason and replay prefix. There are no replacement games.
Continuation reads receipts and never replays a settled slot; an interrupted
attempt stays in the accounting.

Identity mismatches, infrastructure errors, malformed constraints, unexpected
schema names or capacity overflow, and corrupt artifacts stop the job. Inspect
the retained failure and restore the required producer, runtime or artifact
before continuing; do not relax bindings, expand a fitted schema or substitute
a newer checkpoint. Missing split data makes fitting unavailable, with no
invented empty metrics. Unsupported-row omission applies only to the recognized
retained-selection and foretell states.

Keep hidden labels, game seeds and replay prefixes in private artifact storage.
Before publishing through `manabot.data`, verify the private evidence destination;
publishable summaries and the demo expose only viewer-safe information. Follow
the [artifact workflow](../experiments/README.md#data) to publish result bytes,
record their manifest and keep data files out of git. Retain all three sampler
fits and their exact bindings. The result is a candidate for later belief-guided
search; this workflow ends with the saved demo and report.
