# TLA pilot capacity and recovery repair — 2026-11-04

## Accepted outcome and current slice

MTG-213 still requires a trained GPU checkpoint, complete paired evaluation and
a qualified report. V9 Run `2171156493ca` failed after five updates / 251,040
transitions at 41 permanents against capacity 40; initialization scored 7–25,
with 256 scheduled games missing. Evidence and confirmed rental teardown are
retained in `6dff2e919` and `experiments/tla-trophy-pilot.md`.

The verification feedback selects a prospective repair and fresh initialization.
Use the existing `max_permanents_per_player=64` configuration; do not widen a
frozen input version, truncate permanents or alter global defaults. Preserve all
134 decks, v7 effects, card128, matched first64 support, recipe and schedule.
This capacity fits the retained failure, not every reachable battlefield.
Jack Heart subsequently accepted isolated resource-limit recovery at .25
(comment ca1660e8); `scratch/resource-limit-direction.md` owns that direction.

## Delete — do not maintain

- Retired `experiments/tla_pilot_v9.py`: current discovery/analysis use v10;
  historical execution uses source `6da60684c`. Preserve v9 protocol/regime/evidence.
- Removed current README references to absent `tla_pilot_v5.py` and its v5
  usage protocol; current usage points to v10, historical receipts stay linked.
- Removed the duplicate capacity-recovery explanation from the README's Etude
  jobs section; `docs/training.md#game-failures` owns it, with README links.
  Preserve recovery boundaries, denominator, defaults and evaluation limitations.

No deletion targets remain. The surviving runtime uses the existing collector,
ledger and sampler; no parallel recovery authority is needed.

## Implemented in the prospective successor

- V10 schema4 binds permanent64, tensor shapes, regime hash, predecessor v9,
  USD1.25 allocation and USD10 cumulative cap. Discovery/admission reject changed
  capacity, recovery policy and old initialization. Frozen v9 bytes remain intact.
- The offline stream43 / seed32624 / 371-action fixture covers old40 rejection
  and exact41/pilot64 parity through Python, native scalar and vector buffers.
- Typed active-step capacity errors opt into the existing ledger as
  `resource_limit`; setup/reset, buffer and provenance failures remain fatal.
  Recovery preserves other streams and prior completed games, discarding only
  the affected game's uncommitted samples from both seats. Receipts retain the
  encoder contract for replay. Controlled tests cover isolation, sharded
  transport, learnable retained rows and threshold boundaries.
- The accepted .25 threshold uses a 100-settled-game window and warmup of 100.
  Completed games, engine errors, resource limits, truncations and action limits
  enter that denominator; run interruptions do not. 25/100 continues and 26/100
  stops. Evaluation failures remain in their original scheduled slots. This
  recovery biases training toward games that fit and is not universal capacity.

## Remaining acceptance

Gate reviewed the affected recovery, encoding, current-world admission and
debug-native results required by docs/development.md and AGENTS.md; see
`scratch/tla-resource-recovery-gate.md`. Publication is complete; merge, remaining
CI checks and final-source freezing remain unfinished. Execution then requires fresh
inventory/budget reconciliation, CUDA admission, bounded GPU training from a
fresh initialization, trained checkpoint exports, the paired schedule, confirmed
teardown and a qualified report. No v10 paid execution or trained endpoint is
established. Existing v9 initialization cannot fill v10 slots, and missing v9
results remain missing. PR #423's merge did not complete this outcome.
Retain overflow frequency and affected decks/positions for the next capacity
decision (Jack Heart, comment c98cb420). Experiment 690c9b separately owns
common-case representation overhead; it may conclude no architecture change
is worthwhile and does not expand this repair.

Main's `9f42e4e58` renamed the bound Wave from training to experiments and
`19bdee944` removed Eval memory; neither changed the pilot runtime. The current
Graph selects experiments memory curation. No child memory directories exist.
Earlier launch notes remain at `6dff2e919:scratch/tla-pilot-postmerge-verify.md`.

Latest settled estimate USD0.475343889; conservative reservations
USD5.101645926 of USD10. Use USD0.97/hour / USD1.25 whole-rental ceilings after
fresh reconciliation; do not touch another Task's rentals. Paid execution must
retain CUDA admission and initialize afresh, never retry frozen v9 unchanged.

Replay reuses one fixed buffer allocation; the preservation test compares v10
directly with frozen v9. Native code is unchanged by this simplification, so its
prior debug isolation tests and atomic release-extension rebuild remain applicable.

Check: `uv run python` static documentation path/anchor, v10 protocol/regime hash, capacity/support, recovery-policy and budget assertions plus `git diff --check` — passed on `62f20f8c8`; the documentation correction is already present, with no further runtime or documentation repair needed. Prior v10 admission/frozen-v9 checks and 146 focused passes remain retained evidence; remaining affected suites belong to gate.

## Retained Airship investigation

Exact replay only: player1 cast one Airship at action94, turn20. A qualifying
end step doubled 1→2 at106; turns22/24 doubled 2→4 and4→8; turn28 resolved eight
existing triggers to16. Turn30 sacrificed one, leaving15; ten of its15 triggers
resolved before the capacity failure, leaving25 Airships and five pending
frames. Each copying phase had that controller's sacrifice history and matching
Airship source frames. No same-end-step recursive trigger growth was observed.
The sequence is consistent with the card definition and its existing directed
TLA tests; it does not certify every rules interaction.

Across the retained prefix, observed peaks: 41 permanents/40 old slots (64 new),
75 card rows/128, 25 facts/128, 27 effects/128, 27 members/2048, zero details/512,
18 relations/64 and 26 legal actions/64. All prefix observations encode at64.
More qualifying turns can exceed these finite limits. Capacity recovery, not
indefinite allocation escalation, is the accepted bounded pilot treatment.
The 256/512-row, grouping, sparse-attention and GPU minibatch-memory ideas in
comments 52d6354e, 173f02a6 and c98cb420 remain design discussion; no recipe or
attention changes are selected.

## Documentation ownership

Concept review selected the bounded usage correction, now applied: README
introduction, protocol link and local evaluation command point to v10, retaining
uv, `ETUDE_OUTPUT_DIR` and the pinned-source/native-world requirement. Historical
runs remain linked to their evidence. The game-failures guide owns shared
isolation, settled-game accounting, replay and fatal boundaries once, then names
the capacity opt-in and v10's policy. README links to that guide. No runtime,
protocol, regime or historical evidence changed in this documentation pass.

The one-page review remains `scratch/concept-review.html`. Full pre-reduction
notes are preserved as git blob `a10bb3a7d8edb6ed871de457c48510fc97fdd182`;
the original proposed copy is blob `efdaa0114cca17bb53ee33505d213f2303a36409`.
No unresolved product choice blocks this slice.

## Concept review outcome — 2026-11-04

No further design change and no open product choice. The previous usage mismatch
is resolved: the README names the present v10 runner and the shared game-failures
guide owns recovery. Code inspection agrees with Jack Heart's accepted game-local
boundary: the collector removes only the failed game's suffix from both seats,
the existing ledger retains its input contract and allowance, and native reset
replaces only the affected stream. Normal completion remains learnable; an
unencodable replacement root remains fatal. These counterexamples require no
additional recovery concept or broader capacity/representation change.

The remaining gap is execution evidence, not an unsettled product decision.
The next proof remains the affected gate checks followed by final-source frozen
GPU execution, checkpoint export and the original paired schedule under the
accepted cumulative budget. This review inspected implementation and test
definitions; it did not rerun suites or establish a v10 training result.

## Publication validation — 2026-11-04

`lf sync` completed without conflicts. Reviewed the implementation against local
main `510b5ec35` and the remote publication base `3de3a0ebf`; the latter also
includes existing local-main Wave metadata changes, which are preserved.
No runtime or protocol repair was needed during publication review.

- Full debug native suite with `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2`:
  969 passed, two ignored across 28 targets.
- `cargo fmt --check`, all-target/all-feature Clippy with warnings denied, and
  Ruff over changed Python modules passed. Clippy initially could not resolve
  a relative PyO3 interpreter path; using the absolute root-venv path passed.
- Real `tla_pilot_v10.frozen()` admission passed: w17, 134 decks, permanent64,
  resource recovery enabled at .25 with 100/100 window/warmup.
- All 166 added artifact files match their manifest hashes/sizes. Historical
  manifest entries and frozen v9 protocol/regime bytes are unchanged.

The required Python recovery, encoding, admission, monitoring, discovery and
pilot suites passed: 320 tests in 447.83 seconds. This includes recovery,
game-error recovery/ledger, active training, observation encoding, action cutoff,
native buffer capacities, Elemental world, evaluation models/mirror admission,
monitor evaluation, choice separability, regimes, experiment discovery, the TLA
pilot and untracked experiment-data checks. Logs are `/tmp/mtg213-publish-python.log`,
`/tmp/mtg213-publish-rust.log` and `/tmp/mtg213-publish-clippy.log`.
Publication is separate from merge and Task completion; no v10 GPU execution,
trained endpoint or completed paired evaluation is established by these checks.
