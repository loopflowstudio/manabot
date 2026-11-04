# TLA pilot continuation — 2026-11-04 UTC

**Fix forward; incomplete.** Main at
`fb5ad2c1372f9c967b550f0b8ae0f84883f75563` contains PR #419's shared-effect
contract but lacks the card-capacity repair. PR #416 merged at
`3b122de9078c985f846a0f3c5a3a42ebca3b1182`; this Task has no later PR.
The existing repair and prospective v5 runner are ready for gate and a new PR
in this Task. GPU execution and learning acceptance remain outstanding.

## Accepted outcome and constraints

Jack Heart's 2026-11-03 direction selects preliminary self-play and paired
evaluation on independently sampled decks from the same frozen 134-entry trophy
population. Acceptance requires GPU learner updates, an admitted trained
checkpoint, complete evaluation and an honest comparison of learning per
collected decision/compute/wall time with the evidenced Elemental setup. An
honest negative result is possible; initialization scores and throughput alone
establish no learning. Live fetching and held-out decks remain deferred.

The cumulative $10 paid cap covers setup, failures, retries and cleanup; shipping
enabling changes after checks is authorized. Preserve every executed identity
and all evidence. No truncation, deck exclusion, failed-game replacement or
inherited budget. Choices and attribution: `scratch/questions.md`; results:
`experiments/tla-trophy-pilot.md`.

## Retained evidence and delivered prerequisite

Task 5da5f4 delivered v7 in PR #419, merged at 05:55:57 UTC with executed CI
checks passing. Complete effects route to affected objects before attention:
128 × 18 headers, 512 × 3 details, 2048 × 2 memberships and 128 local facts.
Fresh library defaults remain v6. Factoring preserves provenance and
multiplicity; redundant-activation merging was not proved. The inherited
keyword grant/removal ordering defect remains separate. Padded v7 effects plus
local facts occupy 37,888 bytes versus v6/256's 12,288; this establishes no
memory, speed or learning advantage.

V1 exceeded 128 facts at 140; v3 exceeded 256 at 270. Delivered v7 passes both
exact retained prefixes in scalar/vector/Python encoding while old versions
retain their failures. V3 source is the PR #416 merge above; all 120 artifacts
remain published. PR #415's later 306-list admission audit does not expand the
134-entry population. Runtime remains w17 / card digest
`dcb9d47c674a657176c2a9d56e016288d1e175729aa09cd60f0dc333633330ed`.

V4 source `0b6c2bb854ba48b37fb7e5ac360c0c9d047cea34`, run `baa8671d6480`,
cohort `mtg213-pilot-v4`, used RTX A6000 after an L4 quote refusal. That authorized
substitution was recorded before scoring. CUDA admission passed; collection
failed after 104.74 seconds with 238 completed-but-unoptimized games and 19
active prefixes, **zero learner updates**. Stream 22's exact 300-action prefix
reproduces 45 card objects (40 originals + 5 tokens) plus 16 sideboard cards:
`agent_cards plus sideboard 61 > 60`. Initialization evaluation has 32 valid
games, 2 wins / 30 losses, no failures/truncations; analysis retains 288 expected,
256 missing, complete=false, success=null. Advertised `etude-result.json` is
absent. All 110 retained artifacts are published under
`experiments/data/mtg213/gpu-v4-20261104/`. Frozen inputs and evidence stay intact.

## Surviving implementation

V5 binds policy_input_version=7, max_facts=128 and max_cards_per_player=128.
The existing card capacity is an operator allocation, not a representation
change or universal bound. Regime hashes own configuration; schema-3 protocol
shapes reject encoder allocation drift. Recipe, seeds, 100-update target,
milestones and eight-pair/four-leg schedule remain unchanged. Model object
capacity changes require fresh initialization/evaluation.

The shared native `input_buffer_config` derives card capacity from scalar/batched
buffers and validates consumed arrays against it. Previously it retained 60
regardless of allocation. The retained prefix agrees across native scalar,
vector and Python at 61 and 128 rows; 60 rejects without dropping tokens or
sideboards. The ordinary CPU learner/export/reload proof passes at 128.
Default/frozen layouts and rules remain unchanged. Tests use the production
allocator; discovery and callback assertions independently pin the public v5 name.

**Delete — do not maintain:** completed. `experiments/tla_pilot_v3.py` and the
intermediate `experiments/tla_pilot_v4.py` are replaced by
`experiments/tla_pilot_v5.py`; analysis, discovery, tests and current docs use
that runner. Historical runners/analysis remain at their frozen source commits.
Preserve every frozen regime/protocol, failure fixture and evidence artifact.
No remaining deletion target or parallel shared-effect implementation is needed.

## Budget and prepared execution

Prepared Run `e8c76bc2c06f` binds source
`97a2cfb963d7e3a3b2816f31c45dda72f34cdddd`. Later guard, allocation and test
simplifications do not retarget it or replace its own CUDA admission. Its
06:38 UTC A6000 quote was refused; no v5 submission or rental occurred.
The 06:46:27 UTC A6000/L4/A4000/A4500 reads, and the later retained operator
receipt at 06:59:30–33 UTC, all report unavailable capacity. These are dated
observations, not current availability. No unchanged failed-identity retry.

The 06:46:13 UTC inventory confirms all four pilot rentals ended and no provider
pods; the only unsettled session is unrelated non-rented mini. Prior three bills
total $0.17054831; v4's conservative planned-to-ended full-rental estimate is
$0.04314150 pending billing: cumulative billed plus estimated $0.21368981.
Retain all four reservations ($2.95) against $10 until reconciled. The next
allocation must fit the unchanged $1 ceiling. Receipts/scripts:
`.runs/mtg213/v4-continuation/`, `.runs/mtg213/v5-continuation/`, especially
`capacity-check-20261104-0646.json` in the latter.

Nested Doppler manabot/prd then etude/prd supplies artifact/provider credentials.
Etude 0.2.13 is the isolated control client; frozen workers retain 0.2.12.
Freeze source once through `source.freeze`, retain the typed Run and quote/submit
through `client.launch_cohort`. Existing launcher owners control allocation,
deadline and teardown. No automatic retry or unowned inventory cleanup.

## Remaining work

1. Gate and ship the existing repair in a new PR. Gate owns Rust fmt, all-feature
   Clippy and debug `cargo test`, model and active-training suites, current-world
   arena admission/monitoring and choice separability, discovery and experiment-
   data tracking per `docs/development.md`. Focused checks do not discharge these.
2. After delivery, obtain fresh capacity, inventory, billing and bounded quote
   checks, then resume the prepared v5 Run through ordinary CUDA admission within
   $10. Capacity is the external prerequisite; no human choice is missing.
3. Send the requested Task launch milestone only after observed GPU learner
   updates, including job/GPU/progress/rate/deadline evidence. Provisioning, CPU
   validation and initialization do not qualify.
4. Admit trained checkpoints, finish the whole v5 paired schedule and report
   learning gains, uncertainty, errors, total cost and Elemental-relative limits.
   New games cannot fill old identities' missing slots. Shipping is incomplete.

Check: `git diff --check` — passed; retained `uv run --extra dev --extra artifacts --extra evaluation pytest tests/eval/test_tla_pilot.py tests/verify/test_shared_effect_transport.py tests/eval/test_experiment_files.py -q` — 52 passed after release rebuild/atomic extension replacement; no rerun for documentation-only compression, full suites/debug remain with gate.

Prior build, learner/export/reload and 110-file publication verification:
`cce484db70049e81ee81983500cec4664b4962b1:scratch/tla-pilot-verify.md`.
Uncommitted notes before this consolidation are preserved at
`.runs/mtg213/context-before-compress-20261104-card-capacity/`; the current plan
retains their live decisions, contradictory evidence and outstanding work.

## Concept review — 2026-11-04 07:35 UTC

**Outcome: nothing to change and no open product choice.** This review adds no
implementation direction. The existing repair follows Jack Heart's accepted
fixed-population training/evaluation experience and shared-effect decision.
Execution and gate work above remain incomplete; that is not a design ambiguity.

The operator's interaction is: select the frozen pilot, prepare its exact source,
request one bounded GPU allocation, observe learner updates, then evaluate its
checkpoints. The few necessary records are the corpus (population), regime and
protocol (learning/input/evaluation contract), Etude Run (source and execution),
and retained checkpoint/match evidence (what happened). Pilot revision v5 and
policy-input version v7 name different contracts; neither is a progress counter.
There is one current runner, with historical implementations recoverable from
their pinned source. Existing README usage already states the selected runner,
pinned evaluation source and required output directory; no usage or skill
instruction rewrite is needed.

The consequential mismatch was an allocation that claimed a configurable card
axis while native encoding still used 60. The implemented repair makes
ObservationConfig → ObservationSpace → native scalar/vector capacity one chain
of authority; it does not create another population filter or rules model.
`frozen()` checks the hashed regime, native world and all tensor shapes;
`runs()` delegates execution to TrainingJob, while `eval()` and
`comparison_schedule()` retain expected comparisons and reject old pilot
initializations. `analyze()` keeps missing and failed evidence visible. The
existing Etude launch owner continues to own rental/deadline/teardown state;
there is no reason to introduce a second recovery controller here.

Normal path: independent draws from the same 134 entries feed ordinary self-play;
actual optimizer updates support the requested launch milestone, and admitted
checkpoints feed the paired schedule. Failure: the retained 61-object state
(40 original cards + five generated tokens + 16 sideboard cards) disproves 60
as a sufficient allocation. Recovery: retain that failed v4 identity and all
costs, use prospective v5/128 with fresh initialization, then obtain new evidence
within the same $10 cap. Prepared Run e8c76bc2c06f remains bound to its frozen
source, not later HEAD; its own CUDA admission remains necessary. A quote refusal
creates neither learner progress nor permission to replace an existing rental.

Counterexamples remain binding: 140/270-fragment failures invalidate the old
fact allocations; shared-effect provenance and duplicate memberships cannot be
deleted merely because current characteristics match; a future state exceeding
128 card rows must still fail explicitly. The larger card allocation proves
neither universal capacity nor learning quality. Missing old evaluation games
cannot be filled by v5 games, and a missing endpoint is not a negative strength
result. These are already represented in the implementation and retained plan.

Next proof is the existing gate plus bounded GPU execution and complete paired
evaluation, including cost and Elemental-relative learning limits. No new paid
run, publication, merge, tests or provider availability read was performed for
this review. Page: `scratch/concept-review.html`. Loop-or-next owns navigation.
