# training wave memory

Evaluation memory (hosted LLM comparisons, inference receipts and evidence
reuse) lives in `.lf/wave/eval/MEMORY.md`.

## Fixed-corpus TLA pilot (MTG-213, 2026-11-03)

Jack Heart replaced the deferred full-TLA plan with a preliminary trophy-deck
self-play pilot and evaluation on the same frozen eligible population. Each seat
draws independently and uniformly, including mirrors. Unsupported main-deck or
sideboard cards exclude the whole list; no substitutions or replacement events.
Live fetching and a held-out-deck split are deferred, so independent evaluation
seeds do not establish held-out-deck generalization. The first snapshot contains
1,000 observed seven-win events from the permitted public game dump, 134 eligible
and 866 rejected. Source-order event selection is not uniform sampling of all
TLA drafts; missing source losses remain unknown. Corpus and protocol provenance
live in experiments/protocols/tla-trophy-213-v1.md.

Jack Heart subsequently authorized GPUs and shipping enabling changes after
checks, with a hard $10 total paid-compute cap covering setup, failures and
retries. This does not inherit earlier campaigns' budgets or checkpoints. The
first $1 L4 cap is an engineering allocation. Jack Heart corrected the credential
location on 2026-11-03: provider access uses Doppler `etude/prd`, while artifact
access uses `manabot/prd`; nested injection supplies both without copying secrets.
Checking only the application project did not establish missing credentials.
He authorized an available configured GPU alternative within the same cumulative
cap, with hardware/allocation deviations recorded before scoring. Provider
availability and launch quotes are not reservations: Etude 0.2.12 uses the same
template/offer path for both, but catalog responses changed between reads.
An admitted L4 quote subsequently failed launch admission after source preparation,
before acquiring a machine. The upstream cause of the changing replies remains
unknown. The retained 2026-11-03 19:35 UTC audit exhausted all six configured GPU
quotes and found no provider pods, unsettled GPU sessions or pilot cohort, with
$0 paid compute. This is dated capacity evidence, not permanent unavailability;
any later allocation still needs a fresh quote and cumulative-cost reconciliation.
A local one-update/199-decision proof
and 12 replay-valid evaluation games establish the connected implementation path,
not GPU readiness, learning improvement or Elemental-relative pace. The full
catalog/v5 policy and new population differ from the historical Elemental setup.

Paired evaluation has two distinct requirements: exchanging policy/deck/seat
assignments within each sampled pair, and comparing checkpoint gains across the
same complete pair schedule. Retained per-game failure rows and score curves do
not establish the latter. Missing matches/checkpoints remain in the expected
schedule, and incomplete comparisons cannot support a qualified learning claim.
The pilot now derives execution and analysis from one expected schedule and
resamples whole pairs together across checkpoints. Its seeded 2,000-resample
analysis implements the frozen joint criterion; controlled incomplete-cohort
and sealed-block CLI fixtures establish accounting, not empirical improvement.
A failed callback without per-game receipts counts as a failed match while its
unknown games remain missing: attempted games is a lower bound, not zero-cost
or successful evidence. Descriptive Etude curves do not qualify a learning claim;
the companion analysis does, only with a complete cohort and consistent sources.
Retained empirical limits live in experiments/tla-trophy-pilot.md. Training has
no child memory directories at this reconciliation (2026-11-03).

## Training consolidation and credit (MTG-179, 2026-10-22)

Jack Heart accepted one maintained RL execution path: TrainingRegime,
execute_regime, SeatRoutedCollector and Learner. Learner owns train-mode policy
admission and Adam; the executor owns collection, clocks, EMA, exports and
recovery. Bounded diagnostics reuse these primitives. Supervised/search/belief
stages retain distinct contracts. Legacy standalone trainer loops and shaped
training launchers are retired; historical reports remain readable and their
executions require pinned source. Their reward/schedule semantics are not
silently translated into the new local smoke recipe.

Jack Heart also required descriptive trainer concepts. DampedPolicyLearning
and damped_policy name the reusable method; Ataraxos names remain legitimate
experiment/builder provenance. Historical ataraxos_move tags decode at the input
boundary but retain their serialized identity, independently of schema-1's
historical one-seat default. Both compatibility contracts are necessary for
frozen recipe digests; neither authorizes rewriting old inputs or results.

Returns follow retained decisions of the same player, with terminal flags on
the reward's own row. Interleaved native A/B/A observations cannot form one
trajectory merely by changing reward signs. Terminal draws remain zero before
potential correction. Checkpoint loaders return eval mode; optimizer ownership
must explicitly restore train mode, including recurrent CUDA forwards. The
audit's CPU warm-start proof does not certify a CUDA execution.

One-seat exact quotas and both-seat complete-game floors have different row
counts. Exact recovery compares uninterrupted/resumed state under each declared
mode, not a shared expected count. Opt-in game-error recovery also finishes
one-seat games, as described below. Minibatch width remains the native stream
count after complete-game packing into one column. Removing tracking setup must
not remove its RNG reset: the PPO diagnostic seeds after baseline evaluation,
which consumes policy randomness. Adam epsilon, initial rate, empty-filter
iteration/EMA advancement and separate learner/native/optimizer clocks survived
the consolidation unchanged.

Independent asymmetric credit/gradient counterexamples found legacy GAE,
train-mode admission, mixed-player admission and shaped-draw defects; the
tested seat-routed ownership boundaries had no such defect. Full CPU
learnability at source 95ca2059e used w11, including 65,536 PPO transitions and
1,541 games. After sync to 8625df97e, the full w13 CPU diagnostic passed in
66.8 seconds with the same PPO counts and scores .59375 before / 1.0 after /
.640625 no-update (`.runs/learnability/training-consolidation-w13/report.md`).
The retained integration log records 118 passes; the caller also reported 31
current-world admission/monitoring passes. These are separate current-world
receipts; the w11 evidence remains bound to its original source/world.
The frozen w6 study at
fabb96db2bb88f98891aba66cc9d12e65d700545 bypasses the defective paths and explicitly
enables training mode; this is source applicability, not an exact CUDA replay.
Neither those tests nor reported throughput gains establish equal learning
value or full-game strength. Generic time-limit bootstrap remains unsupported.

## Training failure recovery (MTG-176, 2026-10-22)

Jack Heart accepted automatic quarantine/regather for isolated training game
failures, stopping at a configurable settled-game error rate strictly above 10%.
Frozen protocols, cumulative costs, original deadlines, attempt limits and
evaluation failure denominators remain authoritative. No paid campaign restart
is authorized. Rolling window/warmup 100/100 are engineering defaults, not numbers
Jack selected.

MTG-165's both-seat collector finishes games before updating. MTG-176 extends
that boundary to opted-in one-seat recovery so a later-crashed game has no
previously optimized segment. Declare the changed one-seat batching in comparisons.
Durable attempt ledgers own error allowance independently of optimizer snapshots;
restoring learning state must not erase failures after its committed boundary.
Game-level regather does not establish portable CUDA process recovery.

Automatic capture is separate from opt-in tolerance. Strict scalar/vector and
search paths retain inputs on caught exceptions; semantic Commands and arena
pending decisions are journaled before execution. Positional/pool aborts and
hangs still need external supervision. A missing worker outcome remains unknown;
replay advancement after repair is not a completed evaluation. Private seeds,
frames and exception text can disclose hidden state; public summaries use counts
and controlled status/reason fields. Source/native hashes do not archive bytes.

Shared replay enforces world identity; automatic records also bind source/native
bytes unless changed-runtime admission is explicit. Historical arena/tolerance
replay checks native bytes, not source digests. Committed-state restart has its own
stricter world/content/input/runtime checks; source changes require a repair reason.
Replay admission must not be mistaken for training or scientific admission.

Replay's binding, boundary and error agreement are independent diagnostics.
The 2026-10-22 concept review exposed that reaching the same action index alone
does not reproduce a failure: a changed frame is binding divergence, and a
different error there is a new failure. Missing binding/error metadata remains
unavailable. Rust/PyO3 error aliases and vector diagnostic suffixes are normalized
only for comparison; retained bytes and observed error text remain intact.
Strict vector capture establishes a batch failure, not individual stream outcomes.
Neither replay rejection nor repaired advancement resolves an unknown original
outcome, replenishes failure allowance or supplies evaluation credit.

The native Mountain/Gray Ogre fixture with a faulty/repaired payment validator
demonstrates capture, bounded replay and committed-state continuation. It does
not prove MTG-174's native payment repair or reproduce MTG-175's original failure:
that 34-action prefix is unavailable. The bounded CPU fixture connects repaired
continuation to persisted job allocations and existing cohort/job owners using
fake providers. Owner restarts retain unsettled reservations and reconcile a lost
create response without another rental; terminal settlement counts charges once.
Continuation retains cumulative time, the original deadline and one recovery
child per stopped attempt. CheckpointQueue retains checkpoint/protocol identity
across recovery ancestors, charges interrupted evaluations and exhausts the same
allowance without retries or fabricated scores. These are existing owner limits,
not a new numeric process-retry policy.

The proof substitutes CPU placement and evaluator processes; production RunPod
still rejects CUDA process recovery. It does not establish provider service
behavior or an Etude SDK recovery path. Broader validation and acceptance remain
with gate. Neither regather nor these fixtures demonstrate unbiased full-game
strength.
