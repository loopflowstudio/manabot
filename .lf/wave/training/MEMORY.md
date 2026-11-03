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
Preparing the exact source once avoided repeated preparation delay but did not
reserve capacity. Earlier zero-spend refusals were superseded by A6000 and L4
rentals. The retained 20:48 UTC reconciliation records both torn down, no pods or
unsettled GPU sessions, $0.06508817 estimated spend and $1.475 reserved pending
provider billing. A fresh quote and cumulative-cost reconciliation remain
necessary for any later allocation; provider availability is no longer the
observed blocker.

Source preparation also fixes a runtime, not just a job recipe. After integrating
PR #414, the retained corpus and local runtime both report w17 but different
card digests; all 134 eligible decks still admit. Version-name and whole-deck
admission checks therefore do not establish unchanged learning inputs or
checkpoint/evaluation compatibility. The prepared b8835f63b source was verified
against all 1,327 bundled files and its rebuilt native world matched the corpus.
Prospective correction 2adbd0758 changed only a host-dependent timing test;
native, learner, corpus, recipe and evaluator bytes stayed identical. Training
and initialization evaluation used that pin. Shipping newer implementation
does not refresh frozen evidence. Exact clock-accounting fixtures use injected
durations, not inequalities between hardware-dependent learning and I/O time.

The L4 run passed 32 CUDA admission tests and its optimizer probe, then failed
its first collection before any learner update. Seven legal Fire Nation Drill
activations over ten opposing permanents generated 140 keyword-removal fact
rows, exceeding frozen v5's 128. Native and Python agree on that failure; replay matches
the retained frame and error at stream 181, seed 31482, action 87. Cross-platform
replay shares source/world, not Linux/macOS binary bytes. Whole-deck admission
and CUDA admission therefore do not establish reachable-state input capacity.
Encoding errors remain fatal, outside gameplay quarantine. The implementation
audit found all 140 `(object, modification, detail, duration, timestamp, source)`
tuples distinct: seven timestamps per object, two keyword details per timestamp,
and no exact duplicates. Atomic capture retains the same records; text renders
all 70 effects. Under v5's one-object/one-detail row contract, deduplicating
effective keywords would erase observable provenance and alter the model's sum
of fact embeddings. This rules out compaction under those unchanged semantics,
not compression under every possible new contract. Dropping facts, excluding the
deck or retrying for a lucky sample would change the accepted experiment.

The concrete prospective proposal is a new input version/world with 512 fact
rows, preserving frozen v5 at 128 and explicitly binding allocation, encoders,
model shapes and checkpoint receipts. It remains unaccepted and unimplemented;
512 covers this counterexample, not all reachable states. Capacity also feeds
the model's timestamp normalization, so a larger buffer is not merely an
allocation change. Any accepted replacement requires new initialization and
evaluation with explicit runtime/admission binding to the same corpus entries
and weights; retained v5 evidence cannot be relabeled. The local audit and exact
stream identity are recorded in scratch/mtg-213.md.
Thirteen games completed before the batch failed but supplied no learner update;
settled learner counters do not measure all attempted simulation work.

Agreement on one retained failure is not general encoder parity. The subsequent
reconciliation found Python mapping `WhileYouControlSource` to indefinite (2)
while native used 3. The bounded Python repair preserves native's contract and
the parity fixture covers all three durations. It does not repair capacity or
refresh the pinned pilot's runtime evidence. A capacity proposal must preserve
duration, timestamp and source distinctions, not merely fit this one frame.

Initialization evaluation completed all eight pairs/four legs: 3 wins, 29 losses,
32 replay passes. The full schedule still expects 288 games, with 256 missing,
no trained endpoint and no qualified learning result. A separate CPU proof has
one update/199 decisions and 12 valid games. Neither establishes GPU learning or
Elemental-relative pace; the full catalog/v5 policy and population differ.

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
Etude runpy discovery requires canonically imported callbacks. Its table
projection can omit all-null columns; readers supply defaults only for optional
fields, preserving required-column failures. Local evidence-writing evaluation
requires an explicit ETUDE_OUTPUT_DIR. These execution/reader repairs preserve
the frozen schedule and criterion, not missing experimental outcomes.
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
