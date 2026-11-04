# TLA pilot continuation — 2026-11-04 UTC

**Incomplete, fix forward.** V4 passed CUDA admission but failed before its first
learner update at 61 card objects including sideboard against capacity 60.
The prospective v5 implementation uses the existing 128-card configuration,
with the native scalar/vector binding corrected to derive that axis from
ObservationSpace. All shared-effect tensors remain the delivered v7 contract.

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

## Delivered prerequisite and retained attempts

PR #419 merged at 05:55:57 UTC as
`fb5ad2c1372f9c967b550f0b8ae0f84883f75563`, with executed CI checks passing.
Task 5da5f4 owns that representation, now integrated here. V7 encodes complete
effects once, joining them to affected objects before attention: 128 × 18
headers, 512 × 3 details, 2048 × 2 memberships, plus 128 local facts. Fresh
library defaults remain v6. Lossless factoring preserves provenance and
multiplicity; redundant-activation merging was not proved. The inherited
keyword grant/removal ordering defect remains separate. Default padded v7
shared effects plus local facts occupy 37,888 bytes versus v6/256's 12,288;
no speed or learning advantage follows from factoring alone.

V1 exceeded 128 facts at 140; v3 exceeded 256 at 270. The delivered v7 passes
both exact retained prefixes in scalar/vector/Python encoding, preserving old
versions' failures. V3 source is `3b122de9078c985f846a0f3c5a3a42ebca3b1182`;
its 120 evidence artifacts remain in the manifest. V1/v2/v3 frozen inputs and
evidence are unchanged. PR #415's later 306-deck admission audit does not expand
the 134-entry population. Runtime remains w17 / card digest
`dcb9d47c674a657176c2a9d56e016288d1e175729aa09cd60f0dc333633330ed`.

V4 source `0b6c2bb854ba48b37fb7e5ac360c0c9d047cea34`, run `baa8671d6480`,
cohort `mtg213-pilot-v4`, used RTX A6000 after an L4 quote refusal. The GPU
substitution was recorded before scoring under the existing authorization.
Its CUDA gate completed, then collection failed after 104.74 seconds, with
238 completed-but-unoptimized games and 19 active prefixes. Stream 22's exact
300-action prefix/frame reproduces 45 card objects (40 originals + 5 tokens)
and 16 sideboard cards. V4 initialization evaluation finished all 32 games:
2 wins / 30 losses, no failed/truncated games. Analysis retains 288 expected,
256 missing, complete=false, success=null. Advertised etude-result.json is
absent. All 110 retained artifact entries are published under
`experiments/data/mtg213/gpu-v4-20261104/`. Its source/protocol/evidence stay frozen.

## Prospective v5 and deletion cut

V5 explicitly binds policy_input_version=7, max_facts=128 and
max_cards_per_player=128. The schema-3 protocol serializes every tensor shape
and rejects drift. This existing capacity is an operator engineering choice,
not a new representation or guarantee over all reachable games. The recipe,
100-update target, seeds, milestones and eight-pair/four-leg schedule remain.
Fresh initialization/evaluation is mandatory.

The first parity check exposed native input_buffer_config retaining 60 rows
regardless of allocation. Its shared scalar/vector owner now reads the card
axis and validates every consumed array against it. Default/frozen layouts and
rules are unchanged. Exact 61 rows succeed; 60 fail without dropping tokens or
sideboard cards. The full retained prefix agrees across all three encoders at
128; an ordinary CPU learner/export/reload check passes at that configuration.

Delete — do not maintain: the active v3 runner was replaced by v4, now replaced
by v5. Analysis, discovery, tests and docs move together. Historical runners and
analysis remain at their named source commits, rather than parallel maintained
paths. Keep every frozen regime/protocol file and evidence artifact. No competing
shared-effect representation is added here.

## Budget and execution ownership

Nested Doppler manabot/prd then etude/prd supplies artifacts and provider access.
Etude 0.2.13 is the isolated control client; frozen training dependencies retain
0.2.12. Freeze source once through source.freeze and retain the typed Run; quote
and submit through client.launch_cohort. Existing launcher owners control
allocation, deadline and teardown. No automatic retry.

Fresh inventory at 06:13:32 UTC confirms all four pilot rentals ended and no
provider pods. The only unsettled session is unrelated non-rented mini.
Prior three provider bills total $0.17054831; v4's full-rental estimate is
$0.04314118 pending billing. Cumulative billed-plus-estimated cost is $0.21368949.
Retain all four reservations ($2.95) conservatively against $10; a next allocation
must fit the unchanged $1 ceiling. Exact receipts and scripts are under
`.runs/mtg213/v4-continuation/` and `.runs/mtg213/v5-continuation/`.

## Remaining acceptance

1. Prepared v5 run `e8c76bc2c06f` binds source
   `97a2cfb963d7e3a3b2816f31c45dda72f34cdddd`. Its A6000 quote was refused;
   fresh 06:37:17 UTC L4/A4000/A4500 reads all report unavailable. No v5
   submission or rental occurred. Resume this prepared identity only after a
   fresh inventory/budget/quote, then execute through the ordinary CUDA gate.
2. Send the explicitly requested Task launch milestone only after observed GPU
   learner updates, with job/GPU/progress/rate/deadline evidence. Provisioning,
   CPU validation and initialization do not qualify.
3. Admit trained checkpoints and finish the entire v5 paired evaluation schedule;
   later games never fill old identities' slots. Report learning gains, uncertainty,
   errors, full cost and Elemental-relative limits. Shipping alone is incomplete.

Check: 30 pilot/discovery checks, 18 retained card/shared-transport checks and the
128-card learner/export/reload proof pass; release rebuild, Ruff, rustfmt and diff
checks pass. Full affected suites and debug validation remain with gate. All 110 published v4 files were hash/size verified. Context
is within both limits. Older detailed context: `0b6c2bb854ba:scratch/tla-pilot-verify.md`.
