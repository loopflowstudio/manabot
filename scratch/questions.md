# MTG-245 — implementation handoff

## Accepted direction

Jack Heart selected A on 2026-11-04 (“i agree A is the place to start”): authored
functional reference decks, frozen independent per-seat tickets and stable
anchors. B generation and C adaptive search remain alternatives. Selecting A did
not waive actual training-exposure capture. Exact lists, 50/25/25 shares, seeds
and numeric bounds are proposed settings for Task 04's authored freeze.

Jack Heart accepted the four-core-Wave/workflow work inside this Task (“nah i
think jsut leave that in this”), then resolved consolidation: “training is the
main one and the evals tasks get moved into it”. The original Training identity
54bc9162-c883-4574-bab1-af589526c1ad survives as Experiments. Academy stays separate.
The latest reconciliation steer e6c3e03c-65a1-4b6a-8ec2-abed2aa9964f covers the
concurrently recreated Training Wave. No scientific campaigns, rentals, spending
or shipping were authorized here. Existing Task Graph/review boundaries remain.

## Implemented in this continuation

Shared native CardCapture optionally records complete nontrivial positional
choices using the observation owner's action focus, plus completed effect frames
at native finalization. Card capture types moved from sim/card_evidence into
env/card_capture, preserving historical serialization and one owner. No policy
state, input widths, native legality, RNG or rules behavior changed. Search forks
still drop instrumentation. Forced decisions remain skipped in actual runners.
Final inspection found a shared capture counterexample: a face change could make
its recorded name/definition violate the reader's physical-copy invariant. Native
snapshots now retain identity from setup/creation while zone/incarnation remain
current. A controlled face-switch fixture covers this repair; rules and policy
observations are unchanged.

TrainingJob explicitly enables capture for the reference league; omitted/default
jobs retain their configuration. CapturedVector saves terminal captures before
native auto-reset changes their seed/setup, and records interrupted current
prefixes on normal close. On a native failure the existing private failure owner
retains the original error; invalid partial capture cannot replace it. Unobserved
failed prefixes remain missing. The private digest-named JSONL receipts retain
sampled decks, seed/actions, exact copy/incarnation, runtime/input identity and
run/update context. TrainingRun registers these as exposure artifacts, never
checkpoints. Replay inputs use the existing failure-replay owner; these receipts
are **not replay-verified**.

The offline report accepts downloaded exposure directories with repeatable
`--training-evidence`. It validates digest, run/seed/world/source/decks and reports
terminal versus interrupted games, per-stream budgets, both-seat decision
denominators, opening/draw
copies, action-type opportunity/choice counts, per-definition focus counts and
completed frames. Observed zero is distinct from absent receipts. Missing capture
makes `complete` false independently of the primary `modest_check` criterion.
Resolution counts describe completed effect frames by the physical source copy's
initial definition; they do not assert mechanic-specific success, dynamic copied
program identity, strategic benefit, or contribution to an optimizer update.

Training → Experiments, Eval's two open Task moves, and Eval retirement used lf.
History/memory provenance is in docs/workflows/2026-11-04-migration.json. The
concurrent fc6ffb Task created a new generic Training Wave; inspection confirmed
its experimental outcome and separate $25 authority. It moved to Experiments
without altering its live Graph or comments, then its empty Wave was retired.
Final shared reads show the four core Waves plus Academy; abandoned Study remains
historical. PR #421's published head and uncommitted Training-memory bytes remain
unchanged in its checkout. Its eventual integration must carry those edits into
.lf/wave/experiments/MEMORY.md. Provider synchronization remains pending; no auth
repair or moved-backlog execution occurred. Shared graph defaults await delivery;
main's old `iterate` selection is not a replacement of existing captured graphs.

## Remaining ownership and limits

- Gate owns the affected Rust/Python suites and full automated acceptance checks;
  focused checks cover the directed debug fixtures and Python capture/report path. Demo/review remains the next product boundary.
- The frozen v1 JSON is unchanged. Instrumentation changes native source identity,
  so it correctly rejects a changed runtime. Task 04 must freeze the integrated
  prospective protocol, actual Adventure/prepare lists and shared probe IDs,
  numeric settings and execution authority. Neither synthetic fixtures nor the
  proxy lists establish learning, throughput, real exposure or playable coverage.
- Task 08 owns operational Validation B. The deeper strength/generalization track
  begins separately after full FDN; its execution Task is still unassigned.
- Tasks 01/03 remain owners of broader shared history/crash receipts. This change
  extends their existing native capture boundary, adds no mechanics taxonomy and
  launches no competing Task. MTG-243's observation/action version stays intact.
- MTG-210/211/230/231 retain memberships pending an outcome-based mapping; moving
  Eval and resolving the recreated Training identity does not select that backlog.
- PR #421's empirical reservations and fc6ffb/MTG-213's separate budgets confer no
  scientific execution authority on MTG-245.

## Assumptions and deletion cut

For the first reference-league handoff, all surfaced flat-policy seats are
captured. Opportunity means a card is in the native action's focus, counted once
per definition/chooser/decision, not a claim that using it is strategically good.
Resolution means the native effect-frame program completed, not that each effect
changed state. Retaining all choices trades artifact size for auditable bounded
pilot evidence; performance/cost remain for the authorized pilot to measure.
Downloaded job-output directories are explicit analysis inputs so the instrument
does not invent a second Etude checkpoint/artifact store or assume bucket layout.

Delete — do not maintain (completed; no remaining targets): the constant “training exposure unavailable; acceptance
unresolved” report field; the old sim-owned native capture model definitions;
the retired Eval graph config. Existing historical capture bytes and evaluation
readers retain their meanings. The plan's prior review is retained in
scratch/concept-review.html; its two acceptance questions were resolved by the
named 2026-11-04 steers. Earlier exploration detail is at
190b0c603d70e790f2217c61c970c38f82544d08:scratch/explore-design.md, and earlier
protocol/workflow implementation context at a72b5ef1a:scratch/questions.md.

Check: `uv run --extra dev --extra artifacts --extra evaluation pytest tests/training/{test_training_exposure,test_growing_pool_protocol,test_growing_pool_analysis}.py -q` — 53 passed; `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2 --lib exposure_` — 3 passed; rebuilt/atomically installed cp312 extension and 6 native-roundtrip, journal-durability and evaluation-reader checks passed; `uv run python scripts/check_loopflow_workflows.py`, Ruff, rustfmt and `git diff --check` passed. Gate owns affected suites/full acceptance.
No campaign or complete game was run; native proofs use controlled positions only.
