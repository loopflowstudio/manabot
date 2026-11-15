# MTG-350 — bounded mana payment work

## Concept review — next implement direction (2026-11-15)

**Design changed; iterate to implement.** A payment offer must preserve a
completion of the announced payment and every saved caller. The frozen failure
replay and the corrected continuation are different evidence. Requiring the
candidate to consume the same 197 positional inputs after correcting an earlier
infeasible offer contradicts that existing contract. This section supersedes
the exact-replay acceptance below; those earlier requirements and the observed
counterexample remain historical context. No production change is implemented.

This is an engineering consequence of the existing payment contract, not a new
chooser product choice. Jack Heart's recorded direction remains: "launch an
auto task stacked on this to fix the yue bug?" The Task's separate constraint
remains "Keep legal actions, first witnesses and the Engine Budget digests
unchanged". Removing a genuinely legal choice is not permitted. The diagnostic
offer difference alone does not prove that the removed activation is illegal:
the next implementation must establish that from complete resource accounting.
If a legal completion exists, repair the accounting/execution instead of
filtering that choice. No new product question is open in this review.

### Experience and model

The player announces Yue's activation, chooses two waterbend taps, and can
choose a nested mana ability only if its costs and output leave the caller
payable. After choosing a mana color, the engine resumes the actual unpaid
caller with earlier taps and costs intact. A sole legal action is still skipped
natively. No browser recovery button, cancellation, automatic source preference
or invented fallback action is introduced.

The concepts already exist: `Payment` is the announced work;
`PendingChoice::Waterbend` holds its remaining cost and committed tap count;
`Decision`/`SpendCompletion` retain nested callers. `SourceReservation` and
ordered `Obligation` values are derived planning inputs, not another persisted
payment lifecycle. Make resource derivation cover an announced payment
regardless of whether its current phase is mana, sacrifice or waterbend.
Prefer a small shared internal reader where these phases duplicate the same
fact; do not replace the whole payer or serialize a parallel reservation state.

Trace that invariant through `payment_model` (source and life reservations),
`payment_obligations`/`saved_obligations` (the caller chain),
`activation_witness` and mana-color validation (fees before output), and
`pending_choice_action_space`/`tick` (publish or advance a legitimate decision).
The current base omits Waterbend from both the announced-source reservation
branch and the announced-life subtraction branch in `payment_model`; the
source diagnostic only tested the former. Review nested caller reservations
and life accounting too, charging committed costs exactly once. A zero-action
pending payment is a violated invariant, never a forced action to skip. An
assertion can expose it in a regression; it cannot supply the missing repair.

Academy, policy, scalar/vector runners and search continue to consume the same
native authority. Preserve chooser ownership, Command binding and stale
rejection. Existing exact forks retain actual pending state and committed costs;
reconnect must show that state rather than reconstruct a preferred payment.
No observation layout, public API, checkpoint, world identity or receipt schema
change is proposed. Keep `managym/README.md`'s existing explanation; no shipping
docs or skill instructions need a new user operation.

### Evidence, failure and recovery

1. **Frozen failure:** keep the receipt/context bytes and runtime identities.
   The prior unchanged-source rebuild matched the complete frame at input 196
   and stalled; it is labelled changed-runtime reproduction. Original-source
   reproduction and the candidate are never reported as one replay result.
2. **Corrected path:** compare original-source and candidate frames before
   each input, stop at the first changed offer (193 in the diagnostic), and
   retain both frames and the common prefix. Do not feed remaining positional
   inputs after divergence. Identify each retained and removed choice by actor,
   native action kind, source/object identity, ability and parameters; labels
   and offer indexes alone are insufficient. Show why any removed choice has
   no legal completion. Select an explicit legal continuation in the candidate's
   own current frame and binding, recording new inputs separately with candidate
   revision/runtime provenance. This is a corrected continuation of the same
   seeded scenario, not `matches_binding: true` for the original final frame.
3. **Acceptance:** a bounded debug Yue regression and payment-model regression
   must cover competing taps, source reservation, life fees and nested caller
   restoration. Include a nearby payable nested activation to catch overfiltering.
   Verify exact spent/remaining mana classes, taps, sacrifices, life and caller
   progress, plus the next legitimate decision. Fork an intermediate payment and
   compare both continuations. Prove completion with trivial skipping enabled;
   do not refund the two earlier taps or resume a different caller to escape.
   Keep the isolated-step release target under one second and subprocess bounds
   from the original plan; neither has been measured for a repaired candidate.
4. **Replay tooling:** leave runtime admission and binding checks intact. The
   original CLI may correctly report `binding_mismatch` on a repaired runtime;
   that is not repaired-replay acceptance. Use a bounded diagnostic comparison
   for the corrected continuation, not a relaxed production replay path.

The source-tap-only import diverged at input 70 on this base. Do not duplicate
another owner's automatic source preference to force a match. First implement
and verify the shared reservation repair on this Task's native base with a
directed fixture. Use the retained failing-source snapshot solely for bounded
diagnostic comparison, with any patch clearly separated from shipping code.
Full candidate integration on the experimental source-tap trajectory remains
an explicit dependency: it needs an available, compatible owned prerequisite
and a documented minimal diff. If that cannot be obtained without unrelated
payment work, report the dependency; a snapshot-only repair is not Task
completion. No merge, new Task or execution is started by this review.

All four audit cells remain required. Their optimization acceptance still
preserves legal choices, concrete first witnesses and Engine Budget digests;
measure the same-machine 2x targets after the correctness repair. A changed
budget digest or a missed target remains unresolved, not an automatic baseline
update. Next show the bounded regression, the first-divergence proof and
corrected continuation, then the four before/after measurements. The review
page is `scratch/concept-review.html`.

## Implementation checkpoint — design counterexample (2026-11-15)

Dependent implementation stopped; no production Rust change is retained. The
retained case is an empty-decision loop, not an established search explosion.
The requirement to consume the same 197 positional actions with an unchanged
pre-action frame conflicts with correcting the payment authority. Review the
acceptance revision below before implementing the repair; the four audit cells
remain in scope and unmeasured.

### What this run established

- A CPython 3.13 release build of the unchanged failing engine at
  `fd7a7ed36c21553d419504daafeb36f70b3ab4ed`, extracted into an ignored source
  snapshot in this checkout, replayed the original inputs through action 195.
  At action 196 its complete frame matched the original bytes, including
  fingerprint `9b85ba019f7270c05cea9aa92779b40915bfc3a4a6febdc113aa305933880b91`.
  Time after reset to that boundary: 0.045749 s. Both required experiment
  flags were set. This is **changed-runtime reproduction**, not the original
  private binary: rebuilt extension SHA256
  `9458a2ae4652d8098d9439740aafc88713a80dec29f161bc9b2356f955ad14ec`.
- The action did not return within the diagnostic's approximately 58-second
  bound. The worker was killed and reaped. A one-second native sample at
  approximately 12 seconds found `Game::tick`, `tick_priority`,
  `mana_ability_actions` and `payment_model`, rather than one stuck witness.
- A second diagnostic build asserted immediately when `tick` received an
  empty action space. Exact emitted state:
  `pending=Some(Waterbend { player: PlayerId(0), payment: Activation { source: Permanent(PermanentId(21)), ability_index: 0, targets: [], x: 0 }, remaining_generic: 3, tapped: 2 }) suspended=None pools=[Mana { mana: [0, 0, 0, 0, 1, 0], classes: None }, Mana { mana: [0, 0, 0, 0, 0, 0], classes: None }]`.
  With trivial skipping enabled, `tick` repeatedly receives that empty space
  and executes no action. A timeout or search cache cannot repair it.
- `payment_model` reserves the announced activation's resources for
  `ManaPayment` and `SacrificePayment`, but omits `Waterbend`. A diagnostic
  change reserving `Waterbend` through the existing `payment_reservation`
  removes the nested mana activation from the logged frame at action 193:
  original `[activate_ability, waterbend_tap, waterbend_tap]`, diagnostic
  `[waterbend_tap, waterbend_tap]`. The logged frames through 192 agree.
  Continuing the original numeric actions after this offer change changes
  their meaning and is **not** a repaired replay. The diagnostic is evidence
  for the resource-reservation defect, not a verified production fix.
- Importing only experimental source taps into the Task base did not recover
  the original trajectory: after 70 committed actions, input 2 was rejected
  against two priority actions. The imported prerequisite inventory was
  `Action::TapForMana`, action/offer/focus projection, setup switch,
  `fixed_tap_actions`, `tap_for_mana_action`, payment entry/dispatch, and
  `Mana::spendable_units` with its directed test, all from the failing source.
  All were removed after this counterexample; no automatic source preference,
  combat, capture or study changes were imported into production. The unchanged
  source snapshot was used only to diagnose, never as the candidate.

Original receipt SHA256:
`2a575ed1eba230fda450c2be7534182f6ba663922a35ecbeda9a963bf8d9c007`;
acting-viewer context SHA256:
`120d22c30608762ac8f01e11d44fc906dc0ba5a58179c011cb3b1075458bc387`.
Both remain unchanged. The diagnostic adds this branch in
`payment_model`, after the `ManaPayment` reservation and before
`SacrificePayment`:

```rust
if let Some(PendingChoice::Waterbend { payment, .. }) = &self.pending_choice {
    if let Some(reservation) = self.payment_reservation(payment) {
        reserved.push(reservation);
    }
}
```

Local ignored logs are supplementary; the observations, diagnostic change and
reproduction recipe here carry the handoff.

### Proposed acceptance revision and remaining work

The next design must allow correcting infeasible offers before action 196 and
must specify how the source-preference prerequisite is owned; do not silently
import the parent's separate payment work. Preserve the original replay as
failing evidence. A corrected replay should retain the common prefix, compare
frames at the first changed offer, explicitly select a legal continuation by
identity, and check complete costs/resources and the resulting decision. It
must not relabel shifted numeric inputs as the original game advancing.

Build a bounded debug regression of a tap-source waterbend activation with a
nested mana producer that competes for the same taps. Ensure announced source
reservation and life costs survive the whole caller chain, all genuinely legal
choices remain, and no empty pending decision is produced. An empty-space
assertion alone is not the repair. Put rules coverage involving Yue in
`managym/cards/tests/cards/tla.rs`, with payment-model coverage on its one
surviving authority. Then rebuild and demonstrate the corrected continuation.
Only after resolving this acceptance contradiction continue the audit
optimizations and their four same-machine before/after measurements.

Diagnostic reproduction: `git archive fd7a7ed36c21553d419504daafeb36f70b3ab4ed managym content/semantic`
into an ignored source directory; build its CPython 3.13 wheel with
`uv run --no-sync maturin build --release -i .venv/bin/python -m <snapshot>/managym/Cargo.toml -o <wheels>`;
install the extension atomically. With both experiment flags set, reset one
`VectorEnv` from the retained receipt and feed `games[0].actions`, comparing
`semantic_decision_frames_json()[0]` to `games[0].frame` before action 196.
Run in a subprocess with a deadline and always kill/reap it on timeout. The
unchanged build reproduces the stall; the diagnostic patch changes the earlier
offers and therefore fails the original exact-frame acceptance.

Check/result: unchanged failing-source release build + bounded native replay —
matched retained frame, stalled; empty-space diagnostic — reproduced; source
reservation diagnostic — changed frame 193, original acceptance contradicted.
Production edits were removed and the Task-base release extension restored;
no speedup, corrected-replay, debug-regression or audit-target pass is claimed.

## Outcome and scope

Advance the retained source-tap game past stream 24, seed 37037, action 196
(Add {G}, actor 0, turn 25 precombat main), preserving its legal choices,
resource commitments and next legitimate decision. Prioritize that stall, then
speed up the four payment-heavy states identified by MTG-337. This is engine
work; Academy, scalar/vector runners, training and search inherit the same
native decisions. No new browser payment design, policy layout or card gate.

Jack Heart, 2026-11-15: "launch an auto task stacked on this to fix the yue bug?"
The source-bound handoff reports a matching pre-action frame and an individual
call that did not return within 60 seconds. Yue is the outer context, not a
root-cause finding. `scratch/yue-stall-receipt.jsonl` retains the native inputs
and runtime identities; `scratch/stream-24-context.json` retains the acting-viewer
context. Preserve those bytes. The original binary remains private with the
study evidence; it has not been copied into this checkout.

No paid training, GPU rental, active-job control or population changes. A deck
exclusion is a workaround, never the repair. Stay stacked on MTG-347; do not
copy its dirty work or merge PRs #523/#524/#538/#540 to obtain a convenient base.

## Findings and current architecture

Checked on 2026-11-15 at checkout `ddc337f764`:

- The tree is clean, with no `.venv` or native extension. Both source commits
  are available: failing engine `fd7a7ed36c21553d419504daafeb36f70b3ab4ed`
  and study `3b519a3e94edd5570d240e606d636fb9d358fd96`. No native replay or
  performance measurement has been run in this checkout during kickoff.
- This base has no source-tap switch or `Action::TapForMana`. In the failing
  source, `flow/mana.rs::fixed_tap_actions` checks each fixed source against
  the pending cost and saved obligations; `tap_for_mana_action` consumes one.
  `flow/action.rs::pending_choice_action_space` publishes those actions and
  `flow/play.rs` retains a payment when they exist. `flow/setup.rs` captures
  `MANAGYM_EXPERIMENT_SOURCE_TAPS` once in `ManaPaymentStats`.
- This is not an isolated switch diff. The failing source also contains
  `mana/tap_choice.rs`, automatic fixed-source preference work, combat-pick,
  capture changes and other integrations. Import only dependencies shown
  necessary by compilation and prefix replay. Keep a hunk-level prerequisite
  inventory in this plan. In particular, source-taps alone might not reproduce
  the original trajectory if automatic source preference changes it.
- Both base and failing source already use `PaymentModel::witness`,
  `PaymentSearch` source classes, color bounds, exact resource states and the
  direct fixed-source path. Do not assume missing memoization explains this
  stall. `activation_witness` adds the activation's obligations before saved
  callers. `payment_obligations` and `saved_obligations` own that caller chain.
- `flow/decision.rs::ManaChoice` filters and validates colors through
  `mana_choice_completes_payment`; that code is unchanged between the base
  and failing commit. In the failing source this method includes pending
  production bonuses and asks `PaymentModel::payable`. A slow Add {G} step
  could be validation, automatic production, resumed caller work or forced
  decision advancement. A stack sample distinguishes these; the receipt alone
  does not. Follow `advance_mana_upfront`, `finish_mana_payment` and native
  trivial skipping before editing the search.
- `manabot/env/replay.py::_replay_game_inputs` compares the frame immediately
  before the last recorded action and reports `matches_binding`; divergence
  returns `binding_mismatch`, not a native failure. `require_replay_runtime`
  refuses differing native/source hashes without `--allow-changed-runtime`
  and refuses another world even with it. The CLI's `--seconds` wraps the
  worker in a subprocess timeout. A rebuilt original source is still a
  changed runtime unless its hashes actually match.
- MTG-337 already supplied `managym/examples/bench_states.rs` and payment
  counters. Its recipe comments distinguish expensive nested-filter searches
  from repeated small plans in rollouts. Reuse it, not another benchmark.
  `docs/development.md#stress-states` documents sizes and layers.
- `flow/game.rs::Revisions` tracks characteristics, pools, tapping and
  triggers; it does not independently identify a pending payment, life or
  saved decision chain. A cache keyed only by those counters is insufficient.

The Task comment attributed to the session with requester Jack Heart
(2026-11-13) records search-step samples of approximately 1,020 ms for
counted-tap size 16, 355 ms diverse-mana size 16, 260 ms mixed size 16 and
103 ms restricted-filter size 4. These are historical samples, not this
checkout's baseline. Its explicit constraint is "Keep legal actions, first
witnesses and the Engine Budget digests unchanged". The existing
`payment_bound_allows_restricted_mana_to_fund_a_filter` test captures why
subtracting activation fees from gross output is unsafe.

## Chosen approach and alternatives

Keep one payment authority: repair the proven hot path inside the existing
native machinery. Diagnosis found the retained stall is an empty pending
Waterbend decision from incomplete resource reservation, not search blow-up,
so its repair is the accounting fix in the concept review; no bound or cache
addresses it.

For the remaining audit scope, prefer these mechanisms, subject to the measured
work counters:

1. Nested-filter pruning: add a conservative feasibility bound beside existing
   bounds, retaining spending-purpose conversions and saved output obligations.
   It may overestimate supply, never reject a payable branch. Compare the full
   concrete first witness to an unpruned reference on bounded fixtures. A bound
   that needs the original exhaustive search to compute is not an improvement.
2. Repeated plans: use a bounded, per-Game exact-query cache at the payment
   planning boundary. The key includes the complete ordered PaymentModel
   (source identities, outputs, abilities, restrictions, current resources and
   life), pool, ordered obligations and SourcePolicy. Store negative answers
   and full concrete witnesses; boolean answers cannot stand in for plans.
   Use value equality, not a digest alone. Start with 64 FIFO entries, retain
   them on a Game clone by value, and allow safe eviction. Cache lookup costs
   and retained bytes must appear beside hit rates in the measurement. Do not
   add a Revisions-only shortcut until all missing dependencies are covered.
   Exact query equality makes changes to pending callers miss even when the
   board revisions do not move. This cache addresses repeated work only; it
   cannot bound a first-time pathological query.

Rejected routes: fixed-depth/node cutoffs that change legality; taking the
first color; automatic-spend or disabling trivial skipping as the fix; merging
sources by equal mana output; banning Yue; retrying the same call with a larger
timeout; global caches shared across games. Source equivalence remains an
internal feasibility optimization and must not erase source identity in offers
or witnesses.

Wild success: the same replay returns promptly and nested payments preserve
all meaningful source, color and pool choices. Wild failure: a fast cached
answer silently reuses a witness after a tap, life change or caller transition.
Prevent that with exact keys, mutation/fork tests and debug recomputation on
cache hits. If key construction costs outweigh reuse, remove the cache rather
than retain a second ineffective planning system.

## Remaining work and checks

Realigned 2026-11-15 against base `ddc337f764` (parent PR #523 head unchanged)
and `origin/main` at `2176882620`. Nothing below is implemented or measured.

New facts from this pass: the base already offers nested mana activations
during Waterbend (`flow/action.rs::pending_choice_action_space` calls
`mana_ability_actions` in the `PendingChoice::Waterbend` arm), so the
reservation defect is reachable without the experimental source-tap path. Main
has the same omission (`origin/main:managym/src/flow/mana.rs::payment_model`
names only `ManaPayment`/`SacrificePayment` in both branches). The directed
fixture therefore needs no source-tap prerequisite; only the corrected
continuation of the retained trajectory does. PRs #523, #524, #538 and #540
remain open and unmerged.

1. **Repair on the Task base.** Cover Waterbend in `payment_model`'s announced
   source reservation and life subtraction (one shared reader for the announced
   payment), and audit nested caller reservations so each committed cost is
   charged once. Prove from complete resource accounting that any offer the
   repair removes has no legal completion; otherwise fix accounting/execution.
2. **Directed debug regressions.** Yue rules case in
   `managym/cards/tests/cards/tla.rs`; payment-model cases in `flow/mana.rs`
   covering competing waterbend taps, source reservation, life fees, nested
   caller restoration and a nearby payable nested activation (overfiltering
   guard). Check spent/remaining mana classes, taps, sacrifices, life, caller
   progress and the next decision, with trivial skipping on, plus a fork of an
   intermediate payment. Deterministic, far below 300 seconds.
   `CARGO_BUILD_JOBS=2 cargo test --manifest-path managym/Cargo.toml -p managym --lib flow::mana -- --test-threads 2`
   and the cards target, in debug.
3. **Corrected continuation (diagnostic only).** Rebuild the cp313 extension
   from root (`uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`,
   atomic rename). With both experiment flags, compare original-source and
   candidate frames per input, stop at the first changed offer, identify kept
   and removed choices by actor/kind/source/ability/parameters, then select a
   legal continuation by identity and record those inputs with candidate
   provenance. Applying the repair to the retained source-tap trajectory needs
   an owned, compatible prerequisite with a documented minimal diff; if that
   cannot be had without unrelated payment work, report the dependency (a
   snapshot-only repair is not completion). The original CLI
   (`manabot.training.state.game_errors ... --allow-changed-runtime`) may
   correctly report `binding_mismatch`; keep its checks unchanged and run
   `uv run --extra dev pytest tests/training/test_game_errors.py -q`.
   Isolated step target under 1 s in release, not yet measured.
4. **Four audit cells.** Only after the repair: same-machine before/after with
   `cargo run --release --features benchmark --example bench_states --manifest-path managym/Cargo.toml -- counted-tap --sizes 16 --layers search`,
   likewise diverse-mana and mixed at 16 and restricted-filter at 4. Target 2x
   lower search-step median per cell; report misses. Keep legal actions, first
   witnesses and Engine Budget digests; a moved digest stays unresolved, not an
   automatic baseline update. Use the bound and exact-query cache above only
   where counters show the work.
5. **Gate.** Workspace `cargo fmt --manifest-path managym/Cargo.toml --all --check`,
   configured clippy, and debug nextest (`CARGO_BUILD_JOBS=2`,
   `--profile ci --test-threads 2`). No raised timeout.

Done when the repair passes the debug regressions, the corrected continuation
reaches its next legitimate decision with complete resources intact (or the
prerequisite dependency is reported), and the four cells are measured against
the target without changing legal actions, first witnesses, forced-decision
behavior or digests. The rebuilt binary is never relabelled as the original.

## Delete — do not maintain

No existing payment path is slated for deletion during diagnosis. Remove local
stack probes and temporary instrumentation after promoting the minimal useful
regression/counters. Retain the original receipt and its provenance. Do not
introduce a legacy/new payer, Academy-only workaround, alternate replay path,
fallback legality answer, serialized cache or world/checkpoint migration.

## Documentation review

2026-11-15 write-docs: no user-facing documentation change is needed for the
chosen performance repair. It introduces no public API, option, default or
payment interaction. Reuse `managym/README.md`'s payment explanation and
`docs/development.md#stress-states`' existing harness example, size/layer
options, work counters, digest checks and same-machine timing guidance.
Changed shipping doc paths: none. Implementation and measured speedups remain
pending; this review establishes neither. No open product question was exposed;
the parent report's chooser questions remain outside this Task.

Presentation attempted with `lf open --host desktop --file managym/README.md
--file docs/development.md`: "No Desktop Files view attached; nothing opened."
Both files remain at their existing paths.

## Kickoff checks

2026-11-15: committed-source inspection confirmed the base mismatch and replay
admission/binding contracts. Parent scratch history is at `ddc337f764`.
