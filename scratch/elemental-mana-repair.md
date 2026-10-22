# Repair the offered Gnarled Professor cast, then resume the same experiment

Task MTG-174's actual $10 baseline and live Etude reports remain unachieved. The prior Flow660cf043-d8d5-407d-bc57-71f922d96ec8 has completed after retaining the blocking native legality failure. This is a local executable engine defect to fix within the authorized task, not a request for more budget, a new Task, or permission to substitute a smaller experiment. Jack explicitly authorizes necessary Manabot/Etude PRs. No paid work has started. Preserve the successful uniform-sampler implementation and its evidence.

## Authoritative failure and ownership

Read scratch/elemental-baseline-preflight.md's final admission record and the exact retained reproduction at experiments/data/mtg174/admission-native-legality-v1/ (fetch from the artifact store if missing). Its reproduce.py replays w10, seed2165384, 34-action prefix, with WB vs UG exact native setups. Frame35 offers action4, Gnarled Professor; executing it raises not enough producible mana. Original isolated-root SHA256 is2fd006c251f3064018303bd5c124e9bb00fb82b6c4261a790ebbbc033f16a4e1, mana source SHA2569bf74653518bfdaefb88ba69a1611d698c2b8a0d187c736835ded3993ce8a4f7. Keep those failed bytes/source receipts immutable. They came from the adaptive-curriculum Task's native-legality-v1 evidence and reproduced independently here.

The current relevant board has untapped Forest first, Island, Prophetic Prism, Island, Mountain, zero mana pool. Gnarled Professor costs2GG. The original frame includes the Professor cast and Prism activation. A valid payment exists: pay Prism's1 upfront using a non-green land, choose green, then finish the announced spell. A strong diagnosis to verify is that mana_ability_actions simulates pay_mana_ability_cost, whose generic produce_mana loop greedily taps the Forest first; completion filtering then removes the Prism choices, begin_mana_payment sees no actions and auto-commits the impossible fixed-only2GG cost. This is a hypothesis grounded in source/board, not a license to special-case these cards.

The broader optional-payment Task4ba5dd is separate and active; do not wait for or replay that refactor. Keep this repair narrow and preserve a clear source/commit handoff so later integration can retain it. Do not edit another checkout or start its worker.

## Required behavior

Offer generation and actual upfront mana payment must use a consistent feasible payment, preserving at least one legal continuation of the already announced parent cost. Fix the authoritative payment owner shared by native/compiled paths. Do not hide a genuinely payable cast, drop a card/deck/pair, choose a seed/opponent that avoids the counterexample, auto-select a player-visible output/color, add card-specific exceptions, or change the accepted learning recipe. Preserve printed activation cost-before-output order, restricted Lesson mana, tap-for-mana bonuses, reserved/sacrificed sources and nested decisions. Avoid merely sorting lands with a heuristic that still promises impossible completions. Reuse/factor the existing payment search or a bounded exact plan where necessary, keeping decision/Command semantics and one rules authority.

Start with a minimal failing regression of the real Forest/Island/Prism/Island/Mountain configuration, plus the retained exact root replay. Check alternate source orders and appropriate payable/unpayable cases. Verify the cast stays offered, its payment can execute to a legal completed cast with player-visible choices intact, and all offered completions remain valid. Include replay/fork isolation, existing Prism/restricted/granted mana tests, and debug cargo checks per AGENTS.md. Rebuild the extension through uv before Python/native tests. Preserve the previous failed artifact, and retain new exact source-bound passing receipts and a bounded full-roster collector follow-through. Run only affected regression suites and necessary broad native gates after the final repair, not unrelated scientific evaluations.

No observation/action shape change is requested. Follow WORLDS.md and actual compatibility contracts for any unavoidable identity change; never relabel old failed/frozen evidence as passing. Preserve the accepted exact ten decks and compiled pack. Freeze the actual repaired source/runtime before the scientific run.

## Remaining live-evaluation work

Read the Live reporting direction and Etude version verification in scratch/elemental-ten-dollar.md. Current Etude source has typed evaluation jobs and reports; the PyPI0.1.2 wheel does not. The previous isolated local typed evaluation job fbaf4f12b6ec failed before command execution because its uv overlay interpreter was not available in the launched worker; it is terminal, $0.00 rental, with immutable source/log/cleanup receipts. Unvalidated bridge source was removed in23275ed9; historical prototypes remain at3406a548 and44406a97. Do not pretend that attempt launched games or established a live report. Repair the runtime/process boundary and test an actual local Etude evaluation through game completion, with persistent report ownership, before the paid baseline. Do not globally migrate other live campaigns or shared launch state.

Credential admission passed for the existing43,140-second maximum and current L4 quote0.59/hour; rates, prior charges and live-job exclusion must be read fresh at actual allocation. The active training pod belongs to MTG-165; leave it alone. Historical storage allowance0.02/hour makes about7.31 for the max window before reserves/probe charges; this is a projection, not actual spend. Keep the inclusive$10 cap, guardrails, and one scientific run. No paid resources during this repair.

## Investigation and source-only prototype validation — 2026-10-22

This section retains the first repair iteration. Its validation is superseded
by the final pool-aware repair and source-bound receipts below.

The retained WB/UG root, seed 2165384 and its unchanged 34-action prefix,
reproduced `PyAgentError: not enough producible mana` before editing source.
The new minimal native regression also failed against the original
`HEAD:managym/src/flow/mana.rs` with that same error (debug cargo exit 101).
The failure is the payment-owner mismatch described above: admission's search
finds Prism filtering, but the upfront generic payment consumes Forest first,
so the completion probe rejects Prism and spell payment falls through to an
impossible fixed-only cost.

`pay_mana_ability_cost` now finds a concrete finite subset of automatic fixed
sources, fewest additional taps first, and accepts it only when paying the
printed activation cost leaves a legal parent continuation for some output.
Offers and actual activations use the same planner and fixed-source tap/event
executor. The source is reserved first; parent reserved/sacrificed sources
cannot fund that upfront cost. Existing typed pool spending and payment search
retain restrictions and tap bonuses. The planner can add a source even when
the current pool could pay, so a generic cost does not destroy the only green
already pooled. It selects no chosen-color output and changes no Command or
observation schema. This is a rules correction within development w10, not a
new world/shape or a scientific source/protocol freeze.

Native regressions cover Forest/Island/Prism/Island/Mountain and an alternate
order, green already pooled, no-green unpayability, two-Forest multiple-color
continuations, and exhaustive offered payment branches with exact mark/apply/
rollback and independent-root isolation. The compiled Python regression embeds
the retained prefix and checks repeated exact roots and independent forks.
The production wrapper fast-forwards the sole activation/color/payment and
forced priority steps, so its successful offered cast reaches the resolved
Professor's legal Learn decision. The initial Python test incorrectly expected
a stop on the stack; its assertion failure is retained in
`.runs/mtg174-mana-python-initial-assertion.log`, and the boundary assertion was
corrected. Directed native tests exercise the explicit payment decisions.

All 11 old files under `admission-native-legality-v1` were rehashed against
`experiments/data.manifest.json` and match exactly; the verification is retained
at `.runs/mtg174-mana-repair-v1/old-receipts-verified.json`.
The Etude interpreter/worker failure, historical prototypes and their old
receipts remain unchanged. No paid allocation, scientific arm, Task completion,
or parallel worker was started. The per-game uniform sampler and all exact ten
authored setups/compiled content remain unchanged.

The rebuilt cp312 extension executes the exact old root to the resolved
Professor and a legal Learn decision. Its original frame still offers the
Professor at index 4 of eight actions. Root digest:
`3e2da632a41a3a31319c05b7a533c7fdfe77a5d71f0e0468750f73a161cee1eb`;
after executing that offered action through real payment and resolution:
`92d38b45ab8021a3595e4f33a87dae3fc8a8e4710a87b9ee848d7a4ba44a9892`.
The fork leaves the original root unchanged. Do not run the old `reproduce.py`
in place against repaired source: it writes its historical `reproduction.json`.
Use the new compiled regression or the read-only-old-input proof at
`.runs/mtg174-mana-repair-v1/prove.py`.

Bounded collector follow-through used seed 17402, 256 streams × 64 learner
transitions, the existing per-game uniform sampler, fixed learner seats and
64-dimension/two-head/FiLM=false inference. It collected 16,384 learner and
34,233 environment decisions, 60 terminal games and 316 total game roots in
47.32 seconds on local CPU. It reached 93 distinct ordered pairs; this is
finite-sample exposure, not exact uniform realized balance. Native all-pairs
validation separately completes all 100 ordered pairs and mirrors. No optimizer
or scientific strength evaluation ran in this collector gate.

Final-source Python validation: **42 passed in 46.92s**, using:
`uv run --extra dev --extra notebook --extra artifacts --extra reports pytest tests/semantic/test_elemental_world.py tests/semantic/test_temur.py tests/training/test_elemental_uniform.py tests/training/test_matchups.py tests/agent/test_managym.py tests/infra/test_experiment_data_untracked.py -q`.
The test list includes full-roster setup/Lesson binding, scalar compiled Temur,
per-game uniform exact/current recovery, the real local CPU training/checkpoint
test and old matchup scheduling. Formatting, ruff, `git diff --check` and
`cargo clippy --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings`
passed. The required root-environment release wheel build passed with
`uv run maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`;
the wheel's cp312 `.so` was extracted into the required local extension path
before Python tests and the collector gate.

## Pool-aware final repair

Review identified another instance of the same authoritative upfront-payment
problem: Prism plus only `WUUUU` in the pool must offer and cast Serra Angel
(`3WW`) by paying a blue upfront and choosing white. A default generic payment
spends the only white instead. No additional land subset can repair that
case. The final repair therefore factors concrete typed payment remainders in
`state/mana.rs`, sharing colored/hybrid/generic legality and restriction handling
between ordinary payment and the existing feasibility search. Ordinary payment
retains its first-result/colorless-first policy. Upfront-ability planning and
`can_cover_cost` consider the concrete alternatives; execution commits the
selected remainder rather than reselecting generic colors. Restricted mana is
still spent first within a color and combat mana first within its class.
No player-visible output is selected by this search.

The added Serra Angel regression verifies offer generation, the exact remaining
`WUUU` pool before output, the white output decision, and completed payment.
A unit regression checks generic/hybrid alternative remainders and Lesson
restriction retention. The prior source-only receipt, patch and partial native
log are retained under `mana-repair-v1/before-pool-review/`; the earlier full
native run was intentionally stopped in its remaining search-state suite when
this review required the final code change. Final debug/build/Python/collector
validation is run again on the pool-aware source. Those retained earlier
passing checks do not claim coverage of this later revision.

Final-source affected Python suite: **42 passed in 46.66s** with the exact
command above. Final release-wheel build, clippy, formatting, ruff and diff
checks pass. The final 256×64 collector gate produced the same 16,384 learner /
34,233 environment decisions and 60 terminal games in 36.25 seconds. New
source-bound receipt: `experiments/data/mtg174/mana-repair-v1/passing.json`,
SHA-256 `03e65050ad454ba78013b5f05b5331baa2a039c7b475fb54dd7b60edbb5b9a35`.
The retained root and completed-cast digests are unchanged from the earlier
iteration above. Final source/runtime hashes:

| File | SHA-256 |
|---|---|
| `managym/src/flow/mana.rs` | `629de390b66ce100d052bd64d47ea8a043eaaf5813a2aa124d3b4171de9cd48b` |
| `managym/src/state/mana.rs` | `92b66a39e11dc7ce9f97b7177f51be1d113dd6c111098bf244960a271197a84e` |
| `managym/tests/rules/temur_cards.rs` | `44715ec26663abffce5e4b5542d8afb10d8f41986c8fcd494bce3d6444f8e6b2` |
| `tests/semantic/test_elemental_world.py` | `208fbe85294d8c2bab8c7c0071003017d256d07336d19e87ef72ec3027e84658` |
| `managym/_managym.cpython-312-darwin.so` | `1c3e3b8ac762e1168badc5814cdb9dcfedd5beaff50ae040353b563492059511` |

## Final gate and next-owner boundary

`cargo test --manifest-path managym/Cargo.toml` completed on the final pool-aware
repair: **489 passed, zero failures, zero ignored**. This includes 294 rules
tests, all 100 ordered Elemental pairs/mirrors, 12 search-state contracts,
semantic conformance/replay, and the uniform vector/reset tests. The long
clone/undo contract suite completed successfully (521.89s); it was not waived.
The final Python suite is 42/42 as recorded above. No shape or compiled-content
change occurred, so this repair does not select a new world or run the separate
shape-change learnability/scientific diagnostic.

Published new evidence at `experiments/data/mtg174/mana-repair-v1/` through
`uv run --extra artifacts python -m manabot.data publish ...`; 16 manifest
entries were added and all pre-existing entries remain byte-identical.
`passing.json` binds the retained root, exact source/runtime hashes and full-size
collector exposure. `validation.json` records the final gates; `debug.log`,
`python.log`, `build.log`, `clippy.log` and `repair.patch` retain their evidence.
The artifact store owns the result bytes; only their manifest is committed.
The required untracked-data regression passed after publication.

The repair step is finished locally; MTG-174 itself remains started/incomplete.
No paid compute/provider allocation, extra scientific arm, Task completion,
parallel worker, PR publication or merge occurred. Actual Etude-launched local
evaluations through game completion with a persistent report owner remain
unachieved. The next execution step owns that bridge/process-boundary repair,
then an actual source/runtime/protocol freeze and fresh live-job/cumulative-cost
admission for the same single inclusive-$10 run and honest baseline comparison.
Keep the old Etude interpreter failure and source prototypes; no scientific
training checkpoint, live-report completion or training outcome is claimed by
these local gate receipts.

Repair commit created with scoped `lf commit`: `f970a3e9c8aa95552bf682f5a94a6033f1bdcdba`.
Its four source/test file hashes exactly match the passing receipt above;
`experiments/data.manifest.json` is the only other committed file. Existing
untracked Flow/skill files and other scratch notes were preserved.

## Mana concept review and adversarial cases — Jack Heart, 2026-10-22

Jack requested a review of the overall payment model followed by deliberately
invented failure scenarios. Jack accepted: **one payment model shared by offers
and execution**. Preserve visible color choices and existing automatic basic-land
payment. No further paid training, intentional stress loops on main, or broad
flakiness hunts are part of this direction.

Verified evidence: five existing Prism regressions still pass (0.16 seconds of
test execution). The new priority Prism + one Treasure + empty pool regression
fails with `illegal mana activation` (0.01 seconds). Priority affordability uses
the all-source feasibility search, whereas mana activation offers and upfront
execution use fixed-source payment. This proves an offer/execution mismatch;
the failed GPU run did not retain the scalar state, so identity with its root
cause remains unproved.

### Proposed usage documentation (not implemented)

Announce a spell or ability, then complete its payment. Automatic payment may
choose fixed sources and concrete pool remainders, preserving a completion of
the announced cost. Chosen mana outputs remain decisions. Activating a filter
may open a nested payment for its upfront cost; resolve that cost before offering
the filter's output, then resume the original payment. Pure mana abilities do
not become stack objects. An offered command always executes at its offered
state. Rejected stale/invalid commands leave the state unchanged.

Concrete interaction: activate Prism with Treasure and no pooled mana. Open
Prism's {1} obligation; activate/sacrifice Treasure; choose its color; pay {1};
tap Prism and choose its output; resume priority (or the original spell payment).
An implementation must preserve the rules' cost ordering and trigger behavior;
this narrative is not permission to precommit costs in the wrong order.

### Proposed implementation-worker instructions (not implemented)

Model a payment obligation with payer, typed cost/use restriction, reserved
sources, nonmana costs and an explicit continuation. Derive offers and concrete
automatic transitions from the same payment transition model; execution commits
those transitions without selecting a different payment. A feasibility result
must have an executable witness, including required visible choices, rather than
being an unrelated boolean search. A witness is not permission to choose the
agent's color. After a visible choice, derive the next legal transitions from
the resulting state. Preserve source identity where tap/sacrifice/grants matter;
grouping equivalent sources is an optimization needing an equivalence proof.
Keep pool arithmetic/restrictions in state/mana.rs and lifecycle/continuations
in the flow authority; do not create a second payment engine in Academy/Python.
Integrate with the current composable optional-payment design rather than
reintroducing WaterbendOrDiscard. The exact type/API refactor remains proposed.

### Bounded scenario matrix / next proof

Each case needs an expected legal path AND a nearby impossible path. Existing
passing tests establish a floor, not comprehensive proof of each family below.

| Priority | Scenario | Failure sought / expected contract |
|---|---|---|
| 1 | Prism + Treasure, no pool; also Treasure already tapped/absent | Offered activation opens executable nested payment; no unsupported offer in impossible variant. Known failing legal case. |
| 1 | Professor {2GG}, one Forest + Prism + other basics; swap permanent insertion order | Generic upfront payment preserves the only second-green route. Existing regression passes. |
| 1 | Serra Angel {3WW}, Prism, WUUUU pool; remove one blue | Concrete pool remainder preserves white; insufficient total stays unavailable. Existing legal regression passes. |
| 1 | Two filters with no seed mana; then add Treasure or one basic | No circular self-financing; a real seed enables only actual payable sequences. |
| 1 | Spell payment -> Prism upfront payment -> Treasure color -> Prism color -> spell | Neither nested choice loses the parent's payer, reserved costs, targets, or resolution frame; commit once. |
| 1 | Herbalist Lesson-only mana funds Lesson versus Prism's activation or non-Lesson | Restriction belongs to the spend purpose, not the outer spell; no laundering through a filter. |
| 2 | Same pool combines unrestricted and Lesson-restricted mana of one color | Spending and memoization preserve both resource classes and available continuations. |
| 2 | Source owes tap/sacrifice as original activation cost, also has a mana ability | No double-use of reserved source; eligible distinct sources remain available. Explicitly distinguish permitted use before paying another cost from actual reservations. |
| 2 | Waterbend artifact can instead filter, or creature can instead tap for mana | The same permanent cannot pay incompatible taps twice; a complete alternative remains selectable. |
| 2 | Granted mana on Forest/creature, summoning sickness, grant leaves battlefield | Source identity, creature tap legality and live grants match actual execution. |
| 2 | Mana bonus when tapping for mana versus waterbend tapping | Bonus counted once on mana tap, never on ordinary cost tap; probes and execution agree. |
| 2 | Generic/colored/hybrid costs, combat/regular pools, cost reduction to zero | No color/class lost by greedy spending; expiry and zero-cost transitions remain correct. |
| 2 | Optional pay-or-discard with nested mana; decline before starting, not after committed costs | Parent continuation and irreversible-payment boundary survive nested decisions. |
| 3 | Equal-output sources with different sacrifice/death triggers | Grouped feasibility does not conflate strategically/rules-distinct identities. |
| 3 | Small bounded boards with permuted source order and irrelevant tapped sources | Payability independent of enumeration order; automatic source tie-breaking may differ where allowed. |

For tiny boards enumerate each offered payment transition on independent forks,
assert successful application, and search to a completed obligation with a fixed
depth/node cap derived from available resources. At each branch mark/apply/undo
and compare witnesses, and replay a retained complete Command sequence. Add an
independent hand-computed oracle for the small resource examples: agreement of
two callers sharing the same broken solver is insufficient. Do not require all
priority actions to preserve a future spell; completion preservation applies
after commitment under the existing engine contract. Capture failed state,
world/runtime identity and action prefix at vector boundaries for the next
production legality failure. No unbounded random game search is required.

## Shared payment model repair — 2026-10-22

Jack Heart's accepted direction here is engine repair and bounded proof only.
The earlier sections' proposed scientific continuation is historical context:
this step neither restarts training/evaluation owners nor arms merge or completes
MTG-174. The four-update GPU failure has no retained scalar root; this repair
does **not** prove that run's exact cause.

Source integration: inspected origin/main at
`20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f` (PR348) and Task 4ba5dd.
PR344's optional-payment delivery is merged at
`43d807036ed86e1899ca6ffc1bc2e54fe559dfdd`; its Task record remained
started/incomplete with a stopped landing process. No process was restarted.
The current Task's supplied Prism/Treasure regression was checkpointed unchanged
with scoped `lf commit` at `a4ffffed4`, then integrated with supported
`lf sync origin/main --manual`. The large overlapping-history merge required
explicit conflict resolution; a premature sync continuation staged conflict
markers, which were fully removed before delivery. The final integration keeps
this Task's scientific code, manifest and receipts, imports current optional
payments plus its dependent decision-context/projection/schema APIs, and retains
the existing CI selection pipeline. No other checkout was edited.

Current execution uses w11 / elemental-world-v2 and schema 7 as delivered by
PR344. The uniform sampler's current consumers select that pack, including
`SemanticPack::optional_payments()`; `elemental_world()` still names frozen v1.
Old scientific protocols/checkpoints/content are not rewritten or relabeled.
The Task's existing shared pool-remainder arithmetic survives rather than
reintroducing separate ordinary and intermediate generic-payment algorithms.

Delete — do not maintain: the grouped boolean `can_cover_cost`, separate
fixed-only mana-ability cost planner, greedy production loop, recursive
waterbend feasibility search, and whole-object reservation helpers
`reserved_payment_source` / `pending_mana_source` / `produce_mana_without`.
Their consumers now use one transition model in `flow/mana.rs`, backed by the
typed arithmetic in `state/mana.rs`.

`PaymentModel` retains source identities, concrete pool remainders, spend
purpose, life budgets, waterbend eligibility and tap/sacrifice reservations.
A successful search returns fixed taps, activations, exact spending, output
modes and waterbend objects as an executable witness. Bounds can reject a
branch but cannot authorize an offer. Equivalent-looking objects are not grouped.
Ordinary fixed production keeps source order. Only a nested automatic prefix
minimizes early commitments while checking the complete parent continuation.

`Decision::ManaUpfront` retains payer, source, ability and recursive parent
decision in the original effect frame. Upfront payment finishes before the
source's tap/life/sacrifice costs and output. A chosen output is still a visible
decision. Nested Treasure -> filter colors restore the correct spell payment,
optional progress or priority; mana abilities create no stack object.
Reservations prohibit incompatible **uses**, not every use of an object:
a selected mana creature can tap for mana and then be sacrificed, while a tap
already owed to the parent cannot fund that same obligation.

The new bounded native cases supplement the preserved Professor, pooled-white,
restricted/granted mana and optional-payment tests:
- Priority Prism + Treasure + empty pool: payment occurs before Prism tap/output.
- Spell -> filter upfront -> Treasure color -> filter color -> spell completion.
  Every offered branch of two tiny boards is executed on independent forks,
  replayed from the full action prefix and rolled back exactly, at depth <= 8
  and <= 256 nodes per tree. Expected totals come from printed costs:
  Treasure adds one; a filter spends one/adds one; Goblin spends one.
- Two filters without a seed are unavailable; a real Treasure seed enables
  payment. The reserved filter is never used to fund its own upfront cost.
- Mixed unrestricted/Lesson-only mana cannot be laundered through a filter.
- Combat versus regular pool spending retains colors and consumes combat first.
- Waterbend taps and mana taps have distinct, independently asserted bonus
  accounting, including a nearby impossible waterbend contribution.
- Nested life fees commit once, with insufficient-life offers absent.
- A selected Treasure cannot also be sacrificed for mana; a distinct creature
  can be the sacrifice instead. Conversely, Llanowar Elves can tap then pay
  a sacrifice cost, while an announced tap cannot double-fund itself.
- A committed optional cost retains its frame through both nested colors.
  Existing optional decline, grant/sickness/removal, stale Command and recovery
  tests remain part of the affected checks.

This is representative coverage of the 15-family matrix, not exhaustive rules
certification. Generic/hybrid and zero-cost arithmetic use the existing state
tests. Equal-output sources with different death triggers are retained by
identity, but there is no new exhaustive trigger-combination corpus. MTG-177
continues to own the broad rules audit.

The finite-source registry contract asserts that every registered chosen-output
mana mode consumes a tap (sometimes also a sacrifice), and has no unmodeled extra
sacrifice/waterbend cost. This makes one mana use per source exact for the current
registry. Separate tap/sacrifice mana uses on one source, reusable free sources,
or additional activation cost methods require a transition-model extension. Search is finite, with admissible total and
spend-purpose bounds; large-board performance is not certified.

Verification also exposed an unnecessary minimal-tap proof on every offer.
Interrupted source-only conformance recordings and the one-second native sample
are retained locally; minimization is now confined to automatic nested prefixes.
Ordinary source-order payment preserves the existing benchmark fixture and its
original version; the temporary fixture changes were removed. Earlier corpus
bytes remain intact. Repaired w11 evidence is separate at
`conformance/semantic-kernel-w11-payment-v1`: 14 terminal cases, 3,491 Commands,
15 inherited Phase comparison rows. Both executors share low-level rules
primitives; independent resource accounting comes from the tiny directed tests.

The Academy reconnect fixture now waits for six lands before casting the
four-mana Lesson. This leaves both optional outcomes available after the filter
fee. Its earlier insufficient-resource fixture legitimately collapsed the sole
decline under skip_trivial; the engine's skip policy was not changed.

Native vector errors now carry private reset configs, seed, skip policy, the
successful scalar action prefix, attempted index and post-error state/offers.
The post-error snapshot may include partial mutation; replay inputs reconstruct
the pre-action root. These details do not enter public scalar errors. A native
test reconstructs the exact diagnostic root. Source/binary identity is bound
separately by the repair receipt, not guessed from the GPU failure.

Historical integrity: all 777 prior artifact manifest entries are unchanged;
all 27 locally retained native-legality-v1 and mana-repair-v1 files match their
recorded SHA-256. No paid resources, new scientific checkpoint, strength
evaluation, owner restart, Task completion or merge arming occurred.


Final acceptance on the repaired source: **423 passed** across the nine affected
debug Rust targets (77 unit, 312 rules, 3 Elemental, 6 canonical replay,
2 registration conformance, 5 experience protocol, 8 state hash, 6 semantic
conformance and 4 semantic IR tests). The Elemental target completes all 100
ordered pairs/mirrors. The bounded tiny-board exact rollback/replay cases are
part of these targets; the separate long search-state contract suite was not
rerun. All 14 source-bound corpus tapes replay successfully.

The pinned root uv release-wheel build and local cp312 extraction completed
before Python verification: **51 passed in 32.89s** across Elemental/Temur,
optional-payment compiler, uniform recovery/admission/CPU checkpoint tests,
matchup/native wrapper, Academy duplicate/reconnect and untracked-data checks.
Final all-target/all-feature clippy with warnings denied, Rust/Python formatting,
ruff and diff checks pass. No full learnability or strength evaluation ran.

New evidence was published through the artifact-store command at
`experiments/data/mtg174/mana-model-repair-v2/`; 20 entries were added.
The passing receipt SHA-256 is
`9d4281ab3eac98e561f6e24c02d847540f0ffd39b286b690adbc11c862714565`.
It binds the current main baseline, 17 source/runtime hashes, final logs,
the repair patch, the new corpus hashes and independent historical integrity.
Earlier prototype failures and interrupted recordings are labeled as such;
they are not final-source passing checks.

| Final file | SHA-256 |
|---|---|
| `managym/src/flow/mana.rs` | `c8bc60c8b89b60a793a697b9fea088aed5978f190d8f6b60b80b3bd45361610c` |
| `managym/src/state/mana.rs` | `d5ffc52c3464c193144557bccd1d9ee51c42f2669dcab70ef7f0a5992a728089` |
| `managym/src/flow/decision.rs` | `f154b69b65dcec95e1ba4839588a16653fe71814058b2641cd8b60c4f9ed6eb3` |
| `managym/_managym.cpython-312-darwin.so` | `0d120a08664c464bf6673d9ae897023c7a03cc6e71bde2d9cf31a872b56ac672` |
