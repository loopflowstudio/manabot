# MTG-241 — composable rules execution

## Outcome and boundary

Jack Heart's accepted directive (2026-11-04): one PR through the code Graph,
stopping at interactive review; no merge, auto-merge, training or retained-evidence
rewriting. The Task remains **incomplete**. No human judgment is missing.
Original base: `4463e57f5`; integrated main: `1546aae3f` via `d83376181`.
Latest committed implementation: `301877ea1` (retained event batches), simplified at
`7b381909e`, preserved by the sync. The current working tree migrates hidden
selection and its consumers, both samplers, public entrants and library placement.
The prior synchronous-commit feedback describes `cf9f5a24b`, not current code.

The sync restored the selected Wave at `.lf/wave/coverage/` by renaming the
former `engine` directory. Its GOAL and complete MEMORY were read on 2026-11-04;
there are no immediate child directories or child memories. The older chapter
centers faithful shared play/training and measured consumer budgets. Jack Heart's
dated Task directive supplies this broader execution scope without reopening the
chapter or authorizing experiments. Durable execution lessons are reconciled in
the selected memory through the code Graph. Detailed prior evidence is preserved at
`301877ea1:scratch/mtg-241-composable-rules-execution.md`.

## Surviving ownership and contracts

Native `Effect` is the sole authored program. `EffectFrame`/`run_frame` own
execution; `Decision` owns the current question. `FrameCompletion` retains one
return destination: caller frame/decision, operation event, spell cleanup,
paid announcement, activated ability, rule continuation or priority. Iteration
retains remaining targets, creates isolated child results and defers parent
suffix/finalization; branches share their scope. `EffectFrame::new` initializes
fresh results. `ObjectSelector::Targets` applies one verb to all eligible slots
in one batch; `ForEachTarget` instead runs a sequential child program per slot.
Leaves from the Vine uses `Targets` for its simultaneous counter placement.

`EventBatch` retains proposals, stable original ordinals, exact replacement
identities (source, definition index and namespace), cursor, departure LKI,
trigger sources, results and consequent child batches. Applicability is rechecked
after every replacement; the affected player chooses the order. No member commits
until all members are ready. Linked returns finish before the root's results are
bound and never replace those results. `bind_event_results` owns instruction
postambles after completion; `run_frame` owns scheduling and resumption.

Snapshots are limited to batches with departures. Departures match pre-batch
sources; entries match the finished battlefield. LKI is captured before members
leave. Earlier paid tap/life events match before sacrifice can remove sources;
triggers remain queued during replacement choices. Lifelink and damage counter
loss commit within the damage batch, without another trigger pass. Entry counters,
including Saga lore, pass through replacement before entry; emitted counts are
transformed counts. `apply_immediate_event` uses this same executor with an explicit
no-question precondition, never a pending-as-success adapter. The batch owns trigger
processing, including for immediate zone moves.

Mana activation transfers its caller by ownership. `advance_mana_upfront` commits
the automatic mana prefix, tap and life costs, then retains output and original
payment before running sacrifice. Linked-entry replacement choices can suspend
that cost without exposing output, paying again or granting priority. Activated
source/target metadata is captured before sacrifices; card-ability source identity
binds after discard cost. Spell entry/finalization, combat damage, SBA departures,
token creation, Earthbend, library placement, chosen exile/sacrifice groups,
counter distribution and LookAndSelect battlefield movement use retained batches.
First-strike retains normal damage across its SBA interval; SBA progress is
Stable/Changed/Pending. Token/attachment cleanup waits for child work, and
`turn_tick` exposes pending turn actions before stabilization.

`SelectCards` now migrates public results and hidden hand/library selections.
`ZoneSelection` retains owner, zone and predicate on decisions and selected child
scopes. Both samplers remap unknown commitments across the saved caller chain
with one exact-object matching; viewer-known choices remain exact. Incompatible
restricted count worlds are excluded, and materialization rejects impossible
commitments instead of dropping them. Direct determinization retries bounded
compatible deals and returns an explicit error on exhaustion. Ordinary roots take
no additional full clone. Requirements are captured before zone reassignment;
remapping follows selected/remembered/result aliases across caller frames. Mixed
foretell allocations count only compatible ordered fills, rather than retaining
all their weight after a feasibility check. The directed two-selected-instant /
two-foretold-slot witness retains 14 of 20 physical fills and resumes both samplers.
The frozen count-only belief sampler explicitly rejects retained-selection and
foretold domains before inference; its saved schema is not silently extended. Library inspection exposes all permitted members in
stable definition/object order, never library order. Pending battlefield entrants
are public and pinned even before their retained moves commit.

`OptionalDiscardDraw`, `MayDiscard`, `SearchBasicLand`, `SearchLibraryToHand` and
their dedicated continuations/helpers are deleted. Definitions use selection,
movement, conditional effects and shuffle; optional discards commit the completed
subset together, and draws read committed `Moved` results. Library-to-hand movement
reveals definitions and updates known-hand minima. `CardsThisWay` retains latest
mill/discard independently of later moves. Shared selection remains code 28;
6/7 join the retired codes. Existing mill consumers retain their earlier contracts.

`LibraryEnd` now lives with the authored rules types. `Move.library_end` commits
explicit top/bottom placement with the movement, before linked-return questions;
shuffle remains a subsequent instruction. The actual Merchant program's constructed
linked-return witness covers bottom placement, deferred shuffle and completion.
Real Merchant tests exposed sacrifice trigger predicates being read after their
subjects departed: committed sacrifice facts now match all pre-move subjects before
batch zone moves, with trigger placement still deferred.

`OrderLibrary` owns top/bottom ordering. Bottom keeps immediate placement and
remaining-only private visibility; top keeps the full inspected prefix pinned and
visible until complete. Rotation preserves the remaining offer order. Surveil
retains its graveyard partition privately until selection finishes, moves it as
one retained batch, then orders kept cards. Scry/Surveil completion events wait for
all choices (including empty-library completion) and match sources before the
caller's suffix can remove them. Planetarium recognizes both keywords. No current
real definition prints Surveil above 1: constructed programs prove arbitrary
kept order, while Joo Dee proves the registered interaction.

`SuspendedResolution::continuation` owns the root-first borrowed walk through
frames and saved mana decisions. Projection, inspection and determinization share
that ancestry. Existing obligation codes remain fixed: bottom 23; top 27 (mode =
committed prefix length); shared selection 28 (legacy 5 retired); replacement 29.
Labels, source focus, text and Academy name those choices. These surfaces **do not**
complete retained payload observability. Prospective entry conditions now use the
entering controller; SourcePower reads exact current/LKI power, never a returned
incarnation.

## Delete — do not maintain

Completed cuts (history and original checks at the checkpoint references below):
- Local ForEachTarget subframe loop/no-decision assertion; owner-based entry
  conditions; live-only SourcePower; repeated fresh-frame literals.
- Independent frame parent/finalize fields and `FrameFinalize`; cloned mana
  resume chains/placeholder frames; post-call ManaUpfront mutation.
- `OrderLibraryBottom`, `library_bottom_decision`, `MillAndSelect`,
  `ReturnMilledThisWay`, `MilledBonus`, `Decision::SelectMilled` and duplicated
  Lesson/non-Lesson selection branches.
- One-pass replacement sorting, synchronous recursive linked returns/Saga counter
  insertion, `finalize_frame`, synchronous sacrifice/destruction/moved-object and
  explicit-source processing wrappers, post-call damage measurement and combat
  hit loops. Leaves' per-target counter loop is gone.
- Obsolete SBA event-reference/controller collection and immediate zone-move
  trigger postamble. Surveil uses the shared batch-to-decision handoff; batch
  binding iterates moved results directly unless the frame must retain them.
  Shared entry-replacement fixtures remove three copies of identical setup.

Completed hidden consumer cut: optional discard/draw and basic/tutor search
handlers and their private search helpers, including synchronous source-bottoming.
`ZoneSelection` owns hidden-source and slot eligibility checks; count-world
admission and object rebinding share complete bipartite assignment. Inherited
copies of the same constraint are retained once. Migrated definitions share
`CardPredicate::basic_land()` without changing the authored rules values.
Distinct-land split searches, named searches, Scry and LookAndSelect retain their
operation-specific semantics; their remaining overlap must be evaluated with their
visibility, revealed-prefix and ordering contracts rather than deleted blindly.

## Remaining implementation and acceptance

1. Complete the real Leaves from the Vine / Caradora / Doubling Season witness,
   including complete definitions and learning/observation coverage. Current
   `ModifyCounters` drivers use `count * multiplier + additional`; no registered
   card uses that replacement yet. The registered Leaves chapter with constructed
   drivers proves simultaneous placement, not those two complete cards.
2. Finish the hidden-state acceptance audit beyond the migrated consumers.
   Current witnesses cover partial hand/library selections, restricted incompatible
   deals, saved callers/inherited bindings, public precommit entrants, both samplers,
   Commands/replay/stale rejection, divergent forks, frame remembered aliases,
   conditional foretell weights and materialization. Count-only learned sampling
   now rejects unsupported domains before inference. Audit private remembered
   subjects after their selecting frame has ended and unknown legacy
   Scry/LookAndSelect callers, preserving viewer-known inspected prefixes. No claim
   of complete hidden-world coverage follows; new support for the frozen learned
   sampler needs an explicit successor contract, not post-sampling rejection.
3. Project pending batches (including precommit entrants), iteration progress,
   selected/this-way results, completion events and paid/rule callers. Paired states
   with identical current offers must distinguish consequential viewer-safe payloads
   and hide unavailable bindings. Cover native/Python inputs, learning tokens,
   actual model input, LLM, Academy, replay/fork and reconnect. Reject unrepresentable
   older inputs before inference; obligation codes alone are insufficient.
4. Reconcile the successor schema with MTG-244 through ordinary integration. Its
   shared risk directive explicitly names **v8**, but shared comments supplied no
   concrete layout; this Task has not assigned a competing v8. Preserve v1–v7,
   fresh v6, saved omissions, capacities, original action-prefix indices/Commands
   and omitted-overflow-is-error. World/hash, capture boundaries and new conformance
   corpus remain unfinished.
5. Complete Jack Heart's [performance/learning-risk acceptance](mtg-241-risk-validation.md).
   Implementation owns bounded ordinary-step/search probes, equivalent-work versus
   added-branching attribution, targeted mitigations and remeasurement, and paired
   policy/model and hidden-information witnesses alongside the new representation.
   Gate owns final-head same-runner measurements, full debug Rust/Clippy, affected
   Python/headless Academy, current-world admission and Engine Budget. These checks
   cannot substitute for the unfinished implementation or its targeted risk work.
   No speed or learning claim follows from the refactors; no experiment was launched.

The hidden consumer deletion is implemented in the working tree. The next
architectural cut is complete retained-payload projection and explicit version
admission, with the remaining hidden-state consumer audit above. The complete
real-card counter witness and bounded performance/learning-risk measurements are
still acceptance work; this Task is not ready for review or completion.

## Integration and evidence

MTG-242 owns payment resources/search; MTG-243 naming; MTG-244 layers. Dated shared
reads (2026-11-04, this reconciliation) found MTG-242 in compress on its resumed
code Graph, MTG-243 in concept-review with open
[#424](https://github.com/loopflowstudio/manabot/pull/424), and MTG-244 in compress
on its second loop. Publication does not establish integration. Task comments
carry FrameCompletion, paid-output protocol and codes 27–29. MTG-244 received the
schema coordination request and the working-tree `ZoneSelection`/`LibraryEnd`/
public-entry interfaces (comment `dffd3f70-4dc9-462c-8fbc-6706b162ef76`). Other checkouts were neither edited nor copied.
Main's selected casting-cost/X fixes and buffer/overflow contracts remain intact.
The MTG-213 first-64 pilot exception does not authorize truncating new choices.

Jack Heart's publication handoff makes [#425](https://github.com/loopflowstudio/manabot/pull/425)
the source of the approved PRINCIPLES.md/README guide, now integrated through
`d83376181`. The published guide supersedes the older embedded draft and links PRs
#42, #348, #353, #289, #412 and the speed investigation. Research publication does
not complete this Task. Its [milestone](../docs/research/standard-2026-11/milestone.md)
places execution/results beneath later costs, history and object-lifetime work;
those proposed cohorts and separately authorized training Tasks do not expand this
Task into another executor, content rollout or experiment campaign.

Rules evidence previously checked:
- [Guilds release notes](https://magic.wizards.com/en/news/feature/guilds-ravnica-release-notes-2018-09-20):
  surveil triggers after completion, including an empty library.
- [Aetherdrift notes](https://magic.wizards.com/en/news/feature/aetherdrift-release-notes)
  and [Foundations notes](https://magic.wizards.com/en/news/feature/foundations-release-notes):
  affected-controller counter ordering, including entry; (1 + 1) × 2 = 4 vs
  (1 × 2) + 1 = 3. These are behavior requirements, not complete card proofs.

Historical checks and limits:
- `37928095d:scratch/mtg-241-composable-rules-execution.md`: 43 focused debug tests,
  two contextual card tests, one parity case and rebuilt extension. Nested choices
  cover both seats, two levels, forks/replay/stale rejection and deferred SBAs;
  departed Combustion Man reads seven-power LKI rather than returned four power.
- `e6b268956:scratch/mtg-241-composable-rules-execution.md`: completion/payment and
  ordering checks. The constructed sacrifice-then-Surveil witness only isolated
  completion ownership; it did not prove actual linked-entry suspension.
- `cf9f5a24b:scratch/mtg-241-composable-rules-execution.md`: after main sync, 38
  focused debug tests, inherited card/trigger checks, four Python text/parity cases
  and atomically installed extension.
- `301877ea1:scratch/mtg-241-composable-rules-execution.md`: five retained-batch
  Command/fork/replay/stale witnesses, Surveil and code-bound checks; replacement,
  combat, SBA, activation, strike/lifelink filters; three Leaves tests and selector
  token guard; rebuilt/atomically installed extension and Joo Dee text case.
  This checkpoint proves actual mana payment → sacrifice → linked-entry replacement
  → output → original payment exactly once, newly applicable/once-only replacements,
  two-object 3/4-counter commits, simultaneous combat and SBA linked-return/LKI.
  Full projection/model/performance acceptance remains open.

- `7b381909e:scratch/mtg-241-composable-rules-execution.md` records the simplified
  Rust diff SHA256 `a9b4e963263e4db02f60b3b04633649d8c975580d9ceded6924078a69db5a8f8`
  against `301877ea1`: 34 debug resolution, 10 SBA and 5 replacement cases;
  formatting, release extension build/atomic installation and two Python text
  cases passed. This is reused evidence, not a new full-head test run.

Check (2026-11-04): working-tree diff SHA256 `01e9dfdb10f02ed68e360b9aebee69c4fdce5934d7c20b56ba3de2b31b02f9bf` against `d83376181` (`git diff -- managym manabot/belief/sampling_data.py tests/belief/test_sampling_data.py`, new sampling module admitted with `git add -N`): debug `--lib hidden_selection` 6 pass; retained-batch filter 5, migrated-card filters 6, search filter 13 and both effect/selector token guards passed on the applicable unchanged code; root release maturin build/atomic cp312 installation and the two text/parity tests plus `test_count_sampler_rejects_retained_private_subset_before_inference` 3 pass; focused Ruff, `git diff --check` and `lf context` pass. Full affected suites remain gate-owned; the remaining implementation and bounded risk measurements above are not deferred to gate.

Compression check (2026-11-04): same diff command now hashes to `0bb2f485f6b7dfc55d60e2ca1c4d3b9fde16522bb11f8240e9b918c84af939b4`; `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path managym/Cargo.toml` with `--lib hidden_selection` / `-p managym-cards --test cards search` passes 6 / 13 tests. Root release maturin build and atomic cp312 installation pass; the complete learning-catalog JSON is unchanged (SHA256 `4dfdaca1ca80392d4fc9643ffdf8e15224fef432fec81a9e3882ca088fe8997b`). `uv run --extra dev pytest` shared-mill text/native/Python parity and retained-selection sampler rejection pass (2); formatting and diff checks pass. Gate retains the full affected suites and final-head acceptance; no runtime or learning-performance claim is made.
