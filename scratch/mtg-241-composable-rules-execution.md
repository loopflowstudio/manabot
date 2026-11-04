# MTG-241 — composable rules execution

## Outcome and boundary

Jack Heart's accepted directive (2026-11-04): one PR through the code Graph,
stopping at interactive review; no merge, auto-merge, training or retained-evidence
rewriting. The Task remains **incomplete**. No human judgment is missing.
Original base: `4463e57f5`; integrated main: `8e8d702f8` via `58535f9f2`.
Latest implementation checkpoint: `301877ea1` (retained event batches).
The supplied feedback about synchronous commits describes the earlier
`cf9f5a24b` checkpoint, not current code.

`lf context` reports the `coverage` binding, but local Wave GOAL/MEMORY files
are absent. The supplied Task directive is the available scope; Wave memory
has not been edited. Detailed prior implementation/evidence is preserved at
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

`SelectCards` currently migrates **public-zone results**: exact selector candidates,
min/max cardinality and a selected child scope. `CardsThisWay` retains the latest
mill/discard independently of later `Move`. Avatar Destiny preserves its source
return between mill and selection; Ostrich-Horse keeps decline counters; Sparring
Dummy reads its caller's mill result for the Lesson condition. The Command witness
covers partial selection, divergent forks, exact-incarnation rejection, a child
pause and one finalization. Do not extend these snapshots to hidden consumers
before the materialization cut below.

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

Remaining deletion cut, **with consumer migration and materialization together**:
- OptionalDiscardDraw, MayDiscard, SearchBasicLand and overlapping search/move
  handlers and dedicated decisions. Preserve decline, keyword events, private
  inspection, reveal/shuffle, ordered remainders and result binding.
- Synchronous hidden-search movement/finalization postambles; search's
  source-bottoming must retain linked returns. Do not polish these predecessors.

## Remaining implementation and acceptance

1. Complete the real Leaves from the Vine / Caradora / Doubling Season witness,
   including complete definitions and learning/observation coverage. Current
   `ModifyCounters` drivers use `count * multiplier + additional`; no registered
   card uses that replacement yet. The registered Leaves chapter with constructed
   drivers proves simultaneous placement, not those two complete cards.
2. Migrate hidden selection and delete the composites above. Both `Game::determinize`
   and `HiddenHandConstraints::materialize_validated` reassign hidden membership,
   repin inspected library prefixes, then refresh only the active offer. Retained
   exact hand candidates can therefore become library members. Add source/visibility
   provenance and traverse saved decisions/child scopes; preserve viewer-known
   membership and coherently resample unknown selected subsets against the sampled
   zone/predicate. A partial restricted search also constrains compatible deals
   (enough eligible objects must remain). Dropping missing selected IDs erases prior
   commitments; updating only current offers cannot repair saved callers. Audit
   immediate callers during this cut.
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
5. Follow Jack Heart's [performance/learning-risk acceptance](mtg-241-risk-validation.md).
   Gate owns bounded same-runner ordinary-step and search total/tail measurements,
   equivalent-work vs added-branching attribution, profiling and remeasurement of
   targeted mitigations, paired policy/model checks, full debug Rust/Clippy, affected
   Python/headless Academy, current-world admission and Engine Budget. No speed or
   learning claim follows from these refactors, and no experiment was launched.

## Integration and evidence

MTG-242 owns payment resources/search; MTG-243 naming; MTG-244 layers. Dated shared
reads (2026-11-04) found MTG-242 idle after its second loop-or-next, MTG-243 published
for review at demo ([#424](https://github.com/loopflowstudio/manabot/pull/424)), and
MTG-244 implementing. Publication does not establish integration. Task comments
carry FrameCompletion, paid-output protocol and codes 27–29. MTG-244 received the
schema coordination request. Other checkouts were neither edited nor copied.
Main's selected casting-cost/X fixes and buffer/overflow contracts remain intact.
The MTG-213 first-64 pilot exception does not authorize truncating new choices.

Jack Heart's publication handoff makes [#425](https://github.com/loopflowstudio/manabot/pull/425)
the source of the approved PRINCIPLES.md/README guide when syncing. Preserve newer
edits; do not republish the older embedded draft. Current guide links public PRs
#42, #348, #353, #289, #412 and the speed investigation. Research publication does
not merge or complete this Task.

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

Check (2026-11-04, `301877ea1` + Rust diff SHA256
`a9b4e963263e4db02f60b3b04633649d8c975580d9ceded6924078a69db5a8f8`):
`CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2 -p managym --no-default-features`
with `--lib flow::resolution::resolution_tests` passed 34; `--test rules_tests`
filters `cr_704_sba` and `cr_614_replacement` passed 10 and 5; `cargo fmt --manifest-path managym/Cargo.toml --all --check`
passed; root `CARGO_BUILD_JOBS=2 uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`
passed and cp312 extension installed atomically; `uv run --extra dev pytest tests/semantic/test_text_observation.py -q -k 'joo_dee_surveil or shared_mill_selection'`
passed 2; `git diff --check` and `lf context` passed. Gate owns broader suites
and performance/model-risk measurements; hidden selection/projection remains implement-owned.
