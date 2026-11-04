# Coverage memory

Carried from the legacy `rules` wave, subsequently named Engine and renamed
`coverage` on 2026-11-04 (`da355b562`).

## Operating principle

> Build the world, play it, measure it, then isolate what surprised us.

Rules leads with playable curated worlds and real search/Study consumers.
Conformance, fuzzing, benchmarks, and micro-katas are instruments for observed
ambiguity or safety-critical invariants, not substitute products.

## Learn decisions and current evidence

- On 2026-10-04 Jack Heart authorized autonomous implementation/publication/
  landing, with ETU-75 kept open for unmet empirical acceptance. Jack then
  directed synchronization with main `3f297533`: w4 was that delivery's rules
  world, superseding its w3 plan. Current world identity is in `WORLDS.md`.
  ETU-89 owns training regimes and consumes
  the ordinary checkpoint writer/loader contract owned here. Merge `b104fe94`
  preserves w4. Complete semantic-program and public known-hand inputs now
  reach the ordinary Agent; mandatory full-setup bindings protect ordinary
  writers/loaders and configured play/search consumers. Synthetic checkpoint
  proof does not admit a retained trained challenger.

- On 2026-10-04 Jack Heart authorized ETU-89 parent PR #200 to carry the
  remaining verified ETU-75 provider code when serial PR rotation failed.
  Provider head `aba61859` preserves `checkpoint_world` and
  `validate_checkpoint_world` signatures and passes 92 focused checks.
  V4 passed all 16 games and 3,843 Commands against that source with
  byte-identical v3 tapes and no mismatches.
  PR #200 merged at `e30b7b82` with all ten CI checks passing. The receipt-only
  follow-up retains v4 as historical provider proof, not integrated-parent proof.
  ETU-89 received the exact head for supported branch sync; old merged PR #188
  must not be presented as containing these newer commits. Keep ETU-75 open
  and retain historical gate failures and missing human-play evidence.

[Complete Learn/Lesson in the selected matchup · ETU-75](https://linear.app/loopflow/issue/ETU-75/complete-learnlesson-in-the-selected-matchup)
remains unfinished. The [durable Learn record](../../../docs/rules/learn-lesson.md)
preserves accepted decisions, dated proof limits and source history.

- The September reviewer (name unresolved) approved open-decklist sideboards on 2026-09-24: UR gets
  one each Firebending Lesson, It'll Quench Ya! and Accumulate Wisdom; GW gets
  one each Yip Yip! and Fancy Footwork. Preserve the 41/40-card main decks.
  Admission of an ordinary sideboard card is separate from Lesson eligibility.
- Owned outside copies have immutable rosters and zone-derived availability.
  Retrieve through ordinary movement after a definition reveal; never create
  copies, invent a zone or duplicate the inventory in consumers. Carry complete
  compiled setup through seats, reset and replay roots.
- Known hand facts are definition minima with hash coverage.
  Both search samplers must reserve those facts and weight residual deals,
  without exposing which identical copy was retrieved.
- Mode-first controls, previews, Back and reconnect have real browser evidence.
  On 2026-09-25 that reviewer said “works. approved.” This accepts the demonstrated
  working interaction; it supplies no missing per-attempt tapes and does not
  authorize publication or completion. Do not request duplicate acceptance.
- On 2026-10-04 complete policy programs/knowledge, ordinary w4 bindings and
  configured Search in both seats gained focused execution proof. The
  prospectively registered v2 cohort passed all 16 games and 3,843 Commands
  with 19 retrievals, 25 discard/draws and 14 declines. Final-source v3 also
  passed all 16 cases after import-isolation and diagnostic repairs; its tapes
  are byte-identical to v2, with zero state/consequence/viewer mismatches.
  All receipts and the earlier unscored v1 registration remain retained.
- That Learn gate was not green: the broad Python run had 813 passes and 30
  failures; ten current-test/import failures were repaired with focused proof.
  Twenty historical authority/advice and experiment checks remain incompatible.
  In particular, frozen advice binds action ABI 3 / possible-world ABI 1,
  versus current 6 / 2, and its retained checkpoint lacks the ordinary w4
  binding. The repo gate explicitly preserves frozen advice failures as
  delivery blockers; do not rewrite them to make the branch green.
  ETU-75 stays open. No trained-challenger admission, missing human tapes,
  strength or runtime-budget result is supplied by synthetic checkpoint tests.
- Changed rules invalidate compatible-looking old evidence. Preserve frozen
  receipts and register all 16 seed/seat attempts before scoring the corrected
  world. Human interaction acceptance and consumer conformance are separate
  claims. No strength or numeric runtime-budget result follows from this work.

## Shipped substrate

- The process-wide `ContentPack` catalog separates immutable definitions from
  compact mutable match facts. Environments and branches share definitions
  without copying card content; authored presets are decklists over that catalog.
- Stable `CardDefId` values and deterministic state witnesses reproduce seeded
  traces without allocation-address or debug-format identity.
- Card content is the engine's own `CardDefinition` written in its rules
  types in `managym/cards/src/<set>.rs`, one function per card, with shared
  types owned by `managym/state` (MTG-238, main `44388ad42`); the
  earlier checked-in typed IR and its compiler were deleted on 2026-10-30
  (`10eeb1f55`). There is one card representation and no card-name dispatch.
- Priority, targeting, and attacker declarations have uncapped typed offers and
  atomic revision-bound commands. The structured decoder preserved 6,435/6,435
  shared decisions and handled more than 32 legal choices.
- Viewer-safe semantic program projection exposes complete ragged programs with
  explicit structure, binding, schema, and pack identity and rejects unknown
  primitives or silent truncation.
- Object incarnation and LKI, proposed-event replacement/prevention, and
  trigger/SBA fixpoint behavior are explicit for the curated slice.
- Exact fork/root/sibling isolation, nested rollback, deterministic replay,
  viewer projection, stale-reference handling, differential execution,
  property/metamorphic checks, bounded fuzzing, and a pinned Phase overlap
  matrix exist as reusable instruments.
- Study can exact-fork a retained historical managym root, execute a normal
  structured command on one sibling, prove the source and other sibling did
  not move, and consume a `StudyReturnReceipt` bound to the captured source
  digest. This proves the adapter seam, not yet the unified `MatchAuthority` or
  arbitrary historical manabot evidence resolution.

## Runtime evidence

- Jack Heart's MTG-237 brief (2026-11-02) requires a 10% engine budget,
  deliberate baseline moves, search-score preservation and a separate headless
  throughput workload. The implementation compares candidate and committed
  reference on one CI runner: identical T4 code measured 6.8k–10.3k steps/s
  across runner CPU models, so recorded wall times cannot budget another run.
  Jack Heart subsequently accepted ordinary `Env::step` as the training proxy
  during MTG-238 (recorded in `docs/development.md`, delivered `44388ad42`);
  this throughput is not comparable with the retired rul-9 Python Command
  runner's 588.5 steps/s or an end-to-end training measurement.
- Search correctness needs every ordered action score and decision boundary.
  A digest of only simulation count and winning action misses changed losing
  scores. MTG-237 now hashes full scores; its historical T4/T5/T6 measurements
  retain the old limitation and establish no full-score equality. The new
  first full-score baseline came from CI run 41540798745, not reinterpreted
  historical output. Delegated demo judgment on 2026-11-02 subsequently
  accepted run 41552162812 at `88dc8c1ad` as the initial reference for merged
  main `a8abf01b1`, explicitly retaining its measured 10.98% greater per-step
  cost against the older engine. This is not a human statement or shipping
  approval. The tolerance stays 10%; the source comparison does not isolate
  which upstream change caused the cost.
  Detailed measurements and the adoption procedure live in
  `docs/development.md#engine-performance-budget`.
- A green CI job does not prove budget enforcement: PR run 41548559113
  (2026-11-02) reported success despite a throughput FAIL because implicit
  `bash -e` let `tee` mask exit 1. Explicit `defaults.run.shell: bash` enables
  pipefail. Regression coverage must execute the workflow's pipelines, not
  just the comparator CLI, and cover failed benchmarks as well as over-budget
  or changed-digest comparisons. Repaired hosted run 41552162812 failed the
  throughput comparison correctly and retained a downloadable, adoptable
  artifact. Its search timings and all digests/counts passed; 65 focused
  tests cover the controlled pipeline and recovery paths.
- Record the candidate before building the reference so an incompatible
  reference still leaves an adoptable measurement; upload follows in the final
  always-run step. A controlled reference-build failure proves local adoption
  with the real recording CLI. Hosted transport and adoption after comparison
  failure are proven by run 41552162812; hosted reference-build failure
  remains unexercised. Moving digests, accepting a cost or
  retaining a speedup requires a reason and the candidate's own CI artifact.
  Measure the durable PR head, not the ephemeral merge commit.
- Main `51fbacbbf` subsequently merged MTG-236's derived-state revisions,
  indexed records and cache-carrying clones. Run 41555845634 (2026-11-03 UTC)
  measured that engine at `4b5e9c0e4` against `88dc8c1ad` on one runner:
  worst reply took 30.26% less time, total search 28.36% less and each step
  36.65% less, with identical full-score/step digests and counts. Delegated
  demo judgment on 2026-11-02 applied the retained-speedup rule; the production
  reference then adopted that candidate's artifact with a reason and unchanged
  10% tolerance. Retaining the older, slower reference would allow loss of
  the optimization within the budget. This is curated native-workload evidence,
  not an end-to-end training result or shipping approval; the earlier accepted
  cost remains historical evidence.
- MTG-238 delivered the next reference at main `44388ad42`: candidate
  `1616b210f`, CI run 41699683808 (2026-11-03 UTC), with matching full-score/step
  digests and counts on one runner. It retained 49.91% less worst-reply time,
  43.93% less total-search time and 30.43% less time per step against
  `4b5e9c0e4`, preserving the 10% tolerance. The baseline and investigation in
  `docs/development.md` / `docs/research/engine-speed-2026-11.md` own the
  measurements and rejected approaches. This upstream proof does not certify
  MTG-239's corrected rules or any subsequent merged tree.
- Rules corrections can change benchmark work even when the decklists stay
  fixed. MTG-239's PR #415 run 41711692234 (2026-11-03) passed same-run timing
  checks but changed all three search digests and the step digest/count
  (99,923 to 101,650); the old reference reproduced its committed digests.
  Those decks exercise the corrected ordering and contextual-power rules.
  The branch adopted its own candidate artifact with a reason and unchanged
  10% tolerance. This is a rules reference update, not equivalent-workload
  speedup evidence or acceptance of the subsequent repaired head.
- Compact full clone is retained because mutable state is dense, immutable
  definitions are shared, and clone-plus-undo and page-COW did not justify a
  production split under measured workloads.
- RUL-2 ran `full_clone/current_game_v1` through the authored PUCT teacher with
  revision-bound Commands at world, child, and leaf. Selected/reference traces
  were exact with zero fallback; selected delivered 35.76 decisions/s at
  interactive p95 65.99 ms and 4.70 decisions/s saturated at p95 1343.29 ms
  with matched RSS.
- PR #136 recorded a checked seed-0 UR Lessons versus GW Allies release-stack
  authority trace: terminal revision 132, 132 Commands across nine prompt
  families, 26 admitted typed programs, ordered semantic/presentation events,
  and zero legacy fixed-action, card-name, candidate-cap, or client-legality
  fallback. Authored replay parity, workload budgets, and another
  creator-selected increment remain open.
- PRs #129 and #134 provide a Rules-owned Study fork over compact full clone.
  A historical viewer-safe decision can execute a structured command, preserve
  retained source and siblings, and return a consuming receipt with the exact
  canonical source digest, frame, offer, command, event cursor, and
  continuation. Retained-root drift fails closed.
- Revision-bound receipts identify a reproducible relevant source closure.
  Incidental worktree paths, nested agent worktrees, and unrelated files are
  not source identity.

## Durable design decisions

- The destination is the creator's selected decks and matchups, not general
  Magic or Commander support.
- Steal Phase's invariants, not its shapes: incarnation and LKI, typed meaning,
  proposed-event replacement flow, explicit legal interaction, and viewer-safe
  projection.
- Prefer typed rules data over runtime natural-language parsing. Since
  2026-10-30 that data is the Rust `CardDefinition` itself, not a separate IR.
- `Effect` is a verb over an `ObjectSelector` (Jack Heart's concept review,
  2026-11-02, MTG-235). The object a verb acts on (`This`, `Target`,
  `Triggering`, `Attached`, `CreatedThisWay`, `Grantor`, `Moved`, `You`, `Chosen`,
  `Each`) is never written into the verb's name; one `Game::select` resolves
  it, with last known information for a departed source. A verb whose rule
  depends on which object it hit is an `If` composition ("if you do" is
  `Condition::Moved`). Reason: each fused variant was its own resolution arm,
  observability row and opaque learning token, so nothing transferred between
  cards and the slope reached ~400 variants at Standard scale. The accepted
  amendment folds `AffectedSpec` into `ObjectSelector`, merges value twins on
  `EffectValue`, and uses the base verb over `Each` for batch effects. There
  are 70 Effect variants (previously 107) and 14 remaining object-in-name rules
  in the plan's inventory. Directed card behavior tests are unchanged; the
  interrupted run's 485-card debug pass predates recovery fixes, so it does
  not establish final-amendment acceptance. Apply the same test to
  future mechanics: a new rules-type variant names a new rule, not a new
  object for an existing verb.
- `Chosen` suspends a supported object verb before selection, binds its chosen
  permanent for that instruction, then clears it even when a nested counter
  choice stops. Player-affecting verbs resolve the same selector through
  `resolve_affected`; stack abilities have no card `ObjectRef`, so
  `CounterAbility` reads its retained clause target. `moved` records the latest
  Move/Sacrifice independently of discard/mill's `this_way`; an impossible or
  declined choice must clear it before a following "if you do". These are
  recovery lessons from MTG-235 (2026-11-02), covered by focused selector tests.
- Learning tokens and the content digest derive from definitions, so a
  vocabulary change moves them without changing behavior; w17 evidence
  (conformance, authored-match receipts, parity corpora, Academy fixtures,
  belief receipt pins) is re-recorded in place with no world or policy-input
  version bump (MTG-235, 2026-11-02; see WORLDS.md). A version number names a
  frozen world, not a merge.
- Jack Heart clarified on 2026-11-03 (MTG-239, comment 1f3876a7) that known
  rules bugs must not block previously playable cards; categorize defects as
  bug Tasks. Playable does not mean bug-free. This supersedes the assistant's
  strict-completeness admission proposal, while retaining MTG-239's five
  fixes and whole-deck proof. Jack Heart explicitly limited that work to those
  fixes, without a broader policy redesign or additional Tasks.
  On November 3, Jack Heart accepted removal of the empty exclusion API/list
  and pool branch (comment 6b64d212), preserving token, rules-fixture and
  unknown-card rejection. An empty exclusion list still leaves bug-based
  eligibility as a supported policy; deleting that mechanism and reconciling
  its documentation makes admission agree with the decision.
- Admission tests need stable reasons for rejection. MTG-239's first published
  CI run found three native and four training tests still treating repaired
  cards as unsupported. Token fixtures now preserve the negative paths,
  alongside positive repaired-card admission and unchanged roots after a
  rejected replacement. Tests of source, checkpoint and replacement admission
  are separate consumers of the same policy; a catalog-only pass misses them.
- Off-battlefield power is contextual: CDAs read the spell controller on the
  stack and the owner in other zones, including outside the game. Battlefield
  power reads the layered view, including ability loss. Zone filters and
  viewer card rows must use the same reader; printed-star fallback allowed
  Rallier to select an ineligible Suki (MTG-239, 2026-11-03).
- Owner-chosen library order is a shared native decision, with ordinary
  SelectCard Commands, private inspection, explicit placement direction and
  forced-singleton skipping. A test accepting either remainder order cannot
  establish that the player was offered the choice. Native continuation and
  LLM text need direct checks: captured viewer JSON omits continuation frames.
  Reconnect and duplicate submission must preserve the actual pending order.
  Jack Heart selected immediate ordering with clearer draw-order wording on
  2026-11-03 (comment 6b64d212): each choice commits the next card to be drawn
  among the remainder; the final placement is automatic. A whole-order draft,
  confirmation or undo lifecycle is not approved. Retained-session reconnect
  evidence does not establish process-crash durability.
- Correcting a legal choice can change a retained game's outcome, not just
  its hashes. MTG-239's Rallier filter changed the open-w17 authored trace
  from 142 Commands / 69 commitments to 135 / 66; Accumulate Wisdom inserted
  explicit ordering Commands in navigation prefixes. Combined-runtime parity
  after MTG-238 sync covers those current traces, not frozen evidence or
  full rules acceptance. The prior tape remains at
  `22e4aab71:conformance/authored-match-w17-v1/`; MTG-213's frozen corpus and
  projected increase in eligible decks remain separate evidence.
- Unknown primitives and oversized programs fail admission; they never become
  a generic unknown token or a silently truncated sequence.
- Structured offers are legal by construction. Flat enumerate-then-cap action
  lists remain compatibility adapters, not the destination.
- Use safe outer forks and dense transactional execution where real workload
  measurements justify them. Persistent collections, page COW, or undo are not
  adopted by doctrine.
- A historical Study identity resolves to an exact viewer-safe fork, executes
  normal structured commands, preserves the recorded source, and returns to the
  original state and event cursor in one action.
- A new diagnostic must name the play, search, or Study behavior it explains
  and the runtime decision it can change.

- Catalog-scale types (MTG-238, 2026-11-02): counters are sparse by kind,
  keywords are a `Keyword` enum in a bitset, card sets are the `managym-cards`
  crate on `managym-state`, and every duration-bound change is a continuous
  effect record, never a `Permanent` field. A type change that preserves
  serialization moves no digest; removing a serialized field moves the
  state-hash contract and re-records every corpus that embeds state digests
  (semantic-kernel, authored-match, both parity receipts, Academy fixtures).
  Jack Heart approved batching the `BecomeArtifactCreature` fold,
  keyword-list serialization and count-to-condition rewrites on 2026-11-02.
  Learning schema 10 already existed on main; definition-derived token values
  change without changing its kinds, roles or dtypes. Observation `KeywordData`
  remains a boolean surface distinct from sparse definition serialization.
  Jack Heart subsequently selected option 1a on 2026-11-02: remove the
  64-variant capacity limit within MTG-238. Inline word storage now grows
  with the declared parameterless vocabulary, preserving the set API and
  Copy semantics. Serde emits present enum variants in declaration order;
  append variants to preserve indices and learning-token identity. Controlled
  ordinal fixtures exercise word boundaries without inventing mechanics.
  Exact pre/post definition and learning-catalog bytes, content identity and
  authored replay evidence matched; source fingerprints changed. This
  storage-only extension adds no mechanic, observation column, Python API
  migration or world bump. Full-suite and performance acceptance remain
  separate from that focused compatibility evidence.
- Jack Heart selected MTG-238 first on 2026-11-02, superseding the T6 hold
  and stacking prerequisite: publish against main for review, no auto-merge;
  T6 integrates the crate move afterward. Main's T5 sync supplied Airship's
  namesake conversion. Source-relative names read layers 1–3 without asking
  the static index; a conditional type grant can be evaluated during index
  construction, so recursive characteristic reads are unsafe there.
- Concurrent engine vocabulary changes cross structural crate moves. Source
  identity for parity evidence must include the new rules and card crates.
  An unchanged Command tape/game outcome does not imply unchanged object ids:
  earthbend's continuous-effect records allocate ids. Broad proof belongs to
  its tested tree; it does not establish correctness after another engine sync.
- Crate moves and constructor consolidation also relocate live rules-audit authorities and behavioral-test
  anchors in `reviews.json` and its generated `index.jsonl`. Move both together
  while preserving rule dispositions, source spans/digests and frozen receipts;
  keep validation strict. PR #414's first Python Unit Tests run exposed this
  omitted consumer despite its earlier broad engine/Academy checks. The exact
  CI command, including infrastructure and deployment tests, covers a different
  boundary than the engine suite. T6's catalog consolidation exposed the same
  boundary at `Game::new_with_content`: the surviving authority is
  `Game::construct`, and generator defaults must move with reviewed anchors.
  Regeneration against the exact pinned source preserved all 3,322 dispositions,
  spans and hashes; repairing live references supplies no new rules coverage.
- Cache-preserving writes and generic continuous effects must compose: allocate
  timestamps through the typed ID accessor, return before a tracked mutation
  for an empty batch, and retain full invalidation for nonempty effect records.
  Empty batches preserve derived-cache revisions, but `record_continuous`
  still allocates a timestamp: they are not complete state/hash no-ops.
  This is the MTG-238/derived-state-cache merge's ownership invariant, not a
  new performance claim.

## Composable layers and successor inputs (2026-11-04; pending review)

- Jack Heart accepted MTG-244 as one layer-evaluation Task: Beorn/Kellan are
  real definitions, with historical interactions as rules fixtures. The one
  `Game::characteristics` authority owns type/color/ability/P/T composition and
  dependency ordering; adjacent execution and mana refactors retain their own
  responsibilities. Focused witnesses are not exhaustive Magic conformance.
- Jack Heart approved a new input version on 2026-11-04. The implementation
  selects explicit v8 while retaining fresh v6 and frozen saved versions.
  Native/Python parity alone cannot prove compatibility: identical arrays can
  both exceed a retained model's embedding vocabulary. New payloads need the
  selected version's model-forward and checkpoint proofs as well as shared
  pre-inference rejection on older inputs. Card admission stays definition-based.
- An activation offered and paid from layered abilities must resolve the
  captured program, including after source departure. A printed-definition
  lookup at resolution discarded a granted activation in the directed witness.
  Catalog provenance can compress observation input; it is not resolution
  authority. Pending program projection belongs alongside the retained frame.
- Program ownership is a model invariant as well as an engine invariant.
  A GRU spanning all local fact rows let one holder's program influence another.
  Independent live-program reduction fixes that counterexample and avoids
  recurrence on ordinary states. The bounded CPU component comparison reduced
  ordinary inference from 15.36 to 3.94 ms and two-program inference from 15.61
  to 4.76 ms (batch 16, width 64, 512 facts, ten samples, one thread). Exact
  receipts/configuration are retained in MTG-244's risk note at
  `043705783:scratch/mtg-244-risk-validation.md`. These are component costs,
  not full-policy performance, accelerator behavior or learning-quality results.
- Jack Heart also required risk measurement and attempted mitigation before
  review readiness. An anthem board does not exercise same-layer dependency
  ordering, and combining added games with the old step total conceals the
  ordinary-workload comparison. Separate workload results, retain complete
  action/score digests and counts, and attribute rebuild/dependency costs before
  choosing optimizations. Tap-state predicates require characteristic
  invalidation. The subsequent implementation (`b2fb06cba`) retains the raw board
  across mana writes unless a discovered layer condition reads mana, including
  nested conditions and newly granted statics. Finalized activation outputs and
  trigger indexes still refresh; debug recomputation remains the correctness
  oracle. Eight warmed-board pool-write probes fell from 471 to 10 allocations
  each, but this does not measure a complete payment or whole-game throughput.
  Neither a digest refresh nor a component speedup waives the 10% Engine Budget.
  The standard CI budget runs only the original TLA workloads; added dependency
  and policy harnesses require explicit execution and retained receipts.
- Exact numeric transport does not guarantee meaningful model separation.
  V8 integers 4096/8192 shared an embedding bucket and saturated the raw numeric
  projection to identical local-program outputs. Signed log magnitudes repair
  that counterexample in local/shared v8 reducers while retained versions keep
  their paths. Independent shared programs are batched without sharing recurrent
  state. Join/order/padding/gradient and real collector/checkpoint-selection
  checks support these mechanisms, not general injectivity or learning quality.
  Exact receipts and limitations are at
  `b2fb06cba:scratch/mtg-244-risk-validation.md`. Full-model comparative latency,
  final engine/search remeasurement and complete-payment attribution remain
  open; training outcomes need separately authorized experiments.

## Architecture convergence (accepted 2026-07-17)

- managym is the rules engine Manabot specifically needs, including
  authoritative match identity/revision, semantic Commands, committed events,
  replay, viewer Observations, and exact forks. Academy and manabot consume this
  authority; they do not reconstruct it.
- Observation is a central viewer protocol object composed from current
  visible state, ordered visible events, and the complete current semantic
  decision. managym preserves the lossless viewer stream; agent memory is an
  Intelligence concern.
- managym defines the viewer-relative possible-world domain, typed
  `WorldQuery` grammar, compatible-deal measure, and materializer. Intelligence
  owns probability and learned belief over that domain.
- The high-priority provider order is: match/Command authority, canonical
  Observation history, possible-world/query kernel, then a cross-consumer
  Academy → search → replay → Study proof. New card breadth and branch
  representations follow the exercised contracts.

## Ownership boundaries

- **Game:** authored presentation, replay UX, recovery, accessibility,
  packaging, and client adapters.
- **Rules:** authoritative content, match state, semantics, legal interaction,
  events, viewer-safe projection, forks, and deterministic execution.
- **Intelligence:** policy, search algorithms, teacher data, and evaluation.
- **Study:** human-facing retry, reveal, comparison, and branch navigation.

## TLA batches (T1–T5, 2026-10-28 to 2026-11-02)

- Jack Heart (2026-10-28/29) accepted the TLA delivery as seven card batches
  plus decklist import, authored as typed Rust `CardDefinition`s in their set
  file with support by definition (no qualification gate). Each merge requires
  Jack Heart's review; CI runs `cargo test` in debug, so release-only green is
  not evidence. `AGENTS.md` carries the standing authoring and observation rules.
- Jack Heart's T5 design review (2026-10-30, comment 10a31948): a permanent
  copy is layer-1 data (`Card.copy: CopyEffect { of, exceptions }`) whose
  exceptions share the continuous-effect `Modification` vocabulary, observed
  generically as fact rows and tokens (`FactKind::Copy`, `CopyException`), never
  as tensor-only identity bits. New open-ended state keeps following this shape.
- `ContinuationFrame.decision` means an obligation code only for kind-6 frames
  (trigger ordering class for 7/8, placement group for 9, delayed-trigger
  event for 10). `continuation.rs`
  owns `decision_code` and `MAX_DECISION_CODE`; the Python label table is
  tested against that bound. A text-only consumer gap of this kind is invisible
  to the tensor path and surfaced only in the headless LLM-path demo.
- Syncing a branch that was stacked on a since-squashed parent: merge with the
  parent's pre-squash head as the base so only the branch's own diff meets
  main (T5 met ~60 hunks instead of ~1465). Open w17 content changes required
  re-recording its mutable evidence in place; this is not permission to rewrite
  frozen worlds, checkpoints or historical receipts. Affected w17 evidence
  includes the semantic kernel, authored-match receipt and parity
  pins, public-commitment parity, Academy demos, choice navigation and
  trigger-order fixtures. A content-binding change can move hashes alone;
  a projection change can also move observed fields, so inspect the exact
  difference while retaining the intended Commands and state witnesses. A
  re-record that resolves a prefix by offer label must keep the recorded
  offer id when labels repeat (two copies of a card in hand), or the
  position silently changes copy, state digest and permanent order
  (`regen_choice_navigation.py`, fixed 2026-11-02). Until the
  Academy demos are re-recorded, `play_demo.py` rejects their digest before
  any decision is offered, so the Playwright decision-surface spec fails in
  CI as a symptom of the stale demo, not a UI defect.
- Jack Heart's T5 concept review (2026-11-02, decisions 1a/2a): legendary is
  a layered characteristic (`Characteristics.is_legendary`, layer 4: copiable
  supertypes unless a `Nonlegendary` record), read by the legend rule and the
  card row alike; no separate reader. Every `CardData` field that states a
  characteristic of a battlefield card reads `Game::characteristics()`, and
  `tests/cards/observability.rs` classifies each field as layered or printed
  by definition so an unclassified field does not compile. A characteristic
  without a layered value is added to `Characteristics`, never read from the
  printed card. The rule changed no column, token or protocol; the kernel,
  authored-match, demo and trigger-order receipts stayed green, but any
  recorded viewer projection whose card rows sit under active effects moves
  (the choice-navigation fixture's earthbent lands and anthems), and the
  projection's `base_power` stopped being a printed value: printed P/T has
  no other Python-visible source (the pack manifest carries ids and names
  only), so restoring it needs an engine accessor, a decision Jack Heart
  has not made.

## T6 lessons (2026-11-03; merged 2026-11-04 at `5dd11cec4`)

- Jack Heart accepted the one-catalog simplification into T6 on 2026-11-02:
  `ContentPack` owns stable name-ordered definition rows and content identity;
  authored presets are decklists, not separate worlds. Back-face rows follow
  their fronts. The refactor alone preserved content bytes at `2c1ba169f`;
  later card/rules changes intentionally moved the digest. Historical labels
  stay readable, but former authored-pack checkpoint manifests are not
  migrated, and no retained checkpoint binary has established compatibility.
- A shared consumer trace can agree while the rules are wrong. T6's first
  publication gate reproduced targets-before-payment and automatic named-card
  exile despite earlier green suites. Optional X payment must complete before
  a distinct reflexive trigger chooses targets within its retained budget and
  offers priority. A search's optional zero-through-four selection is a real
  native decision; matching the final state is insufficient evidence.
- Private zone search grants temporary physical visibility to its chooser,
  including nonmatching cards, while inspected hand definitions remain known
  afterward. Observation, text, Academy projection and both search samplers
  must consume that same permission. The searched library's owner acquires no
  extra visibility, and replay/Study redaction remains viewer-relative.
- Future behavior depends on history as well as the visible face: a transform
  count detects an intervening round trip; stacked inherited abilities retain
  their lending definition across source departure or replacement; departure
  triggers use LKI. Ordered extra turns/combats and their restrictions need
  both deterministic identity and shared policy/text observation. T6 represents
  them in v5/v6 frames/facts and definition-derived tokens, preserving v1–v4
  widths and exclusions. Hash contract 13 moved to 14 for delayed-object captures
  and 15 for independent lifetime fields; mutable receipts must follow those
  changes, while frozen evidence stays fixed.
- Jack Heart accepted Academy's public schedule completion in T6 on
  2026-11-03 (option 1a). Equal per-player counts can conceal opposite turn
  orders, and zero pending combats can conceal an active restriction. Native
  `TurnState.public_schedule` supplies ordered turns/resume and current/pending
  combats directly; policy continuation frames are a parallel projection, not
  a client scheduling API. Missing historical data is unknown. Current combat
  includes ordinary combat without a new persisted extra-combat flag. Native,
  Svelte and retained-session WebSocket proofs cover these distinctions; they
  establish neither browser geometry nor process-crash durability.
- Jack Heart's 2026-11-03 delayed-trigger correction separates event timing,
  captured subjects and effects. `CreateDelayedTrigger` resolves an optional
  selector immediately; exact `ObjectRef`s travel through the waiting trigger,
  occurrence, stack and suspended resolution as `Remembered`, separately from
  event context. Later incarnation changes cannot redirect the effect. Generic
  end-step timing retains controller-only versus any-player behavior; captured
  death watchers share the same path. Focused debug proofs cover independent
  captures, source departure/control change, reentry, forks and hidden subjects.
  Definition-derived tokens and viewer-safe native/Python/text projection carry
  the state. This changes content/hash identity, so old green receipts are not
  evidence for the corrected engine; frozen evidence remains untouched.
- Jack Heart accepted independent delayed-event matching, firing frequency and
  expiry on 2026-11-03 (option 1a), superseding the earlier open scope choice.
  `DelayedFrequency::{Once, EachMatch}` and `DelayedExpiry::{ThisTurn, Never}`
  compose with event predicates. A once-only watcher is consumed on its match,
  not successful resolution; cleanup removes waiting records, never existing
  occurrences. Exact-object death watchers retire on departure even with no
  stated expiry or repeating frequency. Captured references and event context
  retain separate identities and the original controller. V5/v6 relation kind 9
  carries frequency without changing widths; absent historical frequency stays
  unknown. Focused debug, hash/fork, native/Python/text/token and retained-session
  proofs support this slice, not crash persistence or combined-head acceptance.
  This correction introduces no general printed-trigger migration or second DSL.
- Main's v6 integration (`3b122de90`, merged into T6 at `bf7631c6a`) makes fact
  capacity configurable, fresh default 512, independently of T6's 512-action
  capacity. Frozen v5 remains 128 facts; omitted saved action capacity stays
  64. Timestamp rank normalization remains /128 for either fact capacity.
  Training's retained 140-fact failure demonstrates why deck admission and
  encoder capacity are separate questions; a larger capacity repairs that
  counterexample, not every reachable position. T6's focused v6 parity and
  retained-session proofs do not certify training results or combined-head CI.
- Concurrent vocabulary additions need explicit ordinal reconciliation.
  Main's library ordering keeps decision code 23 and loyalty/defense keep
  counter ordinals 7/8; unpublished T6 payment/search use 24/25, choose-to-exile
  uses 26 and lore appends at 9. Combined receipts inherit main's corrected
  135-Command trace, not the older 142-Command tape. Regeneration and focused
  parity at `64b79d895` do not establish full-suite or review acceptance.
- Referenced-row training avoids encoding unused catalog programs while
  retaining gradients; serving keeps the full cached table because subset GRU
  execution is not bitwise identical. Focused behavioral/gradient evidence
  does not prove retained-checkpoint compatibility, strength or a new runtime
  budget. Fresh action capacity is 512; omitted saved capacity remains 64,
  and overflow must fail explicitly rather than hide legal choices.
- Frozen architecture checks must reconstruct the recorded observation
  configuration through `ObservationConfig.from_checkpoint`, including its
  policy-input version, rather than using fresh defaults. T6's capacity change
  exposed this distinction: loading the original v5/64-action inputs preserves
  the retained architecture hash. Fresh-checkpoint tests separately expect 512;
  neither proof admits an old trained checkpoint under the changed world.
  A new save must also use the agent's actual observation space: the compound
  replay fixture supplied fresh 512-action metadata for an explicit 64-action
  agent, and receipt validation correctly rejected it. Saving
  `agent.observation_space` and checking configuration equality after the final
  sampler reload repairs the fixture without weakening validation. Both uniform
  and learned samplers have focused save/reload/payment-replay proof; this does
  not establish full-head CI or retained-checkpoint compatibility.

- Frozen experiment scheduling/analysis fixtures must isolate historical
  identity from live-runtime admission. T6's published CI exposed 27 pilot
  failures at the world guard despite passing focused gameplay proofs. An
  opt-in compatible-world fixture (also explicit in the fresh-process analysis
  CLI) exercises synthetic outcomes without games, training or allocation;
  separate unpatched checks retain rejection through frozen loading, run
  declaration and evaluation. Pin every declared input capacity: v3's v6/256
  facts contract also requires 64 actions, independent of fresh 512 defaults.
  Temporary changed-input fixtures exercise admission without rewriting frozen
  protocol/regime bytes. The repaired 33-case module passed at that stage;
  subsequent main integrates experiment protocol v5 with policy input v7,
  128 facts, 64 actions and 128 cards per player. Experiment protocol numbers
  and policy-input versions are separate contracts. Synthetic shape isolation
  must remain off encoding and gameplay paths; unpatched identity/schema
  rejection checks preserve strict admission. These are test-isolation proofs,
  not historical-runtime compatibility or full-head CI.
- Main's explicit v7 shared-effect projection (`fb5ad2c13`, integrated into
  T6 at `920bbb52a`) coexists with T6 schedule/capture/lifetime projections;
  fresh inputs remain v6. Constructors use the single catalog, and consumer
  parity must include the selected version rather than a global default shape.
  Focused native, Python/text and discovery checks passed after integration;
  pre-sync green receipts still do not certify the combined source or imply
  acceptance of unrelated experiment work.

## Player control and shared projections (2026-11-03/04; pending review)

- Jack Heart's final MTG-226 clarification makes `ActionSpace.player` the
  authoritative chooser. Native `Decision.player()` and `Action.player()`
  retain the rules subject and resource ownership; `TurnState.active_player`
  / `Game::active_player()` still name whose turn it is. Publication resolves
  control once from legal actions. Consumers must not remap that chooser,
  including when the controller is itself controlled. An extra `acting_player`
  identity was an assistant proposal, not an accepted requirement.
- Native inspection permission covers the controller's own information and
  the controlled player's permitted in-game information, including foretold
  cards. Private sideboards and outside-game selection remain excluded.
  Known hand definitions persist after control expires; physical private-card
  visibility does not. Policy, text, Academy and search share that authority.
  Pending/active controls and delegated decisions use relation kinds 13/14;
  they add no columns. Last-created active control wins for the rules subject.
- Main's v6 capacity change, integrated through T6, makes fresh fact capacity
  configurable (default 512); v5 remains 128 and timestamp rank remains /128.
  Saved observation configurations retain their version and omissions through
  `from_checkpoint`. Tests relying on fresh defaults now exercise v6; explicit
  v5 tests remain necessary evidence for that historical layout.
- Shared projection additions reach every generated schema that embeds the
  type, not only Experience. T7's replay, Study and advice schemas initially
  omitted `player_controls` while their runtime types already included it.
  Regeneration repairs this drift without rewriting historical fixtures;
  absent optional projection data remains unknown, distinct from an empty list.
- T7's focused native and submitted-60-card Academy/training proofs cover
  delegation, durations, payment, privacy and retained cleanup recovery. They
  do not establish all pairwise TLA interactions, process-crash durability,
  training strength or final-head CI. Catalog completeness and behavior
  witnesses remain separate claims in `docs/rules/tla/acceptance.md`.
- Jack Heart selected stable viewer perspective (option 1a) on 2026-11-03.
  “You” stays anchored to the viewer, including intervening own-player
  decisions during control. Inspection-container placement is permission,
  not ownership: private labels use native owner/zone, and a generic search
  heading cannot infer whose library is searched. Native instructions name
  whose choices and resources apply. The 2026-11-04 correction has both-seat,
  both-duration Islandcycling projection/fork/expiry and Svelte-rendering
  evidence; retained cleanup recovery remains separate from search recovery
  and browser geometry. The historical concept-review illustration predates
  implementation and is not a browser capture.

## Open tensions

- Migrate current Academy match/revision/replay bookkeeping into managym without
  a flag day or a permanent second authority.
- Replace action-position and duplicated offer/Command meaning one consumer at
  a time while keeping the curated matchup playable.
- Prove revision-by-revision live/headless/replay parity for the shipped
  authored trace without creating a parallel replay authority.
- Finish Study consumer evidence with fork/apply/return latency and peak RSS
  budgets plus focused multi-seed, nested, stale/pack, and privacy stress.
- Define a possible-world hypothesis quotient that is exact for the first hand
  range and can scale without making physical-copy order public meaning.
- Integrate the selected branch runtime into conditional Intelligence search
  and Study fork/return before optimizing another benchmark-only
  representation.
- Preserve dense-state speed as card semantics and event histories grow.
- Let the next content increment pressure the rules types naturally without
  turning each card into a one-off variant; count the variants that serve one
  card and compose rather than fuse (MTG-235 halved the fused set; remaining
  one-card verbs are the next candidates).
- Keep exactness non-negotiable without allowing proof infrastructure to become
  a substitute for playing the world.
