# Engine memory

Carried from the legacy `rules` wave (renamed Engine).

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

- Immutable versioned `ContentPack` definitions are separated from compact
  mutable match facts. Environments and branches share definitions without
  copying card content.
- Stable `CardDefId` values and deterministic state witnesses reproduce seeded
  traces without allocation-address or debug-format identity.
- Card content is the engine's own `CardDefinition` written in its rules
  types in `managym/src/cardsets/<set>.rs`, one function per card; the
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
  Same-runner comparison and native random-legal `Env::step` throughput remain
  interpretations awaiting Jack Heart's PR review; this throughput is not
  comparable with the retired rul-9 Python Command runner's 588.5 steps/s.
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
  reference now adopts that candidate's artifact with a reason and unchanged
  10% tolerance. Retaining the older, slower reference would allow loss of
  the optimization within the budget. This is curated native-workload evidence,
  not an end-to-end training result or shipping approval; the earlier accepted
  cost remains historical evidence.
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
