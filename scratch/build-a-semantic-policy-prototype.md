# Build a Semantic Policy Prototype in the Real Engine

## Problem

Etude has all three ingredients of a semantic manabot, but they have not yet
been joined into one learned policy:

- protocol-v1 `ExperienceFrame` values expose viewer-safe runtime facts and
  revision-bound `Command` values;
- the semantic learning pack exposes typed, fail-closed ability programs for
  visible objects; and
- `Env.structured_offers()` exposes uncapped `InteractionOffer` choices that
  Rust can decode and apply atomically.

The previous static experiments answered useful negative questions: bag
pooling cannot distinguish equal-token programs, and the first relational
pooling family was too slow and unreliable to nominate. They did not answer
whether a plausible structural model can learn actual engine decisions,
bind static roles to runtime objects, transfer to a held-out composition, or
play a bounded match. INT-2 answers those questions directly. It gives model
developers a runnable manabot and gives the Intelligence wave evidence from
real positions rather than another admission kata.

## The demo

Run:

```bash
./scripts/run-semantic-policy-prototype
```

From a fresh uv-managed checkout, the command builds the local managym
extension, compiles the prototype semantic pack, regenerates replayable
engine positions, trains three matched arms across three seeds, plays the
bounded micro-matchup, runs the held-out transfer and performance suites, and
prints the result artifact and report paths. The report shows actual
`ExperienceFrame` inputs and `Command` outputs, zero illegal applications over
the fixed evaluation (including 35 target candidates and 64 attacker
declarations), per-prompt competencies, held-out transfer, seat-balanced
arena results, and latency/throughput.

## Approach

### 1. Join the existing authority surfaces without inventing a second rules API

Add a reusable semantic-decision adapter in `manabot.semantic`:

1. Start from the acting player's real `managym.Observation`.
2. Build and validate a protocol-v1 `ExperienceFrame` using the existing
   viewer-safe projection and prompt metadata.
3. Obtain the exact `StructuredOfferSet` from Rust. Publish its typed offers in
   the frame, adding only the protocol's transitional `action_type` and
   `focus` presentation fields. Labels, help text, and card names are never
   model features.
4. Bind every visible card/permanent and every offer source/candidate subject
   to one runtime object row. Object subjects join by
   `(entity, incarnation)`; player subjects join by player index. Each visible
   card definition joins to its typed programs through `BoundSemanticPack`.
5. Score the ragged offer and candidate rows, use the existing
   `RaggedPolicyDecoder` to satisfy each role's `min`/`max`/`distinct`
   constraints, and wrap the selected IDs in a revision-bound protocol
   `Command`.
6. Lower only `Command.offer_id` and `Command.answers` to the existing
   `OfferSubmission`. `Env.step_structured()` decodes those IDs against the
   private bound offer set and applies the resulting `AtomicCommand`
   transactionally. Python never reconstructs a legal action.

The adapter fails before scoring when a visible definition, opcode, choice
kind, role binding, subject, or artifact identity is unadmitted. Raw offer and
candidate IDs are addresses only; they are never embeddings or ordinal
features. Training permutes offer/candidate row order while preserving IDs so
the model cannot learn engine enumeration order.

Extend the current narrow Rust structured-offer frontier only enough to make
the micro-matchup honest: add atomic `play_land` and zero-target `cast`
offers alongside the existing pass, single-target cast, and complete attacker
declaration paths. Attacker declarations are the admitted combat command in
this prototype. Blocker, payment, ordering, dependent-choice, and
mid-resolution prompts remain authoritative but use one fixed deterministic
executor shared by every arm. The report records the policy-controlled and
fallback decision counts separately; it never presents fallback behavior as a
learned competency.

### 2. Use a two-stage typed pointer Transformer

Implement one small architecture, `SemanticPointerPolicy`, with a target of
at most 160,000 trainable parameters:

- **Program stage:** a two-layer, pre-norm Transformer (`d_model=48`, four
  heads, feed-forward width 96, GELU, no dropout) encodes each typed program's
  explicit begin/end tokens, preorder position, and structural depth. A
  learned program token produces one embedding. The current maximum is 75
  semantic tokens, so the checked model limit is 76 including the program
  token. At inference, the 37-program catalog is encoded once per checkpoint
  and cached by checkpoint plus semantic-pack hash.
- **Decision stage:** a second two-layer Transformer of the same width attends
  over a global state token, both players, visible objects, offers, choice
  roles, and candidates. Runtime numeric facts come from the viewer-safe frame.
  Object tokens receive their linked program embedding. Offer and candidate
  tokens receive their linked source/subject object embedding plus typed verb,
  role, cardinality, zone, controller, and prompt features.
- **Pointer heads:** a masked offer head scores only published offers; a
  role-conditioned pointer head scores only candidates in that role. Required
  single selection uses cross-entropy. Attacker multi-selection uses binary
  cross-entropy and the existing decoder enforces cardinality. This retains an
  uncapped output dictionary instead of reintroducing the legacy 32-action
  ceiling.

This is intentionally a conventional, batched Transformer/pointer design.
The Transformer provides order-sensitive program encoding; attention over the
runtime object/offer set models interactions without making object-list order
meaningful. Pointer scoring is the established shape for variable-size output
dictionaries. The design follows the relevant properties of the
[Transformer](https://arxiv.org/abs/1706.03762),
[Pointer Networks](https://proceedings.neurips.cc/paper_files/paper/2015/hash/29921001f2f04bd3baee84a12e98098f-Abstract.html),
and attention over sets in the
[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html), while staying
small enough for rollout-adjacent CPU inference.

### 3. Generate authoritative, replayable supervision

Add a versioned workload and deterministic position generator with three
training families:

- **Priority:** land play, zero-target creature/spell casting, Lightning Bolt
  casting, and passing in scripted multi-decision situations.
- **Targeting:** Lightning Bolt positions where a deterministic continuation
  makes exactly one player or permanent target best. The fixed frontier
  includes 33 opposing permanents plus two players, so the selected target is
  one of 35 candidates.
- **Combat:** positions with two through six eligible attackers. The oracle
  enumerates at most 64 attacker subsets, applies each atomic declaration to a
  fork, follows one pinned deterministic blocker/priority continuation, and
  selects the lexicographically best terminal outcome, damage, then material
  result with a stable tie break.

The labeler is a deterministic engine oracle, not a Python rules
reimplementation. It enumerates published commands, applies them to exact
forks, and compares authoritative outcomes after a bounded scripted
continuation. Every position stores:

- world, engine/source/extension, protocol, content-pack, semantic-pack,
  workload, model-input, and structured-offer identities;
- constructor type, decklists, deal/position seeds, scenario operations or
  preceding command prefix, and pre-decision state digest;
- the validated viewer frame, structured offer projection, oracle scores,
  selected protocol `Command`, resulting digest, and outcome summary.

The replay verifier reconstructs each sampled position, checks frame and state
digests, reapplies the recorded command, and checks the resulting digest. Any
misclassified evaluation example can therefore be replayed as an exact engine
position and command rather than only as a tensor row.

Use a single shared dataset for all arms and seeds, split by position recipe
before any row augmentation. Cap the first run at 6,144 training positions,
1,536 in-distribution evaluation positions, and 768 held-out positions. Record
learning curves at 64, 256, 1,024, 4,096, and the full training set; do not
silently add examples after seeing results.

### 4. Put the ablations inside the runnable policy

Train exactly three arms with identical Transformer blocks, pointer heads,
hidden width, optimizer, examples, and model seeds:

1. `semantic_structured`: intact typed program tokens and structure;
2. `identity_structured`: opaque definition identity only, with every held-out
   definition mapped to an untrained `unknown_definition` slot; and
3. `semantic_structure_shuffled`: the same semantic token multiset, but a
   workload-seeded permutation removes preorder and all structural depths are
   neutralized.

All three modules instantiate both the semantic and identity input tables and
use a fixed arm gate, giving equal parameter counts rather than merely similar
headline widths. Inactive input parameters are reported separately from total
parameters. The decoder, dynamic state/offer encoder, losses, and training
budget are identical.

The model never consumes card/offer/candidate labels, help text, raw IDs, or a
target's source-list ordinal. Evaluation repeats with independent wire-row
permutations as a leakage canary.

### 5. Make the holdout a real composition, not an opcode quiz

Create a separate experiment semantic source derived from the reviewed
two-deck source and add the already-authoritative `Lightning Bolt` definition
and typed `deal_damage(target, 3)` spell program. Compile it to a separately
named prototype IR; do not broaden or rename the production
`two_deck.source.json` artifact.

Withhold **Igneous Inspiration** entirely from training positions. Training
contains Lightning Bolt, which supplies `deal_damage`, and Pop Quiz, which
supplies `learn`; the holdout combines `deal_damage + learn` in a single
targeted spell. A pre-run audit must prove:

- the Igneous Inspiration identity never appears in training frames;
- every symbolic primitive in its program appears in training programs;
- its normalized program structure is absent from training; and
- the held-out card remains admitted by the exact ContentPack and prototype
  semantic pack.

Evaluate zero-shot agreement/competency first, then limited retraining at 1,
4, 16, and 64 labeled held-out positions. The identity arm receives only its
unknown-definition slot in zero-shot evaluation. This makes the comparison
about semantic recombination and sample efficiency at a working policy
boundary, not direct opcode classification.

### 6. Play a bounded authoritative micro-matchup

Use a symmetric 40-card micro deck built from admitted basics, Fire Nation
Cadets, Lightning Bolt, Pop Quiz, and Igneous Inspiration. The intact/identity
in-distribution arena excludes Igneous Inspiration; the transfer arena swaps it
in under the frozen holdout contract. Structured policy decisions cover land
play, pass/cast priority, single-target casting, and complete attacker
declarations. The shared deterministic executor handles blockers and the
small number of unresolved mid-resolution prompts.

For each model seed, play a paired, deal-diverse block with each arm on each
seat against the identity baseline, using identical deal seeds. Report overall
and per-seat win rates with Wilson intervals, but treat the three model seeds
as the independent units for method claims. Also report the fraction of
surfaced decisions controlled by the learned structured policy; if the fixed
executor controls enough decisions to explain the arena result, the report
must say so and the result is not promoted as a strength claim.

### 7. One bounded command and one evidence bundle

`./scripts/run-semantic-policy-prototype` is the only documented end-to-end
entry point. It performs the documented uv/maturin build because a fresh
worktree does not contain `managym._managym`, then invokes the Python runner
through `uv run`. It writes a versioned artifact directory containing the
compiled prototype IR, dataset manifest and shards, three checkpoints per arm,
training histories, replay receipts, arena rows, performance samples, result
JSON, and the rendered Markdown report.

The workload budget is zero GPU-hours, at most four CPU workers, at most 2
wall-clock hours, and no retries or substituted seeds. A budget breach ends the
run with a recorded partial result; it does not expand the budget.

## De-risking

| Question | Finding | Impact on design |
|----------|---------|-----------------|
| Can a real viewer-safe frame carry the structured choice path? | `ExperienceFrame` already models typed choices and `Command.answers`. A live probe replaced the positional frame offers with the Rust structured projection, validated the frame and command, and atomically applied the command. The acting view exposed zero opponent hand cards and retained the hidden-card count. | Join the existing surfaces in a manabot adapter; no new protocol schema or client legality layer is needed. |
| Does the structured path really cross the 32-action boundary? | Existing evidence reaches 35 Lightning Bolt candidates and all 64 six-attacker subsets with zero illegal decodes. The live kickoff probe bound all 35 candidate subjects to visible frame rows and applied one command as one atomic engine action. | Keep ragged pointer scoring and the Rust decoder. Make both frontiers mandatory evaluation fixtures. |
| Are static programs available cheaply enough for real play? | W2-215 measured viewer projection at 9.875 us p50 / 15.416 us p95 and about 55.7k encode+batch observations/s at batch 256. The current pack has 31 definitions, 37 programs, 2,088 tokens, and a 75-token maximum program. | Reuse `BoundSemanticPack`; do not tokenize text or rebuild the catalog per decision. |
| Is the proposed model plausibly within the systems budget? | A local CPU microbenchmark of two 48-wide/two-layer Transformer stages measured the static 37x76 program catalog at 9.98 ms p50, the dynamic 1x80 pass at 0.244 ms p50 / 0.281 ms p95, and the 128x80 pass at 8.89k positions/s. The two Transformer stacks contained 75,840 parameters before input/head tables. | Cache the static catalog at inference and set conservative full-path gates of 2 ms batch-1 p95 and 4k positions/s at batch 128. Preserve raw samples in the final receipt. |
| Can the current semantic pack admit Lightning Bolt? | No. The production semantic source is intentionally the exact UR/GW two-deck slice; Lightning Bolt is authoritative engine content but absent from its 31 definitions, so the current projector correctly fails admission. | Compile a separately named experimental semantic source/IR that derives from the reviewed source and adds only Lightning Bolt. Production two-deck artifacts remain unchanged. |
| Can held-out composition be defined without an unknown primitive? | Yes. Lightning Bolt supplies `deal_damage`; Pop Quiz supplies `learn`; Igneous Inspiration combines `deal_damage + learn`. Igneous is already in the reviewed semantic source and ContentPack. | Withhold Igneous identity and positions, run primitive-closure and normalized-structure audits before training, and fail closed on any gap. |
| Can a fresh worktree run the experiment with one command? | `uv run python` created the venv but failed because `managym._managym` was absent. The documented `uv run --python 3.12 --extra play maturin develop --release --manifest-path managym/Cargo.toml --features python` build restored the runtime. | The shell entry point owns this exact uv/maturin preflight before invoking the Python runner. No bare Python command appears in docs or scripts. |
| Are search labels required for a useful first prototype? | No. The structured frontier has at most 35 target candidates and 64 admitted attack subsets, and exact forks plus deterministic continuations already exist. Search would add label cost and hidden-information ambiguity before the data seam is proven. | Use an enumerative deterministic engine oracle for the first supervised dataset. Add search only if an observed arena or competency failure makes label quality ambiguous. |
| Does a Transformer repeat the killed relational design? | No. The killed arm pooled one handcrafted relation-message family and was both hard to optimize and 11-14x slower than bag throughput. The selected model is a conventional batched sequence Transformer over explicit begin/end structure plus a separate runtime set encoder and pointer head. | Do not reuse `RelationalSemanticEncoder` or extend its static kata ladder. Keep the old katas as regression tests only. |

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Extend the first relational-pooling encoder | Reuses existing kata tensors and relation projections. | It ended `KILL_REDESIGN structural_capacity`, missed trainability across seeds, and ran at 0.071-0.088x bag model throughput. Continuing it contradicts both the task and wave memory. |
| Add semantics to the legacy fixed 32-action policy | Smallest model change and easiest PPO reuse. | It cannot represent the required 35 candidates or 64 attacker declarations, cannot emit atomic role-bound commands, and would preserve positional leakage. |
| Build one full graph Transformer over the entire engine state and semantic AST | Maximum expressivity and direct typed edges. | It couples the prototype to a new graph materialization format and pays dynamic graph cost before the simpler explicit program sequence/runtime set has failed. The current 75-token programs are small enough for a standard Transformer. |
| Integrate directly into the production Etude server and checkpoint villain | Produces an immediately player-facing bot. | It would mix the experiment with session lifecycle, UI stops, unsupported offer families, and release behavior. The reusable manabot adapter plus authoritative arena proves the policy boundary first; server integration follows evidence. |
| Generate labels with high-budget MCTS | Potentially stronger strategic targets. | It is expensive, inherits determinization/continuation caveats, and is unnecessary for uniquely scored bounded competencies. Deterministic engine-oracle labels isolate representation and binding first. |
| Add another semantic kata before integration | Cheaply measures structural discrimination. | The live ambiguity is end-to-end binding/learning/transfer, not static classification. Existing katas remain regressions; a new one is allowed only after a measured prototype failure names two explanations it would separate. |

## Key decisions

- **The unit is a semantic decision, not a card classifier.** Every example
  contains a real frame, typed offers, a protocol command, and exact engine
  replay data.
- **Use semantics without names.** Card and offer labels remain useful for the
  report but never enter the model. Identity is an explicit ablation channel,
  not an accidental leak.
- **Cache only immutable semantics.** Program embeddings may be cached after
  checkpoint load; runtime object, state, offer, and binding features are
  recomputed on every decision.
- **Keep output authority in Rust.** Python scores only rows the engine
  published. The Rust offer set resolves IDs, rejects stale/fabricated values,
  and owns atomic mutation.
- **Admit one real transfer composition.** Igneous Inspiration is held out as
  `deal_damage + learn`; no unseen primitive is waved through as transfer.
- **Use exact matched arms.** All arms have identical parameter tables and
  heads; only the fixed input gate changes.
- **Separate competency from strength.** Oracle agreement and prompt
  competencies establish learned behavior. Arena win rate is reported with
  seat and fallback coverage, not used alone to claim intelligence.
- **Do not hide failure behind the fallback.** If the structured policy controls
  too little of the arena or the semantic arm only wins through fixed actions,
  the result is a boundary finding, not a successful transfer claim.

Wild success is not merely a higher classifier score: the intact semantic arm
selects legal 35-way targets, composes the unseen Igneous program with little
or no retraining, and converts that competency into better bounded play while
remaining cheap enough for model-in-loop engine stepping. The reusable seam
then becomes the substrate for later search augmentation.

Wild failure is a superficially good arena number produced by label/order
leakage, the fixed executor, or repeated near-duplicate scenarios. Independent
row permutations, split-by-recipe data, string/ID exclusion, exact replays,
per-family competency, and policy-controlled-decision accounting make those
failure modes visible. If intact and shuffled semantics remain tied, the next
step is a focused diagnostic of the observed decision family, not a broader
static proof ladder.

## Scope

- In scope:
  - a small typed pointer Transformer and three matched arms;
  - a viewer-frame/runtime-offer/program binder;
  - atomic structured play-land, zero-target cast, single-target cast, pass,
    and attacker declarations;
  - deterministic engine-oracle data generation for priority, targeting, and
    attack decisions;
  - an experimental semantic source/IR adding Lightning Bolt without changing
    the production two-deck semantic artifact;
  - exact position/command replay, three-seed training, in-distribution and
    held-out evaluation, the bounded arena, latency/throughput measurement,
    and one end-to-end command;
  - regression use of the existing semantic and structured-decoder tests.
- Out of scope:
  - replacing the production PPO observation/action ABI;
  - learned blocker, payment, assignment, ordering, or dependent-choice
    decoding;
  - PPO, expert iteration, or search-augmented training;
  - new game rules, semantic interpreter behavior, or open-ended card
    admission;
  - production Etude server/checkpoint-villain integration;
  - a new static kata absent an observed and decision-relevant confound.

## Done when

The task is complete when the one-command demo finishes within budget and its
versioned result proves all of the following:

1. The input path validates real viewer-safe `ExperienceFrame` facts, binds
   every visible runtime object and offer role to admitted typed programs, and
   exposes no opponent private cards in replayed frames.
2. The output path emits protocol `Command` values and records zero decoder
   errors, illegal applications, stale/fabricated-ID acceptances, trace
   mismatches, or replay digest mismatches over the fixed evaluation.
3. The fixed frontier contains at least one 35-candidate target decision and
   one six-attacker decision representing all 64 subsets without enumeration
   in the model output.
4. `semantic_structured`, `identity_structured`, and
   `semantic_structure_shuffled` train on identical positions with identical
   parameter counts and at least three declared model seeds.
5. The report includes per-priority/target/attack competency, policy and
   candidate losses, command agreement, learning curves, and row-permutation
   sensitivity for every arm and seed.
6. The holdout audit proves zero Igneous Inspiration training exposure, full
   symbolic primitive closure, and novel normalized composition; zero-shot and
   1/4/16/64-example retraining results are reported for every arm.
7. The symmetric bounded matchup runs through normal engine execution with
   alternating seats and deal-diverse seeds. Win rate, Wilson intervals,
   per-seat results, and policy-versus-fallback decision counts are reported.
8. Full-path batch-1 inference p95 is at most 2 ms after program-cache warmup,
   batch-128 model throughput is at least 4,000 positions/s, and environment
   SPS plus legacy-equivalent actions/s are recorded. A missed performance gate
   is a reported prototype result, not permission to weaken it after the run.
9. Artifacts pin the w2 world, content, engine source and extension, protocol,
   observation, structured-offer, learning schema, semantic IR, workload,
   dataset, model source, checkpoint, seeds, and compute identities.
10. Focused Python tests run through uv, and any Rust change passes debug CI
    parity:

```bash
uv run --extra dev pytest -q \
  tests/semantic/test_semantic_policy.py \
  tests/sim/test_structured_policy.py
cargo fmt --check --manifest-path managym/Cargo.toml
cargo clippy --manifest-path managym/Cargo.toml --all-targets --all-features -- -D warnings
cargo test --manifest-path managym/Cargo.toml
```

These conditions advance the Intelligence measures that require a runnable
manabot over real positions, a semantic policy that emits atomic commands,
matched identity/structure ablations, real held-out composition, explicit
legality/competency/strength/systems evidence, and replayable versioned
artifacts.

## Measure

The runner records raw rows and summaries for:

| Dimension | Measure |
|-----------|---------|
| Legality and replay | decode/apply/replay failures; stale/fabricated-ID rejection; pre/post state and frame digest agreement |
| Competency | exact command agreement by priority, target, and attack family; offer accuracy; single-target accuracy; attacker micro/macro F1; row-permutation agreement |
| Learning | train/validation policy loss, candidate loss, agreement at 64/256/1,024/4,096/full examples, examples and optimizer steps to 80% and 90% competency |
| Transfer | zero-shot and 1/4/16/64-example Igneous agreement/competency, semantic-minus-identity and intact-minus-shuffled paired differences, per-seed spread |
| Arena | paired seat-balanced wins/draws/losses, overall and per-seat Wilson intervals, paired seed-block differences, learned/fallback/forced decision counts |
| Systems | model parameters/bytes, cold bind and catalog encode time, warm batch-1 p50/p95, batch throughput, environment SPS, legacy-equivalent actions/s, peak RSS |
| Provenance and cost | all artifact hashes, model/data/deal/row-permutation seeds, wall/CPU time by stage, worker count, peak memory, zero GPU-hours |

Existing baselines remain contextual rather than gates: the semantic projector
measured 9.875/15.416 us p50/p95; the synthetic structured path measured
116.4/191.8 us focused p50/p95 and 147,732 legacy-equivalent actions/s; the
kickoff model-only microbenchmark measured 0.244/0.281 ms batch-1 dynamic
p50/p95 and 8.89k positions/s at batch 128. The final report reruns full-path
measurements on checked-in code and does not reuse these exploratory numbers
as results.
